import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
from torch.utils.data import DataLoader
import csv

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])

r_state = 41

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

def get_tone_category(ita):
    if ita > 55:
        return 'Very Light'
    elif ita > 41:
        return 'Light'
    elif ita > 28:
        return 'Intermediate'
    elif ita > 10:
        return 'Tan'
    elif ita > -30:
        return 'Brown'
    else:
        return 'Dark'

target_ind = 1
sex_ind = 4
age_ind = 3

def data():
    metadata = pd.read_csv('./train-metadata.csv', low_memory=False)
    malignant_id = metadata[metadata["target"] == 1]["isic_id"].tolist()
    benign_id = metadata[metadata["target"] == 0].sample(n=len(malignant_id)*4, random_state=r_state)["isic_id"].tolist()
    ids = malignant_id + benign_id
    images = []
    labels = []
    data_list = []

    # print("-Creating Original CSV-")
    with h5py.File('./train-image.hdf5', 'r') as file:
        # with open("original_results.csv", mode="w", newline="") as csvFile:
            # writer = csv.writer(csvFile)
        for id in ids:
            # write to original_results.csv
            image_values = metadata[metadata["isic_id"] == id].values.tolist()[0]
            target, sex, age = image_values[target_ind], image_values[sex_ind], image_values[age_ind]
            # lstar = image_values[18]
            # bstar = image_values[12]
            lstar = float(metadata[metadata["isic_id"] == id]['tbp_lv_Lext'].values[0])
            bstar = float(metadata[metadata["isic_id"] == id]['tbp_lv_Bext'].values[0])
            # print(lstar, bstar)
            
            ita = np.arctan2(lstar - 50, bstar)*(180/np.pi)
            category = get_tone_category(ita)
            data_list.append([target, sex, age, ita, category])
            if id in file:
                raw_data = file[id][()]
                image = Image.open(io.BytesIO(raw_data))
                image_pil = Image.fromarray(np.array(image))
                label = metadata[metadata["isic_id"] == file[id].name[1:]].values.tolist()[0][1]
                images.append(image_pil)
                labels.append(label)

                if label == 1:
                    flipped_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
                    images.append(flipped_horizontal)
                    labels.append(label)

                    flipped_vertical = image.transpose(Image.FLIP_TOP_BOTTOM)
                    images.append(flipped_vertical)
                    labels.append(label)

                    rotated = image.rotate(180)
                    images.append(rotated)
                    labels.append(label)
            # writer.writerows(data_list)

    # print("-Finished Original CSV-")
    dataset = ImageDataset(images, labels, data_transforms)
    # Filter the dataset for label == 1 and label == 0
    label_1_count = len(list(filter(lambda x: x[1] == 1, dataset)))
    label_0_count = len(list(filter(lambda x: x[1] == 0, dataset)))

    # Print out counts of labels
    print(f"Number of images with label 1: {label_1_count}")
    print(f"Number of images with label 0: {label_0_count}")

    class_labels = set(dataset.labels)

    train_index, test_index = train_test_split(range(len(dataset)), test_size=0.2, random_state=r_state, stratify=labels)

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=32, shuffle=False, num_workers=4)
    return train_loader, test_loader, metadata, ids


def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):

    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Currently Using:", device)
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    for i, block in enumerate(model.features):
        if i >= 5:
            for param in block.parameters():
                param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
    
    best_test = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total

        if test_acc > best_test:
            best_test = test_acc
            torch.save(model.state_dict(), "skin_lesion_model_best.pth")
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return model

def save_plot(train_acc, train_loss, test_acc, test_loss):
    epochs = range(1, len(train_acc) + 1)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training and testing accuracy
    axs[0].plot(epochs, train_acc, label='Train Accuracy', color='blue')
    axs[0].plot(epochs, test_acc, label='Test Accuracy', color='green')
    axs[0].set_title('Training and Test Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Plot training and testing loss
    axs[1].plot(epochs, train_loss, label='Train Loss', color='red')
    axs[1].plot(epochs, test_loss, label='Test Loss', color='orange')
    axs[1].set_title('Training and Test Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig('training_testing_plot.png')
    

def learner_results(model, metadata, ids):
    model.load_state_dict(torch.load('skin_lesion_model_best.pth'))
    model.eval()
    device = next(model.parameters()).device
    data_list = []
    with h5py.File('train-image.hdf5', 'r') as file:
        with open("learner_results.csv", mode="w", newline="") as csvFile:
            writer = csv.writer(csvFile)
            for id in ids:
                if id in file:
                    raw_data = file[id][()]
                    image = Image.open(io.BytesIO(raw_data))
                    image_pil = Image.fromarray(np.array(image))
                    true_label = metadata[metadata["isic_id"] == file[id].name[1:]].values.tolist()[0][1]
                    transform = transforms.ToTensor()
                    transformed_image = transform(image_pil).unsqueeze(0)
                    transformed_image = transformed_image.to(device)
                    pred_label = None
                    with torch.no_grad():
                        raw_output = model(transformed_image)
                        pred_label = torch.argmax(raw_output, dim=1).item()

                    image_values = metadata[metadata["isic_id"] == id].values.tolist()[0]
                    lstar = float(metadata[metadata["isic_id"] == id]['tbp_lv_Lext'].values[0])
                    bstar = float(metadata[metadata["isic_id"] == id]['tbp_lv_Bext'].values[0])
                    
                    ita = np.arctan2(lstar - 50, bstar)*(180/np.pi)
                    category = get_tone_category(ita)
                    sex, age = image_values[sex_ind], image_values[age_ind]
                    wrong_pred = None
                    if pred_label == true_label:
                        wrong_pred = 0
                    else:
                        wrong_pred = 1

                    data_list.append([wrong_pred, sex, age, ita, category])

            writer.writerows(data_list)

def run():
    print("-Loading Images-")
    train_loader, test_loader, metadata, ids = data()
    print("-Finished Loading Images-")
    print("# of Training Images:", len(train_loader.dataset), "# of Test Images:", len(test_loader.dataset))
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    # print("-Training Images-")
    train_model(model, train_loader, test_loader)
    print("-Creating Results CSV, Making Predictions-")
    learner_results(model, metadata, ids)
    print("-Finished Results CSV-")
    
if __name__ == "__main__":
    run()
    # train_acc = [79.40, 89.34, 92.76, 95.71, 96.46, 95.98, 97.50, 97.38, 97.34, 97.14]
    # train_loss = [0.4412, 0.2665, 0.1833, 0.1202, 0.1033, 0.1065, 0.0724, 0.0803, 0.0757, 0.0781]
    # test_acc = [84.42, 85.37, 85.85, 89.35, 89.19, 89.03, 91.57, 90.46, 91.41, 90.94]
    # test_loss = [0.3798, 0.3582, 0.3612, 0.3170, 0.4237, 0.3304, 0.3380, 0.3496, 0.3116, 0.3318]
    # save_plot(train_acc, train_loss, test_acc, test_loss)







