import pandas as pd


df = pd.read_csv('./train-metadata.csv')


import numpy as np


df['ITA'] = np.arctan2(df['tbp_lv_Lext'] - 50 , df['tbp_lv_Bext'])*(180/np.pi)


df[['tbp_lv_Lext', 'tbp_lv_Bext', 'ITA']].head()
def categorize_ita(ita):
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


df['skin_tone_category'] = df['ITA'].apply(categorize_ita)


print(df[['tbp_lv_Lext', 'tbp_lv_Bext', 'ITA', 'skin_tone_category']].head())
