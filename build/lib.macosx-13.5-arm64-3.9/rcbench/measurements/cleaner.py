import numpy as np

def clean_data(df):
    return df.replace('nan', np.nan).dropna(axis=1, how='any').astype(float)
