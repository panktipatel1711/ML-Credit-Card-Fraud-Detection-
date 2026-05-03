import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Removes duplicates and drops unnecessary columns.
    """
    df = df.drop_duplicates()
    if 'Time' in df.columns:
        df = df.drop(['Time'], axis=1)
    return df

def scale_features(df, target_col='Class'):
    """
    Standardizes the 'Amount' feature for better model performance.
    """
    scaler = StandardScaler()
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    if 'Amount' in X.columns:
        X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    return X, y, scaler