import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
from sklearn.decomposition import PCA

TARGET_COL = "TARGET"

def basic_preprocess(df, drop_id=True):
    df = df.copy()
    # Conditionally drop SK_ID_CURR
    if drop_id and 'SK_ID_CURR' in df.columns:
        df.drop(columns=['SK_ID_CURR'], inplace=True)

    # Fill missing numeric with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical with mode
    for col in df.select_dtypes(exclude=np.number).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def encode_and_scale(X):
    encoder = OneHotEncoder(use_cat_names=True)
    X_enc = encoder.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    return X_scaled, encoder, scaler

def apply_pca(X_scaled, n_components=None):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca
