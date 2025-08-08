import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import sys
import streamlit as st
st.write(f"Python executable: {sys.executable}")

MODEL_DIR = "models"

def list_joblib_files(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.endswith(".joblib")]

st.title("Credit Risk Scoring Dashboard")

st.sidebar.title("Select model and preprocessing files")

model_files = list_joblib_files(MODEL_DIR)
encoder_files = list_joblib_files(MODEL_DIR)
scaler_files = list_joblib_files(MODEL_DIR)

if not model_files:
    st.sidebar.error(f"No model files found in '{MODEL_DIR}' folder.")
if not encoder_files:
    st.sidebar.error(f"No encoder files found in '{MODEL_DIR}' folder.")
if not scaler_files:
    st.sidebar.error(f"No scaler files found in '{MODEL_DIR}' folder.")

model_choice = st.sidebar.selectbox("Model file", model_files)
encoder_choice = st.sidebar.selectbox("Encoder file (OneHotEncoder)", encoder_files)
scaler_choice = st.sidebar.selectbox("Scaler file (StandardScaler)", scaler_files)

model_path = os.path.join(MODEL_DIR, model_choice) if model_choice else None
encoder_path = os.path.join(MODEL_DIR, encoder_choice) if encoder_choice else None
scaler_path = os.path.join(MODEL_DIR, scaler_choice) if scaler_choice else None

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

if model_path and encoder_path and scaler_path:
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading model or preprocessing objects: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Upload CSV for Scoring", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        y_true = df["TARGET"] if "TARGET" in df.columns else None
        X = df.drop(columns=["TARGET"], errors='ignore')

        try:
            X_enc = encoder.transform(X)
            X_scaled = scaler.transform(X_enc)
            X_final = X_scaled
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        y_proba = model.predict_proba(X_final)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        if y_true is not None:
            st.write("Precision:", precision_score(y_true, y_pred))
            st.write("Recall:", recall_score(y_true, y_pred))
            st.write("ROC AUC:", roc_auc_score(y_true, y_proba))

        st.write("Predictions:", y_pred)
else:
    st.info("Please select model, encoder, and scaler files from the sidebar.")
