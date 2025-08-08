import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import basic_preprocess

def main(model_path, encoder_path, scaler_path, csv_path, threshold, output_csv):
    # Load model, encoder, scaler
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    # Load data
    df = pd.read_csv(csv_path)

    # Keep SK_ID_CURR if present, else None (for output)
    ids = df['SK_ID_CURR'] if 'SK_ID_CURR' in df.columns else None

    # Check if TARGET column exists (for evaluation)
    has_target = 'TARGET' in df.columns

    # Preprocess but keep SK_ID_CURR for output
    df_preprocessed = basic_preprocess(df, drop_id=False)

    if has_target:
        y_true = df_preprocessed['TARGET']
        X = df_preprocessed.drop(columns=['TARGET', 'SK_ID_CURR'])
    else:
        y_true = None
        if 'SK_ID_CURR' in df_preprocessed.columns:
            X = df_preprocessed.drop(columns=['SK_ID_CURR'])
        else:
            X = df_preprocessed

    # Encode and scale features
    X_enc = encoder.transform(X)
    X_scaled = scaler.transform(X_enc)

    # Predict probabilities and labels
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Print evaluation metrics if TARGET is available
    if has_target:
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        auc = roc_auc_score(y_true, y_proba)
        print(f"ROC AUC: {auc:.4f}")

    # Prepare output dataframe
    output_df = pd.DataFrame()
    if ids is not None:
        output_df['SK_ID_CURR'] = ids
    output_df['PRED_PROBA'] = y_proba
    output_df['PRED_LABEL'] = y_pred
    if has_target:
        output_df['TARGET'] = y_true

    # Save predictions to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the trained model .joblib file")
    parser.add_argument("encoder_path", help="Path to the encoder .joblib file")
    parser.add_argument("scaler_path", help="Path to the scaler .joblib file")
    parser.add_argument("csv_path", help="Path to the test CSV file")
    parser.add_argument("threshold", type=float, help="Threshold for classification")
    parser.add_argument("output_csv", help="Path to save predictions CSV")
    args = parser.parse_args()

    main(args.model_path, args.encoder_path, args.scaler_path, args.csv_path, args.threshold, args.output_csv)
