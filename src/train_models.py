import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from utils import load_csv, save_model, split_data
from preprocess import basic_preprocess, encode_and_scale, apply_pca, TARGET_COL

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, output_dir):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    print(f"{model_name} AUC: {auc:.4f}")
    save_model(model, os.path.join(output_dir, f"{model_name}.joblib"))
    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_components", type=int, default=100)
    parser.add_argument("--n_estimators", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_csv(args.data_path)
    df = basic_preprocess(df)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Encode and scale
    X_scaled, encoder, scaler = encode_and_scale(X)

    # Save encoder and scaler
    joblib.dump(encoder, os.path.join(args.output_dir, "encoder.joblib"))
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    models = {
        "logistic_regression_no_pca": LogisticRegression(max_iter=500),
        "random_forest_no_pca": RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1),
        "xgboost_no_pca": XGBClassifier(n_estimators=args.n_estimators, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    }

    for name, model in models.items():
        train_and_evaluate(X_train, X_test, y_train, y_test, name, args.output_dir)

    # PCA version
    X_pca, pca = apply_pca(X_scaled, args.n_components)

    # Save PCA
    joblib.dump(pca, os.path.join(args.output_dir, "pca.joblib"))

    X_train_pca, X_test_pca, y_train, y_test = split_data(X_pca, y)

    models_pca = {
        "logistic_regression_pca": LogisticRegression(max_iter=500),
        "random_forest_pca": RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1),
        "xgboost_pca": XGBClassifier(n_estimators=args.n_estimators, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    }

    for name, model in models_pca.items():
        train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test, name, args.output_dir)
