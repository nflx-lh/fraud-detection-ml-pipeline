# scripts/run_training.py

import pandas as pd
import os
import mlflow
from glob import glob
from src.data_loader import load_gold_parquet
from src.data_splitter import time_based_split
from src.features import preprocess_features
from src.imbalance_handler import handle_imbalance
from src.model_trainer import train_logistic_regression_tuned, train_xgboost_tuned
from src.model_evaluator import evaluate_model

def main():
    # Configuration
    MONTHS = pd.date_range("2017-01-01", "2019-10-01", freq="MS").strftime("%Y_%m").tolist()
    CUTOFFS = {
        "oot1": "2018-11-01",
        "oot2": "2019-03-01",
        "oot3": "2019-07-01"
    }
    FEATURE_DIR = "/app/datamart/gold/feature_store"
    LABEL_PATH = "/app/datamart/gold/label_store/gold_labels.parquet"

    # Load and split data
    df = load_gold_parquet(FEATURE_DIR, LABEL_PATH, MONTHS)
    splits = time_based_split(df, date_col="date", target_col="is_fraud", cutoffs=CUTOFFS)

    for key in splits:
        splits[key] = (
            splits[key][0],
            splits[key][1].astype(str).str.lower().map({"no": 0, "yes": 1}).astype(int)
        )

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    X_oot1, y_oot1 = splits["oot1"]
    X_oot2, y_oot2 = splits["oot2"]
    X_oot3, y_oot3 = splits["oot3"]

    print("Date range in dataset:")
    print(df["date"].min(), "→", df["date"].max())

    # Logistic Regression pipeline
    print("[Step] Preprocessing features for Logistic Regression...")
    X_train_lr, lr_pipeline = preprocess_features(X_train, model_type="logistic", fit_pipeline=True)
    X_val_lr, _ = preprocess_features(X_val, model_type="logistic", fit_pipeline=False, pipeline=lr_pipeline)
    X_test_lr, _ = preprocess_features(X_test, model_type="logistic", fit_pipeline=False, pipeline=lr_pipeline)
    X_oot1_lr, _ = preprocess_features(X_oot1, model_type="logistic", fit_pipeline=False, pipeline=lr_pipeline)
    print("[Done] Feature preprocessing completed.")

    print("[Step] Handling imbalance with SMOTE...")
    X_train_lr, y_train_lr = handle_imbalance(X_train_lr, y_train, strategy="smote")
    print("[Done] SMOTE resampling completed.")

    # Set experiment and start training
    print("[Step] Starting MLflow experiment...")
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run(run_name="LogisticRegression"):
        print("[Step] Training logistic regression with Optuna tuning...")
        logreg_model = train_logistic_regression_tuned(X_train_lr, y_train_lr, X_val_lr, y_val, pipeline=lr_pipeline, X_raw_train=X_train)
        print("[Done] Logistic regression training completed.")

        print("[Step] Evaluating on test set...")
        evaluate_model(logreg_model, X_test_lr, y_test, model_name="LogReg", dataset_label="Test")
        print("[Done] Test set evaluation completed.")

    print("[ALL COMPLETE] Logistic regression pipeline executed successfully.")

    # XGBoost pipeline
    print("[Step] Preprocessing features for XGBoost...")
    X_train_xgb, _, xgb_input_example = preprocess_features(X_train, model_type="xgboost", return_sample=True)
    X_val_xgb, _ = preprocess_features(X_val, model_type="xgboost")
    X_test_xgb, _ = preprocess_features(X_test, model_type="xgboost")
    X_oot1_xgb, _ = preprocess_features(X_oot1, model_type="xgboost")
    print("[Done] Feature preprocessing completed.")

    print("[Step] Handling imbalance for XGBoost...")
    X_train_xgb, y_train_xgb = handle_imbalance(X_train_xgb, y_train, strategy="undersample")
    print("[Done] Imbalance handling completed.")

    print("[Step] Starting MLflow run for XGBoost...")
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run(run_name="XGBoost"):
        print("[Step] Training XGBoost with Optuna tuning...")
        xgb_model = train_xgboost_tuned(
            X_train_xgb, y_train_xgb, X_val_xgb, y_val, input_example=xgb_input_example
        )
        print("[Done] XGBoost training completed.")

        print("[Step] Evaluating XGBoost on test set...")
        evaluate_model(xgb_model, X_test_xgb, y_test, model_name="XGBoost", dataset_label="Test")
        print("[Done] Test set evaluation completed.")

        print("[Step] Ending MLflow run for XGBoost...")
        mlflow.end_run()
        print("[Done] MLflow run ended.")

    print("[ALL COMPLETE] XGBoost pipeline executed successfully.")

if __name__ == "__main__":
    main()
