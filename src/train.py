import joblib
import os

from src.data import load_data, split_data
from src.preprocessing import build_pipeline
from src.config import MODEL_PATH

def train(data_path):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, X_test, y_test


if __name__ == "__main__":
    train("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"Model saved to {MODEL_PATH}")