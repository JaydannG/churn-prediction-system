import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data import load_data, split_data
from src.config import MODEL_PATH, BEST_THRESHOLD

def evaluate(data_path):
    df = load_data(data_path)
    _, X_test, _, y_test = split_data(df)

    pipeline = joblib.load(MODEL_PATH)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= BEST_THRESHOLD).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return results


if __name__ == "__main__":
    results = evaluate("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")