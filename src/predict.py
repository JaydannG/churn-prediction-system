import joblib
import pandas as pd

from src.config import MODEL_PATH, BEST_THRESHOLD


def predict_churn(customer_data):
    pipeline = joblib.load(MODEL_PATH)

    customer_df = pd.DataFrame([customer_data])

    churn_probability = pipeline.predict_proba(customer_df)[:, 1][0]
    churn_prediction = int(churn_probability >= BEST_THRESHOLD)

    return {
        "churn_probability": churn_probability,
        "churn_prediction": churn_prediction
    }