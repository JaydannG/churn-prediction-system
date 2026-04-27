import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET, RANDOM_STATE

def load_data(path):
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )