# 📉 Customer Churn Prediction System

An end-to-end machine learning project that predicts customer churn using the Telco Customer Churn dataset.
The project includes data preprocessing, model comparison, feature analysis, and an interactive Streamlit app for real-time predictions.

---

## Overview

Customer churn is a critical problem for subscription-based businesses. 
This project builds a machine learning system to identify high-risk customers and provide actionable insights.

### Key Features
- End-to-end ML pipeline (preprocessing + model)
- Multiple model comparison (Logistic Regression, Random Forest, XGBoost)
- Threshold tuning to optimize F1-score
- Feature importance analysis
- Reduced-feature model for interpretability
- Interactive streamlit app for predictions

---

## Dataset

This project uses the **Telco Customer Churn dataset**

🔗 Download here:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

After downloading, place it in `/data`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-system.git
cd churn-prediction-system
```

Install dependencies:

`pip install -r requirements.txt`

## Project Structure

```
churn-prediction-system/
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## Modeling Approach

Models evaluated:

* Logistic Regression
* Random Forest
* XGBoost

**Final Model:** XGBoost with threshold tuning 

## Results

| Model | F1 | ROC-AUC | Precision | Recall |
| ----- | -- | ------- | --------- | ------ |
| Logistic Regression | 0.61 | 0.84 | 0.50 | 0.78 |
| Random Forest | 0.54 | 0.82 | 0.62 | 0.48 |
| XGBoost (tuned) | **0.62** | **0.84** | 0.56 | 0.7 |

- The final XGBoost model uses a tuned classification threshold (0.35) to balance precision and recall, improving F1-score compared to the default threshold.
- Cross-validation F1: ~0.58 (5-fold)
- Reduced-feature model (top 10 features): F1 ≈ 0.60

## Key Insights

* Contract type is the strongest predictor
    * Month-to-month customers are much more likely to churn
* Fiber optic customers show higher churn risk
* Lack of support/security services increases churn
* A small subset of features captures most predictive power

## Streamlit App

Run the app:

```
streamlit run app.py
```

Features:
* Input customer data
* Predict churn probability
* Display risk level (Low / Medium / High)
* Provide recommended retention actions

## How to Reproduce

Train the model:

```
python -m src.train
```

Evaluate the model:

```
python -m src.evaluate
```

Run the app:

```
streamlit run app.py
```

## Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

## Future improvements

* Hyperparameter tuning with GridSearchCV
* SHAP for model explainability
* Deployment (Docker / cloud hosting)
* Real-time API endpoint

## App Preview

![App Screenshot](/images/app.png)