# decodelabs-project2-fraud-detection
# Fraud Detection Pipeline — Supervised Learning

## Overview
This project builds and tunes a classification model to detect fraudulent credit card transactions in a highly imbalanced dataset. It was completed as Project 2 of the Data Science Industrial Training Kit at DecodeLabs (2026 batch).

## Dataset
[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle — 284,807 transactions, of which only 0.17% are fraudulent. Download the dataset from Kaggle and place `creditcard.csv` in the project root before running the notebook.

## Problem
With fraud making up just 0.17% of transactions, accuracy is a misleading metric — a model that predicts "legitimate" every single time would still score 99.83% accuracy while catching zero fraud. This project instead optimizes for **Precision**, **Recall**, and **ROC-AUC**.

## Approach
- **Train/test split first** (stratified, 80/20) — before any scaling or resampling, to prevent data leakage
- **SMOTE** (Synthetic Minority Over-sampling) applied only within the training fold, using `imblearn.pipeline.Pipeline` instead of `sklearn.pipeline.Pipeline`, so resampling never touches the test set
- Two models trained and tuned with `GridSearchCV`:
  - **Logistic Regression** (with `StandardScaler`, since it's scale-sensitive)
  - **Random Forest** (no scaling needed — tree splits are scale-invariant)
- Models evaluated on the untouched test set using confusion matrix, precision, recall, F1-score, and ROC-AUC

## Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Recall (fraud caught) | 0.918 (90/98) | 0.888 (87/98) |
| Precision | 0.059 | 0.515 |
| F1-score | 0.111 | 0.652 |
| ROC-AUC | 0.972 | **0.985** |
| False positives | 1,433 | 82 |

## Conclusion
Logistic Regression caught slightly more fraud cases but generated 1,433 false positives — impractical for production, as it would block a large number of legitimate transactions. **Random Forest** offered a much stronger overall balance: 87 of 98 fraud cases caught, only 82 false positives, and the highest ROC-AUC (0.985). It is the recommended model for this task.

## Tech Stack
- Python, pandas, scikit-learn
- imbalanced-learn (SMOTE)
- Google Colab

## How to Run
1. Download `creditcard.csv` from the Kaggle link above
2. Open the notebook in Google Colab
3. Upload the dataset when prompted
4. Run all cells in order
