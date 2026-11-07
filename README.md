# Telecom Customer Churn Prediction

Predicts whether a telecom customer will switch to another provider (churn), using logistic regression.

## Data

Two files, merged on customer: `customer_data.csv` and `churn_data.csv` (7,043 customers). The task uses 21 predictor variables.

## Approach

1. Merge and inspect the data
2. Prepare it: convert Yes/No fields to 0/1, one-hot encode multi-level categories, check outliers, impute missing values
3. Split train/test and scale features
4. Drop highly correlated dummy variables
5. Select features with RFE, refine the model with statsmodels p-values and VIF
6. Evaluate beyond accuracy: confusion matrix, sensitivity, specificity, ROC curve, and an optimal cutoff that balances sensitivity and specificity

## Files

| File | Purpose |
| --- | --- |
| `Telecom+Churn+Case+Study.ipynb` | Full analysis |
| `customer_data.csv`, `churn_data.csv` | Input data |

## Stack

Python, pandas, scikit-learn, statsmodels, matplotlib, seaborn
