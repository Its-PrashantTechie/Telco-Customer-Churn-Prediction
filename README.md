# Telco Customer Churn Prediction - EDA, Visualization & Deep Learning (MLP)

**Course:** U21ADP05 - Exploratory Data Analysis and Visualization (Assignment II, Case Study)
**Student:** Prashant Sharma Dhungana | **Roll No:** 24AD130 | **Dept:** AI & DS (AD-B), III Year
**Institution:** KPR Institute of Engineering and Technology

## Problem
Telecom companies lose revenue when customers leave (churn). This project explores *why* customers churn and builds a deep learning model to predict *who* will churn.

## Dataset
IBM Telco Customer Churn - 7,043 customers, 21 fields (demographics, services, billing, churn label).
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn (copy included in `data/`).

## What is inside
| Path | Description |
|---|---|
| `churn_analysis.ipynb` | Full code + outputs: loading, EDA, cleaning, 8 visualizations, statistical/distribution analysis, MLP, tuning, evaluation |
| `report/` | Project report (Word + PDF) |
| `figures/` | All generated charts (incl. interactive Plotly HTML) |
| `data/` | Dataset CSV |
| `model/churn_mlp.keras` | Trained model |
| `results.json` | All numeric results |

## Method summary
1. **Cleaning:** 11 blank `TotalCharges` (new customers, tenure 0) set to 0; no duplicates; no outliers by IQR rule.
2. **Preprocessing:** binary + one-hot encoding (40 features), standard scaling, stratified 70/15/15 split.
3. **EDA & stats:** histograms, boxplots, heatmap, scatter, pairplot, category churn rates, interactive chart; normality tests, Bernoulli/Binomial analysis, chi-square and Mann-Whitney tests.
4. **Model:** MLP (Dense 32-16, ReLU, Dropout 0.4, Sigmoid), Adam, class weights, early stopping, grid search of 12 hyper-parameter sets.
5. **Evaluation:** loss/accuracy vs epoch, confusion matrix, ROC-AUC, threshold analysis, permutation feature importance.

## Key results (test set)
| Model | AUC | Accuracy | Recall | F1 |
|---|---|---|---|---|
| MLP (tuned) | 0.839 | 0.746 | 0.762 | 0.615 |
| Logistic Regression | 0.845 | 0.747 | 0.779 | 0.621 |

**Findings:** churn (26.5%) is driven by short tenure, month-to-month contracts, fibre-optic service, electronic-check payment and missing security/support add-ons. Gender and phone service are not significant.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook churn_analysis.ipynb   # Run All (about 2-3 minutes)
```
