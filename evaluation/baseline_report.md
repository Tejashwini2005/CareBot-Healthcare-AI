# CareBot Healthcare AI — Baseline Evaluation Report

## 1. Objective

The objective of Milestone 2 is to build a baseline machine learning model, establish an evaluation benchmark, and document the initial model performance.

## 2. Dataset

The provided healthcare symptoms dataset contains:

- Total records: 800
- Features and fields: 17 columns
- Diagnosis classes: 8
- Records per diagnosis: 100

The diagnosis classes are:

1. Influenza
2. Hypertension
3. Type2Diabetes
4. Pneumonia
5. Healthy
6. Anaemia
7. Gastroenteritis
8. Migraine

The dataset is balanced, with 100 records for each diagnosis class.

## 3. Data Split

A stratified 70/15/15 split was used:

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 560 | 70% |
| Validation | 120 | 15% |
| Test | 120 | 15% |
| Total | 800 | 100% |

The random state was set to 42 to make the split reproducible.

## 4. Preprocessing

The following preprocessing steps were used:

- `Patient_ID` was removed because it is an identifier rather than a predictive feature.
- `Gender` was encoded using One-Hot Encoding.
- Numerical features were standardized using StandardScaler.
- Unknown categorical values are handled using `handle_unknown="ignore"`.

The preprocessing and model were combined into a single scikit-learn Pipeline.

## 5. Baseline Model

The baseline classifier is **Logistic Regression**.

Configuration:

- Maximum iterations: 1000
- Random state: 42
- Multiclass classification
- Preprocessing and classification combined in one Pipeline

## 6. Evaluation Metrics

The following metrics were used:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Weighted One-vs-Rest AUC

These metrics provide a baseline for comparing future improvements.

## 7. Validation Results

| Metric | Score |
|---|---:|
| Accuracy | 0.8083 |
| Precision | 0.8115 |
| Recall | 0.8083 |
| F1 Score | 0.8068 |
| AUC | 0.9744 |

## 8. Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.8000 |
| Precision | 0.8130 |
| Recall | 0.8000 |
| F1 Score | 0.7989 |
| AUC | 0.9756 |

## 9. Test Classification Results

| Diagnosis | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Anaemia | 0.71 | 0.67 | 0.69 | 15 |
| Gastroenteritis | 0.85 | 0.73 | 0.79 | 15 |
| Healthy | 0.70 | 0.93 | 0.80 | 15 |
| Hypertension | 0.60 | 0.80 | 0.69 | 15 |
| Influenza | 0.92 | 0.73 | 0.81 | 15 |
| Migraine | 0.73 | 0.53 | 0.62 | 15 |
| Pneumonia | 1.00 | 1.00 | 1.00 | 15 |
| Type2Diabetes | 1.00 | 1.00 | 1.00 | 15 |

## 10. Baseline Summary

The Logistic Regression baseline achieved:

- 80.00% test accuracy
- 81.30% weighted precision
- 80.00% weighted recall
- 79.89% weighted F1 score
- 0.9756 weighted one-vs-rest AUC

The baseline provides a reproducible benchmark for future model development and improvement.

## 11. Model Artifact

The trained model and preprocessing pipeline were saved together as:

`models/baseline_model.joblib`

## 12. Data Privacy

Only synthetic/project-provided data is used for this implementation. No real patient data is used.

## 13. Next Steps

Future work will focus on:

- Improving the baseline model where appropriate
- Evaluating clinical information extraction using the synthetic clinical notes
- Extracting diagnosis, medications, and follow-up information
- Producing structured JSON output
- Developing the clinical note summarization component
- Evaluating the final pipeline against the project acceptance criteria