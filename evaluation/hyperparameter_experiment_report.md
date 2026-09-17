# CareBot Healthcare AI — Hyperparameter Experiment Report

## 1. Objective

The objective of this experiment is to train the primary classification model using multiple hyperparameter configurations, track the experiments using MLflow, and establish evidence for selecting the best configuration.

## 2. Primary Model

The primary model used for experimentation is Logistic Regression.

The same preprocessing pipeline from Milestone 2 was used for all experiments:

- `Patient_ID` removed
- Numerical features standardized using `StandardScaler`
- `Gender` encoded using `OneHotEncoder`
- Preprocessing and model combined using a scikit-learn Pipeline
- Maximum iterations: 1000
- Random state: 42

## 3. Hyperparameter Configurations

Three configurations were evaluated by changing the Logistic Regression `C` parameter.

| Experiment   |    C | Solver |
| ------------ | ---: | ------ |
| Experiment 1 |  0.1 | lbfgs  |
| Experiment 2 |  1.0 | lbfgs  |
| Experiment 3 | 10.0 | lbfgs  |

All experiments used the same training, validation, and test datasets.

## 4. MLflow Experiment Tracking

Experiment name:

`CareBot-Primary-Model-Tuning`

Each configuration was logged as a separate MLflow run.

| Configuration | MLflow Run ID                      |
| ------------- | ---------------------------------- |
| C=0.1         | `7c5992d4572d4ecea797a3ad8f5c8e75` |
| C=1.0         | `3f0833713c3149d2989750f144e4c1e9` |
| C=10.0        | `af20e4f9f40d49c0be9d5e9f2ec771b0` |

The logged information includes model parameters and validation/test performance metrics.

## 5. Experimental Results

|    C | Validation F1 | Validation AUC |    Test F1 |   Test AUC |
| ---: | ------------: | -------------: | ---------: | ---------: |
|  0.1 |    **0.8307** |         0.9724 | **0.8151** |     0.9735 |
|  1.0 |        0.8068 |     **0.9744** |     0.7989 | **0.9756** |
| 10.0 |        0.7910 |         0.9752 |     0.8084 |     0.9748 |

## 6. Best Configuration

Validation F1-score was selected as the primary model-selection metric.

Based on the validation results, the best configuration was:

- Model: Logistic Regression
- C: **0.1**
- Solver: **lbfgs**
- Validation F1: **0.8307**
- Validation AUC: **0.9724**
- Test F1: **0.8151**
- Test AUC: **0.9735**
- MLflow Run ID: `7c5992d4572d4ecea797a3ad8f5c8e75`

This configuration achieved the highest validation F1-score among the three tested configurations.

## 7. Evidence for Selection

The validation F1-scores were:

- C=0.1 → 0.8307
- C=1.0 → 0.8068
- C=10.0 → 0.7910

Therefore, C=0.1 produced the highest validation F1-score.

The selected configuration also achieved a test F1-score of 0.8151 on the held-out test set.

The results demonstrate that changing the regularization parameter affected model performance, providing an empirical basis for selecting the configuration.

## 8. Reproducibility

All experiments used:

- The same training/validation/test split
- Random state: 42
- The same preprocessing pipeline
- The same model family
- The same maximum iteration setting

MLflow was used to record the experiment parameters and evaluation metrics.

## 9. Conclusion

Three Logistic Regression hyperparameter configurations were successfully trained and tracked using MLflow.

The configuration with `C=0.1` achieved the highest validation F1-score of 0.8307 and was selected as the best configuration based on the predefined validation metric.

The experiment results provide a documented benchmark for subsequent model development and comparison.

## 10. Data Privacy

Only synthetic/project-provided data was used during these experiments. No real patient data was used.
