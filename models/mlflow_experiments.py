from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

TARGET = "Diagnosis"
DROP_COLUMNS = ["Patient_ID"]

X_train = train_df.drop(columns=[TARGET] + DROP_COLUMNS)
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET] + DROP_COLUMNS)
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET] + DROP_COLUMNS)
y_test = test_df[TARGET]

# Feature types
categorical_features = ["Gender"]
numeric_features = [
    column for column in X_train.columns
    if column not in categorical_features
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# MLflow experiment
mlflow.set_experiment("CareBot-Primary-Model-Tuning")


def run_experiment(C, solver):
    """Train and evaluate one Logistic Regression configuration."""

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        # Validation
        val_predictions = model.predict(X_val)
        val_probabilities = model.predict_proba(X_val)

        # Test
        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)

        # Metrics
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_precision = precision_score(
            y_val,
            val_predictions,
            average="weighted",
            zero_division=0,
        )
        val_recall = recall_score(
            y_val,
            val_predictions,
            average="weighted",
            zero_division=0,
        )
        val_f1 = f1_score(
            y_val,
            val_predictions,
            average="weighted",
            zero_division=0,
        )
        val_auc = roc_auc_score(
            y_val,
            val_probabilities,
            multi_class="ovr",
            average="weighted",
        )

        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0,
        )
        test_recall = recall_score(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0,
        )
        test_f1 = f1_score(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0,
        )
        test_auc = roc_auc_score(
            y_test,
            test_probabilities,
            multi_class="ovr",
            average="weighted",
        )

        # Log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", 1000)

        # Log validation metrics
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("val_auc", val_auc)

        # Log test metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_auc", test_auc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("\nExperiment completed")
        print(f"C       : {C}")
        print(f"Solver  : {solver}")
        print(f"Val F1  : {val_f1:.4f}")
        print(f"Val AUC : {val_auc:.4f}")
        print(f"Test F1 : {test_f1:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Run ID  : {mlflow.active_run().info.run_id}")


# Experiment 1
run_experiment(C=0.1, solver="lbfgs")

# Experiment 2
run_experiment(C=1.0, solver="lbfgs")

# Experiment 3
run_experiment(C=10.0, solver="lbfgs")

print("\nAll MLflow experiments completed successfully.")