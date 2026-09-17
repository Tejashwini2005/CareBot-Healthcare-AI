from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
MODEL_DIR = BASE_DIR / "models"

# Load datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Target column
TARGET = "Diagnosis"

# Remove ID because it is not a useful predictive feature
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

# Baseline model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

# Train
print("=== CareBot Baseline Model ===")
print("Training model...")
model.fit(X_train, y_train)

# Validation prediction
val_predictions = model.predict(X_val)
val_probabilities = model.predict_proba(X_val)

# Test prediction
test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)

# Evaluation function
def evaluate_model(name, y_true, predictions, probabilities):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    auc = roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr",
        average="weighted",
    )

    print(f"\n{name} Results")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    return accuracy, precision, recall, f1, auc


val_metrics = evaluate_model(
    "Validation",
    y_val,
    val_predictions,
    val_probabilities,
)

test_metrics = evaluate_model(
    "Test",
    y_test,
    test_predictions,
    test_probabilities,
)

# Detailed test classification report
print("\nTest Classification Report")
print("-" * 30)
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)

# Save model
MODEL_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "baseline_model.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
print("\nBaseline model training completed successfully!")