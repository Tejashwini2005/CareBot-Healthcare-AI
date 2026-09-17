from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

TARGET = "Diagnosis"
DROP_COLUMNS = ["Patient_ID"]

train_df = pd.read_csv(DATA_DIR / "train.csv")

X_train = train_df.drop(columns=[TARGET] + DROP_COLUMNS)
y_train = train_df[TARGET]

categorical_features = ["Gender"]

numeric_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]

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

best_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=0.1,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

print("Training selected model...")
best_model.fit(X_train, y_train)

model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_model, model_path)

print(f"Best model saved to: {model_path}")
print("Configuration: Logistic Regression, C=0.1, solver=lbfgs")