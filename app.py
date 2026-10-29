
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="CareBot Healthcare AI API",
    description="API for the CareBot diagnosis classification model.",
    version="1.0.0",
)


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int = Field(ge=0, le=120)
    Gender: str = Field(min_length=1, max_length=30)
    Fever: int = Field(ge=0, le=1)
    Cough: int = Field(ge=0, le=1)
    Fatigue: int = Field(ge=0, le=1)
    Breathlessness: int = Field(ge=0, le=1)
    Chest_Pain: int = Field(ge=0, le=1)
    Nausea: int = Field(ge=0, le=1)
    Headache: int = Field(ge=0, le=1)
    BMI: float = Field(gt=0, le=100)
    BP_Systolic: float = Field(gt=0, le=300)
    BP_Diastolic: float = Field(gt=0, le=200)
    SpO2: float = Field(gt=0, le=100)
    Glucose_mg_dL: float = Field(gt=0, le=1000)
    HbA1c: float = Field(ge=0, le=30)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(features: PatientFeatures):
    input_data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]
    class_probabilities = {
        str(label): round(float(probability), 4)
        for label, probability in zip(model.classes_, probabilities)
    }

    return {
        "predicted_diagnosis": str(prediction),
        "class_probabilities": class_probabilities,
        "disclaimer": (
            "Educational demonstration only. "
            "This model is not a medical diagnosis."
        ),
    }
