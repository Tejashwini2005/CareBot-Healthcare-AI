
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def valid_payload():
    return {
        "Age": 35,
        "Gender": "Female",
        "Fever": 1,
        "Cough": 1,
        "Fatigue": 1,
        "Breathlessness": 0,
        "Chest_Pain": 0,
        "Nausea": 0,
        "Headache": 0,
        "BMI": 24.5,
        "BP_Systolic": 120,
        "BP_Diastolic": 80,
        "SpO2": 98,
        "Glucose_mg_dL": 95,
        "HbA1c": 5.4,
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_valid_input():
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_diagnosis"] in {
        "Anaemia",
        "Gastroenteritis",
        "Healthy",
        "Hypertension",
        "Influenza",
        "Migraine",
        "Pneumonia",
        "Type2Diabetes",
    }
    assert len(result["class_probabilities"]) == 8
    assert "disclaimer" in result


def test_predict_invalid_input():
    payload = valid_payload()
    payload["Age"] = 150

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_field():
    payload = valid_payload()
    del payload["Age"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
