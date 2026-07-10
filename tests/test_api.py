"""Tests for FastAPI REST API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.api import app
import app.api as api_module


@pytest.fixture(autouse=True)
def ensure_model_loaded():
    """Ensure the model is loaded before tests."""
    if api_module.predictor is None or not api_module.predictor.is_loaded:
        from src.inference.predictor import RiskPredictor
        api_module.predictor = RiskPredictor()


class TestRootEndpoint:
    def test_root_returns_api_info(self):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "EduRisk AI"
        assert "docs" in data
        assert "predict" in data

    def test_root_has_version(self):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "version" in response.json()


class TestHealthEndpoint:
    def test_health_returns_status(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data


class TestPredictEndpoint:
    def test_predict_valid_input(self):
        client = TestClient(app)
        response = client.post("/predict", json={
            "gender": "Male",
            "age": 20,
            "academic_pressure": 3,
            "cgpa": 2.5,
            "study_satisfaction": 3,
            "sleep_duration": "7-8 hours",
            "dietary_habits": "Moderate",
            "work_study_hours": 6,
            "financial_stress": 2,
            "family_history": "No",
            "suicidal_thoughts": "No",
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "shap" in data
        assert data["prediction"] in [0, 1, 2]

    def test_predict_high_risk_input(self):
        client = TestClient(app)
        response = client.post("/predict", json={
            "gender": "Female",
            "age": 22,
            "academic_pressure": 5,
            "cgpa": 1.5,
            "study_satisfaction": 1,
            "sleep_duration": "Less than 5 hours",
            "dietary_habits": "Unhealthy",
            "work_study_hours": 10,
            "financial_stress": 5,
            "family_history": "Yes",
            "suicidal_thoughts": "Yes",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1, 2]
        assert "shap" in data
        assert "interpretation" in data["shap"]

    def test_predict_invalid_gender(self):
        client = TestClient(app)
        response = client.post("/predict", json={
            "gender": "Invalid",
            "age": 20,
            "academic_pressure": 3,
            "cgpa": 2.5,
            "study_satisfaction": 3,
            "sleep_duration": "7-8 hours",
            "dietary_habits": "Moderate",
            "work_study_hours": 6,
            "financial_stress": 2,
            "family_history": "No",
            "suicidal_thoughts": "No",
        })
        assert response.status_code == 422

    def test_predict_invalid_age(self):
        client = TestClient(app)
        response = client.post("/predict", json={
            "gender": "Male",
            "age": 15,
            "academic_pressure": 3,
            "cgpa": 2.5,
            "study_satisfaction": 3,
            "sleep_duration": "7-8 hours",
            "dietary_habits": "Moderate",
            "work_study_hours": 6,
            "financial_stress": 2,
            "family_history": "No",
            "suicidal_thoughts": "No",
        })
        assert response.status_code == 422

    def test_predict_missing_field(self):
        client = TestClient(app)
        response = client.post("/predict", json={
            "gender": "Male",
            "age": 20,
        })
        assert response.status_code == 422


class TestModelInfoEndpoint:
    def test_model_info_returns_features(self):
        client = TestClient(app)
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "feature_names" in data
        assert "n_features" in data
        assert "classes" in data
        assert data["n_features"] == 11
