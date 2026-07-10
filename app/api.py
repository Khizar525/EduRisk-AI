"""
EduRisk AI — FastAPI REST API.

Provides a RESTful interface to the same prediction engine used by Gradio.
Run with: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predictor import RiskPredictor
from src.utils.validators import validate_prediction_inputs

# ── Predictor Singleton ────────────────────────────────────────

predictor: Optional[RiskPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global predictor
    predictor = RiskPredictor()
    if predictor.is_loaded:
        print(f"Model loaded: {predictor.model_name}")
    else:
        print("Warning: Model artifacts not found. Train the model first.")
    yield
    predictor = None


# ── App Setup ──────────────────────────────────────────────────

app = FastAPI(
    title="EduRisk AI",
    description="Student Academic Risk Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ─────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Input schema for risk prediction."""

    gender: str = Field(..., description="Gender (Male/Female)", examples=["Male"])
    age: int = Field(..., ge=17, le=30, description="Age (17-30)", examples=[20])
    academic_pressure: int = Field(..., ge=1, le=5, description="Academic pressure (1-5)", examples=[3])
    cgpa: float = Field(..., ge=0.0, le=4.0, description="CGPA (0.0-4.0)", examples=[2.5])
    study_satisfaction: int = Field(..., ge=1, le=5, description="Study satisfaction (1-5)", examples=[3])
    sleep_duration: str = Field(
        ...,
        description="Sleep duration category",
        examples=["7-8 hours"],
    )
    dietary_habits: str = Field(
        ...,
        description="Dietary habits (Healthy/Moderate/Unhealthy)",
        examples=["Moderate"],
    )
    work_study_hours: int = Field(..., ge=0, le=12, description="Daily work/study hours (0-12)", examples=[6])
    financial_stress: int = Field(..., ge=1, le=5, description="Financial stress (1-5)", examples=[2])
    family_history: str = Field(
        ..., description="Family history of mental illness (Yes/No)", examples=["No"]
    )
    suicidal_thoughts: str = Field(
        ..., description="History of suicidal thoughts (Yes/No)", examples=["No"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male",
                    "age": 20,
                    "academic_pressure": 4,
                    "cgpa": 2.2,
                    "study_satisfaction": 2,
                    "sleep_duration": "Less than 5 hours",
                    "dietary_habits": "Unhealthy",
                    "work_study_hours": 8,
                    "financial_stress": 4,
                    "family_history": "Yes",
                    "suicidal_thoughts": "Yes",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for risk prediction."""

    prediction: int = Field(..., description="Risk level (0=Low, 1=Medium, 2=High)")
    risk_level: str = Field(..., description="Human-readable risk level")
    risk_label: str = Field(..., description="Risk label without emoji")
    confidence: str = Field(..., description="Model confidence percentage")
    confidence_value: float = Field(..., description="Raw confidence value (0-1)")
    advice: str = Field(..., description="Recommendation based on risk level")
    probabilities: dict = Field(..., description="Class probabilities (percentage)")
    shap: dict = Field(..., description="SHAP explanation data")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    feature_count: int


# ── Endpoints ──────────────────────────────────────────────────


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — API info."""
    return {
        "name": "EduRisk AI",
        "version": "1.0.0",
        "description": "Student Academic Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded if predictor else False,
        model_name=predictor.model_name if predictor and predictor.is_loaded else "none",
        feature_count=len(predictor.feature_names) if predictor and predictor.feature_names else 0,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict student academic risk level.

    Returns risk classification (Low/Medium/High) with confidence,
    SHAP-based explanation, and personalized recommendations.
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.",
        )

    # Validate inputs
    is_valid, error_msg = validate_prediction_inputs(
        request.gender, request.age, request.academic_pressure,
        request.cgpa, request.study_satisfaction, request.sleep_duration,
        request.dietary_habits, request.work_study_hours,
        request.financial_stress, request.family_history,
        request.suicidal_thoughts,
    )

    if not is_valid:
        raise HTTPException(status_code=422, detail=error_msg)

    # Predict
    result = predictor.predict(
        gender=request.gender,
        age=request.age,
        academic_pressure=request.academic_pressure,
        cgpa=request.cgpa,
        study_satisfaction=request.study_satisfaction,
        sleep_duration=request.sleep_duration,
        dietary_habits=request.dietary_habits,
        work_study_hours=request.work_study_hours,
        financial_stress=request.financial_stress,
        family_history=request.family_history,
        suicidal_thoughts=request.suicidal_thoughts,
    )

    return PredictionResponse(
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        confidence_value=result["confidence_value"],
        advice=result["advice"],
        probabilities=result["probabilities"],
        shap=result["shap"],
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "model_name": predictor.model_name,
        "feature_names": predictor.feature_names,
        "n_features": len(predictor.feature_names),
        "classes": ["Low Risk", "Medium Risk", "High Risk"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
