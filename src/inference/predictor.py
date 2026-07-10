"""
Prediction service — loads model and artifacts, makes predictions.
"""

import numpy as np
from typing import Dict, Optional, List

from src.config import PATHS, RISK_LABELS_EMOJI, RISK_ADVICE, RISK_LABELS
from src.preprocessing.pipeline import load_artifacts
from src.explainability.shap_utils import (
    compute_local_explanation,
    compute_local_explanation_with_interpretation,
)
from src.training.models import get_shap_compatible_model


class RiskPredictor:
    """Production prediction service for student academic risk."""

    def __init__(self):
        self.artifacts = load_artifacts()
        self.model = self.artifacts.get("model")
        self.scaler = self.artifacts.get("scaler")
        self.encoders = self.artifacts.get("encoders")
        self.feature_names = self.artifacts.get("feature_names")
        self._explainer = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    @property
    def model_name(self) -> str:
        """Get human-readable model name."""
        return type(self.model).__name__

    @property
    def explainer(self):
        """Lazy-load SHAP explainer."""
        if self._explainer is None:
            import shap
            model_type = type(self.model).__name__
            is_tree = any(
                kw in model_type.lower()
                for kw in ["forest", "xgb", "gradient", "tree"]
            )
            if is_tree:
                self._explainer = shap.TreeExplainer(self.model)
            else:
                background = shap.sample(
                    self.scaler.transform(np.zeros((1, len(self.feature_names)))),
                    50,
                )
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
        return self._explainer

    def predict(
        self,
        gender: str,
        age: int,
        academic_pressure: int,
        cgpa: float,
        study_satisfaction: int,
        sleep_duration: str,
        dietary_habits: str,
        work_study_hours: int,
        financial_stress: int,
        family_history: str,
        suicidal_thoughts: str,
    ) -> Dict:
        """
        Make a risk prediction with full explanation.

        Returns:
            Dictionary with prediction, probabilities, advice, and SHAP data.
        """
        # Encode inputs
        input_data = np.array([[
            self.encoders["Gender"].transform([gender])[0],
            age,
            academic_pressure,
            cgpa,
            study_satisfaction,
            self.encoders["Sleep Duration"].transform([sleep_duration])[0],
            self.encoders["Dietary Habits"].transform([dietary_habits])[0],
            work_study_hours,
            financial_stress,
            self.encoders["Family History of Mental Illness"].transform([family_history])[0],
            self.encoders["Have you ever had suicidal thoughts ?"].transform([suicidal_thoughts])[0],
        ]])

        # Scale
        input_scaled = self.scaler.transform(input_data)

        # Predict
        prediction = int(self.model.predict(input_scaled)[0])
        probs = self.model.predict_proba(input_scaled)[0]

        # Get SHAP explanation with interpretation
        shap_data = compute_local_explanation_with_interpretation(
            self.explainer, input_scaled, self.feature_names, prediction
        )

        return {
            "prediction": prediction,
            "risk_level": RISK_LABELS_EMOJI[prediction],
            "risk_label": RISK_LABELS[prediction],
            "confidence": f"{max(probs) * 100:.1f}%",
            "confidence_value": float(max(probs)),
            "advice": RISK_ADVICE[prediction],
            "probabilities": {
                RISK_LABELS_EMOJI[i]: round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            },
            "probabilities_raw": {i: float(p) for i, p in enumerate(probs)},
            "input_scaled": input_scaled,
            "shap": shap_data,
        }
