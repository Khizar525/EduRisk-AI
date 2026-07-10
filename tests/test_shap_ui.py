"""
Tests for SHAP explanations and Gradio UI functions.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.explainability.shap_utils import (
    _is_tree_model,
    _nice_feature_name,
    _extract_class_shap,
    _extract_single_row_shap,
    compute_local_explanation_with_interpretation,
    plot_local_waterfall,
    plot_probability_gauge,
)
from src.inference.logger import PredictionLogger


class TestSHAPHelpers:
    """Tests for internal SHAP helper functions."""

    def test_is_tree_model_random_forest(self):
        assert _is_tree_model("RandomForestClassifier") is True

    def test_is_tree_model_xgboost(self):
        assert _is_tree_model("XGBClassifier") is True

    def test_is_tree_model_svm(self):
        assert _is_tree_model("SVC") is False

    def test_is_tree_model_mlp(self):
        assert _is_tree_model("MLPClassifier") is False

    def test_nice_feature_name_known(self):
        assert _nice_feature_name("CGPA") == "Academic performance (CGPA)"
        assert _nice_feature_name("Sleep Duration") == "Sleep duration"

    def test_nice_feature_name_unknown(self):
        assert _nice_feature_name("unknown_feature") == "unknown_feature"

    def test_extract_class_shap_list(self):
        shap_list = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        result = _extract_class_shap(shap_list, 0)
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    def test_extract_class_shap_3d(self):
        # Shape: (samples, features, classes)
        shap_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = _extract_class_shap(shap_3d, 1)
        # class_idx=1 selects the second class column: [[2,4],[6,8]]
        np.testing.assert_array_equal(result, np.array([[2, 4], [6, 8]]))

    def test_extract_single_row_shap_list(self):
        shap_list = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
        result = _extract_single_row_shap(shap_list, 1)
        np.testing.assert_array_equal(result, np.array([4, 5, 6]))


class TestLocalExplanation:
    """Tests for local explanation with interpretation."""

    def test_returns_complete_dict(self):
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [np.array([[0.1, -0.2, 0.3]])]

        input_scaled = np.array([[1.0, 2.0, 3.0]])
        feature_names = ["f1", "f2", "f3"]

        result = compute_local_explanation_with_interpretation(
            mock_explainer, input_scaled, feature_names, prediction=0
        )

        assert "sorted_features" in result
        assert "shap_values" in result
        assert "interpretation" in result
        assert "risk_factors" in result
        assert "protective_factors" in result
        assert "top_risk" in result
        assert "top_protective" in result

    def test_interpretation_contains_prediction(self):
        mock_explainer = MagicMock()
        # List of 3 arrays (one per class), each with 1 sample and 3 features
        mock_explainer.shap_values.return_value = [
            np.array([[0.1, -0.1, 0.05]]),   # class 0
            np.array([[-0.2, 0.3, -0.1]]),   # class 1
            np.array([[0.5, -0.3, 0.1]]),     # class 2
        ]

        input_scaled = np.array([[1.0, 2.0, 3.0]])
        feature_names = ["Academic Pressure", "CGPA", "Sleep Duration"]

        result = compute_local_explanation_with_interpretation(
            mock_explainer, input_scaled, feature_names, prediction=2
        )

        assert "High Risk" in result["interpretation"]


class TestPlots:
    """Tests for new plot functions."""

    def test_waterfall_plot(self, tmp_path):
        features = ["Academic Pressure", "CGPA", "Sleep Duration"]
        values = [0.3, -0.2, 0.1]
        fig = plot_local_waterfall(features, values, 1, save_path=tmp_path / "waterfall.png")
        assert (tmp_path / "waterfall.png").exists()

    def test_probability_gauge(self, tmp_path):
        probs = {0: 0.2, 1: 0.5, 2: 0.3}
        fig = plot_probability_gauge(probs, 1, save_path=tmp_path / "gauge.png")
        assert (tmp_path / "gauge.png").exists()


class TestLogger:
    """Tests for prediction logger analytics."""

    def test_empty_log_analytics(self, tmp_path):
        log_path = tmp_path / "empty_log.csv"
        logger = PredictionLogger(log_path)
        analytics = logger.get_analytics()
        assert analytics["total"] == 0

    def test_log_and_read(self, tmp_path):
        log_path = tmp_path / "test_log.csv"
        logger = PredictionLogger(log_path)

        inputs = {"gender": "Male", "age": 20}
        result = {"risk_level": "Low Risk", "confidence": "85.0%"}

        logger.log(inputs, result)
        history = logger.get_history()
        assert len(history) == 1
        assert history[0]["Risk Level"] == "Low Risk"

    def test_export_csv(self, tmp_path):
        log_path = tmp_path / "test_log.csv"
        export_path = tmp_path / "export.csv"
        logger = PredictionLogger(log_path)

        inputs = {"gender": "Male", "age": 20}
        result = {"risk_level": "Low Risk", "confidence": "85.0%"}
        logger.log(inputs, result)

        exported = logger.export_csv(export_path)
        assert exported is not None
        assert export_path.exists()
