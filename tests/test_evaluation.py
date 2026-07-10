"""
Tests for evaluation module — metrics, error analysis, plots.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.evaluation.metrics import compute_metrics
from src.evaluation.error_analysis import analyze_errors
from src.evaluation.plots import (
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    plot_learning_curves, plot_calibration_curves, plot_model_comparison,
    plot_class_distribution, plot_model_radar,
)


class TestMetrics:
    """Tests for metric computation."""

    def test_compute_metrics_returns_dict(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        result = compute_metrics(y_true, y_pred, model_name="test")
        assert "Accuracy" in result
        assert "report" in result
        assert "confusion_matrix" in result

    def test_perfect_accuracy(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        result = compute_metrics(y, y, model_name="test")
        assert result["Accuracy"] == 1.0

    def test_low_accuracy(self):
        y_true = np.array([0, 0, 0, 1, 2])
        y_pred = np.array([1, 1, 1, 1, 1])
        result = compute_metrics(y_true, y_pred, model_name="test")
        assert result["Accuracy"] == 0.2

    def test_with_probabilities(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_prob = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        result = compute_metrics(y_true, y_pred, y_prob, model_name="test")
        assert "ROC-AUC" in result
        assert result["ROC-AUC"] > 0.9


class TestErrorAnalysis:
    """Tests for error analysis."""

    def test_analyze_errors_returns_dict(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 2, 0, 1])

        X_test = np.random.randn(5, 3)
        y_test = np.array([0, 1, 2, 1, 0])  # Last two are errors
        feature_names = ["f1", "f2", "f3"]

        result = analyze_errors(mock_model, X_test, y_test, feature_names, "test")

        assert "accuracy" in result
        assert "n_misclassified" in result
        assert "error_by_class" in result
        assert "top_confusions" in result

    def test_perfect_model_no_errors(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 2])

        X_test = np.random.randn(3, 2)
        y_test = np.array([0, 1, 2])
        feature_names = ["f1", "f2"]

        result = analyze_errors(mock_model, X_test, y_test, feature_names, "test")

        assert result["accuracy"] == 1.0
        assert result["n_misclassified"] == 0


class TestPlots:
    """Tests for plot generation (no errors = pass)."""

    def test_plot_class_distribution(self, tmp_path):
        y = np.array([0, 0, 0, 1, 1, 2])
        plot_class_distribution(y, save_path=tmp_path / "class_dist.png")
        assert (tmp_path / "class_dist.png").exists()

    def test_plot_model_comparison(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "Accuracy": [0.9, 0.85],
            "ROC-AUC": [0.92, 0.88],
            "CV Mean": [0.88, 0.83],
        }, index=["RF", "XGB"])
        plot_model_comparison(df, save_path=tmp_path / "comparison.png")
        assert (tmp_path / "comparison.png").exists()
