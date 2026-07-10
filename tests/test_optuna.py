"""
Tests for Optuna tuning module.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.training.tuner import (
    tune_grid_search, tune_optuna,
    rf_param_space, xgb_param_space, svm_param_space, mlp_param_space,
    OPTUNA_SPACES,
)


@pytest.fixture
def small_dataset():
    """Small dataset for fast testing."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.random.choice([0, 1, 2], 200)
    return X, y


class TestGridSearch:
    """Tests for GridSearchCV tuning."""

    def test_grid_search_returns_model_and_params(self, small_dataset):
        X, y = small_dataset
        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

        best_model, best_params, best_score = tune_grid_search(
            model, param_grid, X, y, "RF"
        )

        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)

    def test_grid_search_no_params(self, small_dataset):
        X, y = small_dataset
        model = RandomForestClassifier(random_state=42)

        best_model, best_params, best_score = tune_grid_search(
            model, {}, X, y, "RF"
        )

        assert best_params == {}
        assert best_model is not None


class TestOptuna:
    """Tests for Optuna tuning."""

    def test_optuna_returns_model_and_params(self, small_dataset):
        X, y = small_dataset

        best_model, best_params, best_score = tune_optuna(
            model_class=RandomForestClassifier,
            X_train=X,
            y_train=y,
            param_space_fn=rf_param_space,
            model_name="RF",
            n_trials=5,
            random_state=42,
        )

        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert best_score > 0

    def test_param_spaces_registered(self):
        assert "Random Forest" in OPTUNA_SPACES
        assert "XGBoost" in OPTUNA_SPACES
        assert "SVM" in OPTUNA_SPACES
        assert "MLP" in OPTUNA_SPACES

    def test_rf_param_space_returns_dict(self):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        params = rf_param_space(trial)
        assert isinstance(params, dict)
        assert "n_estimators" in params
