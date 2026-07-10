"""
Hyperparameter tuning — GridSearchCV and Optuna.

GridSearchCV is the default (deterministic, reproducible).
Optuna is available for deeper search (Bayesian optimization).
"""

import warnings
from typing import Dict, Any, Tuple, Optional, Callable

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator
import numpy as np

from src.config import TRAINING

# Suppress Optuna's verbose logging
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")


def tune_grid_search(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "model",
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Tune a model using GridSearchCV.

    Args:
        model: Unfitted model instance.
        param_grid: Parameter grid for search.
        X_train: Training features.
        y_train: Training labels.
        model_name: Name for logging.

    Returns:
        Tuple of (best_model, best_params, best_cv_score).
    """
    if not param_grid:
        model.fit(X_train, y_train)
        print(f"  {model_name}: Fitted (no tuning)")
        return model, {}, 0.0

    print(f"  {model_name}: Running GridSearchCV...")

    n_splits = min(TRAINING.cv_folds, min(len(y_train), len(set(y_train))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=TRAINING.random_state)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=TRAINING.scoring,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    print(f"  {model_name}: Best params = {grid_search.best_params_}")
    print(f"  {model_name}: Best CV score = {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_optuna(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_space_fn: Callable,
    model_name: str = "model",
    n_trials: int = 50,
    random_state: int = 42,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Tune a model using Optuna (Bayesian hyperparameter optimization).

    Args:
        model_class: The model class (e.g., RandomForestClassifier).
        X_train: Training features.
        y_train: Training labels.
        param_space_fn: Function(trial) -> dict of parameters.
        model_name: Name for logging.
        n_trials: Number of Optuna trials.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (best_model, best_params, best_cv_score).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_splits = min(TRAINING.cv_folds, min(len(y_train), len(set(y_train))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = param_space_fn(trial)
        params["random_state"] = random_state

        model = model_class(**params)

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=TRAINING.scoring)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = random_state

    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)

    print(f"  {model_name}: Optuna best CV = {study.best_value:.4f}")
    print(f"  {model_name}: Best params = {best_params}")

    return best_model, best_params, study.best_value


# ── Optuna Parameter Spaces ────────────────────────────────────

def rf_param_space(trial) -> dict:
    """Random Forest parameter space for Optuna."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 15, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def xgb_param_space(trial) -> dict:
    """XGBoost parameter space for Optuna."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "eval_metric": "mlogloss",
    }


def svm_param_space(trial) -> dict:
    """SVM parameter space for Optuna."""
    return {
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "probability": True,
    }


def mlp_param_space(trial) -> dict:
    """MLP parameter space for Optuna."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"n_units_{i}", 32, 128, step=32))
    return {
        "hidden_layer_sizes": tuple(layers),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "max_iter": 500,
        "early_stopping": True,
    }


# Registry of Optuna parameter spaces
OPTUNA_SPACES = {
    "Random Forest": rf_param_space,
    "XGBoost": xgb_param_space,
    "SVM": svm_param_space,
    "MLP": mlp_param_space,
}
