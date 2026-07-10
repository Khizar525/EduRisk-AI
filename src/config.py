"""
Central configuration for EduRisk AI.

All paths, hyperparameters, feature definitions, and constants
are defined here as the single source of truth.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ── Project Root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Paths ──────────────────────────────────────────────────────
@dataclass
class Paths:
    """File and directory paths."""
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"
    notebooks: Path = PROJECT_ROOT / "notebooks"
    logs: Path = PROJECT_ROOT / "logs"

    # Specific files
    dataset_csv: Path = data_raw / "Student Depression Dataset.csv"
    best_model: Path = models / "best_model.pkl"
    scaler: Path = models / "scaler.pkl"
    encoders: Path = models / "encoders.pkl"
    feature_names: Path = models / "feature_names.pkl"
    prediction_log: Path = PROJECT_ROOT / "data" / "prediction_log.csv"


PATHS = Paths()


# ── Features ───────────────────────────────────────────────────
# Columns selected for modeling (matching original notebook)
COLS_TO_KEEP: List[str] = [
    "Gender",
    "Age",
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "Have you ever had suicidal thoughts ?",
    "Depression",
]

# Categorical columns requiring encoding
CATEGORICAL_COLS: List[str] = [
    "Gender",
    "Sleep Duration",
    "Dietary Habits",
    "Family History of Mental Illness",
    "Have you ever had suicidal thoughts ?",
]

# Numerical columns
NUMERICAL_COLS: List[str] = [
    "Age",
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
]


# ── Risk Engineering Thresholds ────────────────────────────────
# Preserved from original notebook — composite scoring for Risk_Level
RISK_THRESHOLDS = {
    "depression_weight": 3,
    "academic_pressure_weight": 2,
    "low_cgpa_weight": 2,
    "financial_stress_weight": 1,
    "suicidal_thoughts_weight": 3,
    "low_sleep_weight": 1,
    "high_cgpa_bonus": -2,
    "high_satisfaction_bonus": -1,
    "low_threshold": 1,      # score <= 1 → Low Risk (0)
    "medium_threshold": 4,   # score <= 4 → Medium Risk (1)
    # else → High Risk (2)
}


# ── Training ───────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Model training configuration."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    scoring: str = "accuracy"

    # Random Forest grid
    rf_param_grid: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    })

    # XGBoost grid
    xgb_param_grid: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    })

    # SVM config
    svm_kernel: str = "rbf"
    svm_probability: bool = True

    # MLP config
    mlp_hidden_layers: tuple = (64, 32)
    mlp_max_iter: int = 500
    mlp_early_stopping: bool = True

    # Optuna settings
    use_optuna: bool = False          # Set True to use Optuna instead of GridSearch
    optuna_n_trials: int = 50         # Number of Optuna trials per model
    optuna_timeout: int = 300         # Timeout in seconds per model


TRAINING = TrainingConfig()


# ── Class Labels ───────────────────────────────────────────────
RISK_LABELS: Dict[int, str] = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

RISK_LABELS_EMOJI: Dict[int, str] = {
    0: "✅ Low Risk",
    1: "✒  Medium Risk",
    2: "❌ High Risk",
}

RISK_ADVICE: Dict[int, str] = {
    0: "You seem to be in a healthy zone. Keep up the good habits!",
    1: "Moderate risk detected. Consider speaking with a faculty advisor or counselor.",
    2: "High risk detected. Please reach out to your university counselor as soon as possible.",
}


# ── Kaggle ─────────────────────────────────────────────────────
KAGGLE_DATASET: str = "hopesb/student-depression-dataset"
