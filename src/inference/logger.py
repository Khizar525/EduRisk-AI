"""
Prediction logging — timestamped CSV logging with analytics.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.config import PATHS


class PredictionLogger:
    """Logs predictions to a CSV file with timestamps and provides analytics."""

    COLUMNS = [
        "Timestamp", "Gender", "Age", "CGPA", "Academic Pressure",
        "Study Satisfaction", "Work/Study Hours", "Sleep Duration",
        "Dietary Habits", "Financial Stress", "Family History",
        "Suicidal Thoughts", "Risk Level", "Confidence",
    ]

    def __init__(self, log_path: Path = None):
        self.log_path = log_path or PATHS.prediction_log
        self._ensure_file()

    def _ensure_file(self):
        """Create log file with headers if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def log(self, inputs: Dict, result: Dict) -> None:
        """Log a prediction."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inputs.get("gender", ""),
                inputs.get("age", ""),
                inputs.get("cgpa", ""),
                inputs.get("academic_pressure", ""),
                inputs.get("study_satisfaction", ""),
                inputs.get("work_study_hours", ""),
                inputs.get("sleep_duration", ""),
                inputs.get("dietary_habits", ""),
                inputs.get("financial_stress", ""),
                inputs.get("family_history", ""),
                inputs.get("suicidal_thoughts", ""),
                result.get("risk_level", ""),
                result.get("confidence", ""),
            ])

    def get_history(self, n: int = 100) -> List[Dict]:
        """Read the last N predictions from the log."""
        if not self.log_path.exists():
            return []

        with open(self.log_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        return rows[-n:]

    def get_analytics(self) -> Dict:
        """Get analytics from prediction history."""
        if not self.log_path.exists():
            return {"total": 0, "risk_distribution": {}, "avg_confidence": 0}

        df = pd.read_csv(self.log_path)
        if df.empty:
            return {"total": 0, "risk_distribution": {}, "avg_confidence": 0}

        total = len(df)

        # Risk level distribution
        risk_dist = df["Risk Level"].value_counts().to_dict() if "Risk Level" in df.columns else {}

        # Average confidence
        if "Confidence" in df.columns:
            df["Confidence_num"] = df["Confidence"].str.rstrip("%").astype(float, errors="ignore")
            avg_conf = float(df["Confidence_num"].mean()) if not df["Confidence_num"].isna().all() else 0
        else:
            avg_conf = 0

        # Predictions over time (by date)
        if "Timestamp" in df.columns:
            df["Date"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.date
            daily_counts = df.groupby("Date").size().to_dict()
        else:
            daily_counts = {}

        return {
            "total": total,
            "risk_distribution": risk_dist,
            "avg_confidence": round(avg_conf, 1),
            "daily_predictions": {str(k): v for k, v in daily_counts.items()},
            "recent": self.get_history(5),
        }

    def export_csv(self, export_path: Path) -> Path:
        """Export prediction log to a specified path."""
        if self.log_path.exists():
            df = pd.read_csv(self.log_path)
            df.to_csv(export_path, index=False)
            return export_path
        return None
