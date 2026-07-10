"""
SHAP explainability utilities — global, local, and human-readable explanations.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches

import shap

from src.config import PATHS, RISK_LABELS


# ── SHAP Value Computation ─────────────────────────────────────

def compute_shap_values(
    model,
    X_test: np.ndarray,
    X_background: np.ndarray,
    feature_names: List[str],
    model_name: str,
    n_test_samples: int = 500,
    n_background_samples: int = 100,
) -> tuple:
    """
    Compute SHAP values for a model.

    Returns:
        Tuple of (shap_values, explainer, X_test_sub).
    """
    X_test_sub = X_test[:n_test_samples]
    background = shap.sample(X_background, min(n_background_samples, len(X_background)))

    if _is_tree_model(model_name):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sub)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_sub[:100])
        X_test_sub = X_test_sub[:100]

    return shap_values, explainer, X_test_sub


# ── Global Explanations ────────────────────────────────────────

def plot_shap_summary_bar(
    shap_values,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    class_idx: int = 2,
    save_path: Optional[Path] = None,
) -> None:
    """Plot SHAP summary bar chart for a specific class."""
    shap_vals = _extract_class_shap(shap_values, class_idx)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_vals,
        features=X_test[: shap_vals.shape[0]],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        color="#e74c3c",
    )
    plt.title(f"SHAP Feature Importance — {model_name} ({RISK_LABELS[class_idx]})", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_beeswarm(
    shap_values,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    class_idx: int = 2,
    save_path: Optional[Path] = None,
) -> None:
    """Plot SHAP beeswarm chart for a specific class."""
    shap_vals = _extract_class_shap(shap_values, class_idx)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_vals,
        features=X_test[: shap_vals.shape[0]],
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Beeswarm — {model_name} ({RISK_LABELS[class_idx]})", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(
    shap_values,
    X_test: np.ndarray,
    feature_names: List[str],
    feature_idx: int,
    model_name: str,
    class_idx: int = 2,
    save_path: Optional[Path] = None,
) -> None:
    """Plot SHAP dependence plot for a single feature."""
    shap_vals = _extract_class_shap(shap_values, class_idx)

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feature_idx,
        shap_vals,
        features=X_test[: shap_vals.shape[0]],
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Dependence — {feature_names[feature_idx]} ({model_name})", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Local Explanations ─────────────────────────────────────────

def compute_local_explanation(
    explainer,
    input_scaled: np.ndarray,
    feature_names: List[str],
    prediction: int,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Compute SHAP values for a single prediction.

    Returns:
        Tuple of (sorted_feature_names, shap_values, bar_colors).
    """
    shap_values = explainer.shap_values(input_scaled)
    sv_row = _extract_single_row_shap(shap_values, prediction)

    sorted_idx = np.argsort(np.abs(sv_row))
    sorted_vals = sv_row[sorted_idx]
    sorted_feats = [feature_names[i] for i in sorted_idx]
    bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in sorted_vals]

    return sorted_feats, sorted_vals, bar_colors


def compute_local_explanation_with_interpretation(
    explainer,
    input_scaled: np.ndarray,
    feature_names: List[str],
    prediction: int,
) -> Dict:
    """
    Compute SHAP values with human-readable interpretation.

    Returns:
        Dictionary with shap values, interpretation, and visualization data.
    """
    shap_values = explainer.shap_values(input_scaled)
    sv_row = _extract_single_row_shap(shap_values, prediction)

    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(sv_row))[::-1]  # descending
    sorted_vals = sv_row[sorted_idx]
    sorted_feats = [feature_names[i] for i in sorted_idx]

    # Generate human-readable interpretation
    interpretation = _generate_interpretation(sorted_feats, sorted_vals, prediction)

    # Colors for visualization
    bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in sorted_vals]

    # Risk factors (positive SHAP = increases risk)
    risk_factors = [
        {"feature": f, "impact": float(v), "direction": "increases risk"}
        for f, v in zip(sorted_feats, sorted_vals) if v > 0
    ]

    # Protective factors (negative SHAP = decreases risk)
    protective_factors = [
        {"feature": f, "impact": float(v), "direction": "decreases risk"}
        for f, v in zip(sorted_feats, sorted_vals) if v < 0
    ]

    return {
        "sorted_features": sorted_feats,
        "shap_values": sorted_vals.tolist(),
        "bar_colors": bar_colors,
        "interpretation": interpretation,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "top_risk": risk_factors[:3] if risk_factors else [],
        "top_protective": protective_factors[:3] if protective_factors else [],
    }


def _generate_interpretation(
    features: List[str], values: np.ndarray, prediction: int
) -> str:
    """Generate human-readable interpretation of SHAP values."""
    risk_label = RISK_LABELS[prediction]

    lines = [f"**Prediction: {risk_label}**\n"]
    lines.append("The model's decision was driven by:\n")

    # Top risk factors
    risk_factors = [(f, v) for f, v in zip(features, values) if v > 0]
    if risk_factors:
        lines.append("**Factors increasing risk:**")
        for fname, val in risk_factors[:3]:
            nice_name = _nice_feature_name(fname)
            strength = "strongly" if abs(val) > 0.3 else "moderately" if abs(val) > 0.1 else "slightly"
            lines.append(f"- {nice_name}: {strength} increases risk")

    # Top protective factors
    protective = [(f, v) for f, v in zip(features, values) if v < 0]
    if protective:
        lines.append("\n**Factors decreasing risk:**")
        for fname, val in protective[:3]:
            nice_name = _nice_feature_name(fname)
            strength = "strongly" if abs(val) > 0.3 else "moderately" if abs(val) > 0.1 else "slightly"
            lines.append(f"- {nice_name}: {strength} decreases risk")

    return "\n".join(lines)


def _nice_feature_name(name: str) -> str:
    """Convert feature name to human-readable format."""
    mapping = {
        "Academic Pressure": "Academic pressure level",
        "CGPA": "Academic performance (CGPA)",
        "Study Satisfaction": "Satisfaction with studies",
        "Work/Study Hours": "Daily study/work hours",
        "Financial Stress": "Financial stress level",
        "Sleep Duration": "Sleep duration",
        "Dietary Habits": "Dietary habits",
        "Gender": "Gender",
        "Age": "Age",
        "Family History of Mental Illness": "Family history of mental illness",
        "Have you ever had suicidal thoughts ?": "History of suicidal thoughts",
    }
    return mapping.get(name, name)


# ── Visualization Helpers ──────────────────────────────────────

def plot_local_waterfall(
    sorted_features: List[str],
    shap_values: List[float],
    prediction: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a waterfall-style visualization of local SHAP values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Take top 10 features
    n = min(10, len(sorted_features))
    feats = sorted_features[:n][::-1]
    vals = shap_values[:n][::-1]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    bars = ax.barh(range(n), vals, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([_nice_feature_name(f) for f in feats], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("SHAP Value → Increases Risk | ← Decreases Risk", fontsize=10)
    ax.set_title(f"Feature Contributions — {RISK_LABELS[prediction]}", fontsize=12, pad=10)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, vals)):
        x_pos = val + 0.01 if val > 0 else val - 0.01
        ha = "left" if val > 0 else "right"
        ax.text(x_pos, i, f"{val:.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")

    # Legend
    risk_patch = mpatches.Patch(color="#e74c3c", label="Increases Risk")
    safe_patch = mpatches.Patch(color="#3498db", label="Decreases Risk")
    ax.legend(handles=[risk_patch, safe_patch], loc="lower right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_probability_gauge(
    probabilities: Dict[str, float],
    prediction: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a probability gauge/donut chart."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    # Donut chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # Center text
    risk_label = RISK_LABELS[prediction]
    ax.text(0, 0.05, risk_label, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0, -0.15, f"{max(values):.1f}%", ha="center", va="center", fontsize=11, color="gray")

    # Legend
    ax.legend(
        wedges, labels,
        title="Risk Classes",
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Internal Helpers ───────────────────────────────────────────

def _is_tree_model(model_name: str) -> bool:
    """Check if model is tree-based."""
    tree_keywords = ["forest", "xgb", "gradient", "tree", "random"]
    return any(kw in model_name.lower() for kw in tree_keywords)


def _extract_class_shap(shap_values, class_idx: int) -> np.ndarray:
    """Extract SHAP values for a specific class from multi-class output."""
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    elif len(shap_values.shape) == 3:
        return shap_values[:, :, class_idx]
    else:
        return shap_values


def _extract_single_row_shap(shap_values, class_idx: int) -> np.ndarray:
    """Extract SHAP values for a single sample and class."""
    if isinstance(shap_values, list):
        return shap_values[class_idx][0]
    elif len(shap_values.shape) == 3:
        return shap_values[0, :, class_idx]
    else:
        return shap_values[0]
