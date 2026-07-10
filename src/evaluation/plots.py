"""
Evaluation visualization — confusion matrices, ROC curves, PR curves,
calibration plots, learning curves, and model comparison charts.
"""

from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve as sk_learning_curve

from src.config import PATHS


# ── Confusion Matrix ───────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred, model_name: str, save_path: Optional[Path] = None
) -> None:
    """Plot and optionally save a confusion matrix."""
    from sklearn.metrics import confusion_matrix

    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── ROC Curves ─────────────────────────────────────────────────

def plot_roc_curves(
    models: dict, X_test, y_test, save_path: Optional[Path] = None
) -> None:
    """Plot ROC curves for all models (One-vs-Rest)."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    class_names = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    fig.suptitle("ROC Curves — One-vs-Rest (All Models)", fontsize=14, y=1.02)

    for ax, (name, info) in zip(axes, models.items()):
        model = info["model"]
        y_score = model.predict_proba(X_test)

        for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, cls_idx], y_score[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("False Positive Rate")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.legend(loc="lower right", fontsize=7)

    axes[0].set_ylabel("True Positive Rate")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Precision-Recall Curves ────────────────────────────────────

def plot_precision_recall_curves(
    models: dict, X_test, y_test, save_path: Optional[Path] = None
) -> None:
    """Plot Precision-Recall curves for all models (One-vs-Rest)."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    class_names = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Precision-Recall Curves — One-vs-Rest (All Models)", fontsize=14, y=1.02)

    for ax, (name, info) in zip(axes, models.items()):
        model = info["model"]
        y_score = model.predict_proba(X_test)

        for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, cls_idx], y_score[:, cls_idx]
            )
            ap = average_precision_score(y_test_bin[:, cls_idx], y_score[:, cls_idx])
            ax.plot(recall, precision, color=color, lw=2, label=f"{cls_name} (AP={ap:.2f})")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Recall")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.legend(loc="lower left", fontsize=7)

    axes[0].set_ylabel("Precision")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Learning Curves ────────────────────────────────────────────

def plot_learning_curves(
    models: dict, X_train, y_train, save_path: Optional[Path] = None
) -> None:
    """Plot learning curves for all models to diagnose bias/variance."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Learning Curves — Bias/Variance Diagnosis", fontsize=14, y=1.02)

    for ax, (name, info) in zip(axes, models.items()):
        model = info["model"]

        train_sizes, train_scores, val_scores = sk_learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color="#3498db")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color="#e74c3c")
        ax.plot(train_sizes, train_mean, "o-", color="#3498db", lw=2, label="Training")
        ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", lw=2, label="Validation")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Training Samples")
        ax.set_xlim([train_sizes[0], train_sizes[-1]])
        ax.set_ylim([0.5, 1.02])
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("Accuracy")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Calibration Plots ──────────────────────────────────────────

def plot_calibration_curves(
    models: dict, X_test, y_test, save_path: Optional[Path] = None
) -> None:
    """Plot calibration curves (reliability diagrams) for each class."""
    from sklearn.calibration import calibration_curve

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    class_names = ["Low Risk", "Medium Risk", "High Risk"]
    class_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    model_markers = ["o", "s", "^", "D"]

    n_models = len(models)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Calibration Curves — Reliability Diagrams", fontsize=14, y=1.02)

    for cls_idx, (cls_name, ax) in enumerate(zip(class_names, axes)):
        for marker, (name, info) in zip(model_markers, models.items()):
            model = info["model"]
            y_prob = model.predict_proba(X_test)[:, cls_idx]

            fraction_pos, mean_pred = calibration_curve(
                y_test_bin[:, cls_idx], y_prob, n_bins=8, strategy="uniform"
            )

            ax.plot(mean_pred, fraction_pos, marker=marker, lw=2,
                    label=f"{name}", alpha=0.8)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
        ax.set_title(cls_name, fontsize=11)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.legend(loc="upper left", fontsize=7)

    axes[0].set_ylabel("Fraction of Positives")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Model Comparison ───────────────────────────────────────────

def plot_model_comparison(comparison_df, save_path: Optional[Path] = None) -> None:
    """Plot bar chart comparing models across metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_df[["Accuracy", "ROC-AUC", "CV Mean"]].plot(
        kind="bar", ax=ax, colormap="Set2", edgecolor="white"
    )
    ax.set_title("Model Comparison — Accuracy, ROC-AUC, 5-Fold CV Mean", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Radar / Spider Chart ───────────────────────────────────────

def plot_model_radar(
    comparison_df, save_path: Optional[Path] = None
) -> None:
    """Plot radar chart comparing models across multiple metrics."""
    categories = list(comparison_df.columns)
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("Set2", len(comparison_df))

    for idx, (name, row) in enumerate(comparison_df.iterrows()):
        values = row.tolist()
        values += values[:1]  # close the polygon
        ax.plot(angles, values, "o-", lw=2, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — Radar Chart", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Class Distribution ─────────────────────────────────────────

def plot_class_distribution(y, save_path: Optional[Path] = None) -> None:
    """Plot target class distribution."""
    from collections import Counter

    counts = Counter(y)
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    values = [counts.get(i, 0) for i in range(3)]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

    for bar, count in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count:,}\n({count/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Class Distribution — Risk Level (Target Variable)", fontsize=13)
    ax.set_ylabel("Number of Students")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
