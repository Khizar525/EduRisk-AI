"""
Error analysis — identifying and categorizing misclassifications.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.metrics import classification_report

from src.config import RISK_LABELS


def analyze_errors(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "model",
) -> Dict:
    """
    Comprehensive error analysis for a trained model.

    Returns:
        Dictionary with error statistics and misclassified samples.
    """
    y_pred = model.predict(X_test)

    # Overall accuracy
    correct = (y_pred == y_test)
    accuracy = correct.mean()

    # Find misclassified samples
    misclassified_idx = np.where(~correct)[0]
    n_misclassified = len(misclassified_idx)

    # Error breakdown by true class
    error_by_class = {}
    for cls in [0, 1, 2]:
        cls_mask = y_test == cls
        cls_errors = (~correct) & cls_mask
        error_by_class[RISK_LABELS[cls]] = {
            "total": int(cls_mask.sum()),
            "errors": int(cls_errors.sum()),
            "error_rate": float(cls_errors.sum() / cls_mask.sum()) if cls_mask.sum() > 0 else 0,
        }

    # Confusion pairs — most common misclassifications
    confusion_pairs = []
    for idx in misclassified_idx:
        true_cls = int(y_test[idx])
        pred_cls = int(y_pred[idx])
        confusion_pairs.append((RISK_LABELS[true_cls], RISK_LABELS[pred_cls]))

    # Count confusion pairs
    from collections import Counter
    pair_counts = Counter(confusion_pairs)
    top_confusions = pair_counts.most_common(5)

    # Feature analysis of misclassified samples
    feature_error_importance = {}
    if n_misclassified > 0 and len(feature_names) == X_test.shape[1]:
        correct_features = X_test[correct]
        error_features = X_test[misclassified_idx]

        for i, fname in enumerate(feature_names):
            correct_mean = np.mean(np.abs(correct_features[:, i]))
            error_mean = np.mean(np.abs(error_features[:, i]))
            feature_error_importance[fname] = {
                "correct_mean": float(correct_mean),
                "error_mean": float(error_mean),
                "ratio": float(error_mean / correct_mean) if correct_mean > 0 else 0,
            }

    results = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "n_total": len(y_test),
        "n_correct": int(correct.sum()),
        "n_misclassified": n_misclassified,
        "error_by_class": error_by_class,
        "top_confusions": [(f"{t} → {p}", c) for (t, p), c in top_confusions],
        "feature_error_importance": feature_error_importance,
    }

    # Print summary
    print(f"\n{'='*50}")
    print(f"  ERROR ANALYSIS — {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy: {accuracy:.4f} ({int(correct.sum())}/{len(y_test)})")
    print(f"  Misclassified: {n_misclassified} ({(1-accuracy)*100:.1f}%)")
    print(f"\n  Error Rate by Class:")
    for cls_name, stats in error_by_class.items():
        print(f"    {cls_name}: {stats['errors']}/{stats['total']} ({stats['error_rate']*100:.1f}%)")
    if top_confusions:
        print(f"\n  Top Confusion Pairs:")
        for pair, count in top_confusions:
            print(f"    {pair}: {count} times")
    print(f"{'='*50}")

    return results


def plot_error_distribution(
    error_results: List[Dict], save_path: None
) -> None:
    """Plot error distribution comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Error count by model
    names = [r["model_name"] for r in error_results]
    error_counts = [r["n_misclassified"] for r in error_results]
    accuracies = [r["accuracy"] for r in error_results]

    colors = sns.color_palette("Set2", len(names))
    bars = axes[0].bar(names, error_counts, color=colors, edgecolor="white")
    axes[0].set_title("Misclassified Samples by Model", fontsize=12)
    axes[0].set_ylabel("Number of Errors")
    for bar, acc in zip(bars, accuracies):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    axes[0].set_xticklabels(names, rotation=15, ha="right")

    # Right: Error rate by class
    class_labels = list(RISK_LABELS.values())
    x = np.arange(len(class_labels))
    width = 0.8 / len(names)

    for i, (name, r) in enumerate(zip(names, error_results)):
        rates = [r["error_by_class"][cls]["error_rate"] for cls in class_labels]
        offset = (i - len(names) / 2 + 0.5) * width
        axes[1].bar(x + offset, rates, width, label=name, color=colors[i], edgecolor="white")

    axes[1].set_title("Error Rate by Class", fontsize=12)
    axes[1].set_ylabel("Error Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_labels)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_error_comparison(
    error_results: List[Dict], save_path: None
) -> None:
    """Plot feature value distributions for correct vs misclassified samples."""
    import seaborn as sns

    # Use the model with most errors for detailed analysis
    most_errors = max(error_results, key=lambda r: r["n_misclassified"])

    if not most_errors["feature_error_importance"]:
        return

    features = list(most_errors["feature_error_importance"].keys())
    correct_means = [most_errors["feature_error_importance"][f]["correct_mean"] for f in features]
    error_means = [most_errors["feature_error_importance"][f]["error_mean"] for f in features]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(features))
    width = 0.35

    ax.barh(x - width / 2, correct_means, width, label="Correct", color="#2ecc71", edgecolor="white")
    ax.barh(x + width / 2, error_means, width, label="Misclassified", color="#e74c3c", edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Mean Absolute Feature Value")
    ax.set_title(f"Feature Values: Correct vs Misclassified — {most_errors['model_name']}", fontsize=12)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
