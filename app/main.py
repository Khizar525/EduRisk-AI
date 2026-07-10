"""
EduRisk AI — Gradio Application Entry Point.

Modern dashboard for student academic risk prediction.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

from src.inference.predictor import RiskPredictor
from src.inference.logger import PredictionLogger
from src.utils.validators import validate_prediction_inputs
from src.explainability.shap_utils import (
    compute_local_explanation,
    plot_local_waterfall,
    plot_probability_gauge,
)

# ── Initialize Services ────────────────────────────────────────
predictor = RiskPredictor()
logger = PredictionLogger()


# ── Prediction Function ────────────────────────────────────────

def predict_risk(
    gender, age, academic_pressure, cgpa, study_satisfaction,
    sleep_duration, dietary_habits, work_study_hours,
    financial_stress, family_history, suicidal_thoughts,
):
    """Main prediction function — returns all outputs."""
    # Validate
    is_valid, error_msg = validate_prediction_inputs(
        gender, age, academic_pressure, cgpa, study_satisfaction,
        sleep_duration, dietary_habits, work_study_hours,
        financial_stress, family_history, suicidal_thoughts,
    )
    if not is_valid:
        return (
            f"❌ {error_msg}", "", "", None, None,
            "", "", "",
        )

    inputs = {
        "gender": gender, "age": age, "academic_pressure": academic_pressure,
        "cgpa": cgpa, "study_satisfaction": study_satisfaction,
        "sleep_duration": sleep_duration, "dietary_habits": dietary_habits,
        "work_study_hours": work_study_hours, "financial_stress": financial_stress,
        "family_history": family_history, "suicidal_thoughts": suicidal_thoughts,
    }

    result = predictor.predict(**inputs)
    logger.log(inputs, result)

    # ── Risk Level Display ─────────────────────────────────────
    risk_text = result["risk_level"]

    # ── Confidence & Probabilities ─────────────────────────────
    prob_lines = []
    for label, prob in result["probabilities"].items():
        bar_len = int(prob / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        prob_lines.append(f"{label}: {bar} {prob}%")
    prob_text = "\n".join(prob_lines)

    # ── Advice ─────────────────────────────────────────────────
    advice_text = result["advice"]

    # ── SHAP Waterfall Image ───────────────────────────────────
    shap_img = None
    try:
        fig = plot_local_waterfall(
            result["shap"]["sorted_features"],
            result["shap"]["shap_values"],
            result["prediction"],
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        shap_img = Image.open(buf).copy()
        plt.close(fig)
        buf.close()
    except Exception as e:
        print(f"SHAP waterfall error: {e}")

    # ── Probability Donut ──────────────────────────────────────
    gauge_img = None
    try:
        fig = plot_probability_gauge(
            result["probabilities_raw"],
            result["prediction"],
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        gauge_img = Image.open(buf).copy()
        plt.close(fig)
        buf.close()
    except Exception as e:
        print(f"Gauge error: {e}")

    # ── Human-Readable Interpretation ──────────────────────────
    interpretation = result["shap"]["interpretation"]

    # ── Top Factors Summary ────────────────────────────────────
    risk_factors_text = ""
    if result["shap"]["top_risk"]:
        risk_factors_text = "**Top Risk Factors:**\n"
        for f in result["shap"]["top_risk"][:3]:
            risk_factors_text += f"- {f['feature']} (impact: {f['impact']:.3f})\n"

    protective_text = ""
    if result["shap"]["top_protective"]:
        protective_text = "**Top Protective Factors:**\n"
        for f in result["shap"]["top_protective"][:3]:
            protective_text += f"- {f['feature']} (impact: {f['impact']:.3f})\n"

    return (
        risk_text,
        prob_text,
        advice_text,
        shap_img,
        gauge_img,
        interpretation,
        risk_factors_text,
        protective_text,
    )


# ── Analytics Function ─────────────────────────────────────────

def get_analytics():
    """Get prediction analytics."""
    analytics = logger.get_analytics()

    if analytics["total"] == 0:
        return "No predictions yet.", None

    # Summary text
    lines = [
        f"**Total Predictions:** {analytics['total']}",
        f"**Average Confidence:** {analytics['avg_confidence']}%",
        "",
        "**Risk Distribution:**",
    ]
    for level, count in analytics.get("risk_distribution", {}).items():
        pct = count / analytics["total"] * 100
        lines.append(f"- {level}: {count} ({pct:.1f}%)")

    summary = "\n".join(lines)

    # Distribution chart
    dist_chart = None
    if analytics.get("risk_distribution"):
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(analytics["risk_distribution"].keys())
        values = list(analytics["risk_distribution"].values())
        colors = ["#2ecc71", "#f39c12", "#e74c3c"][:len(labels)]
        ax.barh(labels, values, color=colors, edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title("Prediction Distribution")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        dist_chart = Image.open(buf).copy()
        plt.close(fig)
        buf.close()

    return summary, dist_chart


def export_predictions():
    """Export predictions to a downloadable CSV."""
    export_path = PROJECT_ROOT / "data" / "prediction_export.csv"
    result = logger.export_csv(export_path)
    if result:
        return str(result)
    return "No predictions to export."


# ── Custom CSS ─────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
}
.risk-card {
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-size: 1.5em;
    font-weight: bold;
}
.risk-low { background-color: #d4edda; color: #155724; }
.risk-medium { background-color: #fff3cd; color: #856404; }
.risk-high { background-color: #f8d7da; color: #721c24; }
.factor-card {
    border-left: 4px solid;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    background: #f8f9fa;
}
.risk-factor { border-color: #e74c3c; }
.protective-factor { border-color: #3498db; }
"""


# ── UI Layout ──────────────────────────────────────────────────

with gr.Blocks(
    title="EduRisk AI — Student Risk Predictor",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css=CUSTOM_CSS,
) as app:

    # Header
    gr.Markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="margin-bottom: 5px;">🎓 EduRisk AI</h1>
            <p style="color: #666; font-size: 1.1em;">
                Predicting academic risk before it becomes a crisis
            </p>
        </div>
        """
    )

    with gr.Tabs():
        # ═══ TAB 1: PREDICTION ═══════════════════════════════════
        with gr.Tab("🔍 Predict", id="predict"):
            with gr.Row():
                # ── Input Panel ─────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### Student Profile")
                    with gr.Row():
                        gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                        age = gr.Slider(17, 30, value=20, step=1, label="Age")

                    with gr.Row():
                        cgpa = gr.Slider(0.0, 4.0, value=2.5, step=0.1, label="CGPA")
                        academic_pressure = gr.Slider(1, 5, value=3, step=1, label="Academic Pressure")

                    with gr.Row():
                        study_satisfaction = gr.Slider(1, 5, value=3, step=1, label="Study Satisfaction")
                        work_study_hours = gr.Slider(0, 12, value=6, step=1, label="Work/Study Hours")

                    with gr.Row():
                        sleep_duration = gr.Dropdown(
                            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
                            value="7-8 hours", label="Sleep Duration",
                        )
                        dietary_habits = gr.Dropdown(
                            ["Healthy", "Moderate", "Unhealthy"],
                            value="Moderate", label="Dietary Habits",
                        )

                    with gr.Row():
                        financial_stress = gr.Slider(1, 5, value=2, step=1, label="Financial Stress")
                        family_history = gr.Radio(["Yes", "No"], label="Family History of Mental Illness", value="No")
                        suicidal_thoughts = gr.Radio(["Yes", "No"], label="Suicidal Thoughts?", value="No")

                    predict_btn = gr.Button("🔍 Predict Risk", variant="primary", size="lg")

                # ── Output Panel ────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### Results")

                    risk_out = gr.Textbox(label="Risk Level", interactive=False, lines=1)
                    prob_out = gr.Textbox(label="Probability Distribution", interactive=False, lines=6)
                    advice_out = gr.Textbox(label="Recommendation", interactive=False, lines=3)

                    with gr.Row():
                        gauge_out = gr.Image(label="Risk Gauge", type="pil", interactive=False)
                        shap_out = gr.Image(label="Feature Contributions", type="pil", interactive=False)

            # ── Explanation Section ─────────────────────────────
            gr.Markdown("### Model Explanation")
            with gr.Row():
                with gr.Column(scale=1):
                    interpretation_out = gr.Markdown(label="Interpretation")
                with gr.Column(scale=1):
                    risk_factors_out = gr.Markdown(label="Risk Factors")
                    protective_factors_out = gr.Markdown(label="Protective Factors")

            # Wire up prediction
            predict_btn.click(
                fn=predict_risk,
                inputs=[
                    gender, age, academic_pressure, cgpa, study_satisfaction,
                    sleep_duration, dietary_habits, work_study_hours,
                    financial_stress, family_history, suicidal_thoughts,
                ],
                outputs=[
                    risk_out, prob_out, advice_out,
                    shap_out, gauge_out,
                    interpretation_out, risk_factors_out, protective_factors_out,
                ],
            )

        # ═══ TAB 2: ANALYTICS ═══════════════════════════════════
        with gr.Tab("📊 Analytics", id="analytics"):
            gr.Markdown("### Prediction Analytics")

            with gr.Row():
                with gr.Column(scale=1):
                    analytics_summary = gr.Markdown("Click to load analytics.")
                    refresh_btn = gr.Button("🔄 Refresh Analytics")
                with gr.Column(scale=1):
                    analytics_chart = gr.Image(label="Distribution", type="pil", interactive=False)

            export_btn = gr.Button("📥 Export Predictions (CSV)")
            export_output = gr.Textbox(label="Export Path", interactive=False)

            refresh_btn.click(fn=get_analytics, outputs=[analytics_summary, analytics_chart])
            export_btn.click(fn=export_predictions, outputs=[export_output])

        # ═══ TAB 3: ABOUT ═══════════════════════════════════════
        with gr.Tab("ℹ️ About", id="about"):
            gr.Markdown(
                """
                ## About EduRisk AI

                **EduRisk AI** is a machine learning system that predicts the academic risk level
                of university students based on self-reported lifestyle, psychological, and
                academic indicators.

                ### How It Works

                1. **Input**: Enter student profile data (demographics, academic metrics, lifestyle)
                2. **Model**: A trained ML classifier (Random Forest, XGBoost, SVM, or MLP) analyzes the input
                3. **SHAP**: Per-prediction explanations show which factors drove the decision
                4. **Output**: Risk level (Low/Medium/High) with confidence and recommendations

                ### Model Performance

                | Metric | Value |
                |--------|-------|
                | Model | Auto-selected best performer |
                | Features | 11 (demographic, academic, lifestyle) |
                | Target | 3-class risk level |
                | Validation | 5-fold stratified cross-validation |

                ### Team

                | Name | Role |
                |------|------|
                | M. Khizar Akram | Team Lead — App & Deployment |
                | Safwan Marwat | Data Collection & EDA |
                | Syed Mughees | Preprocessing & Feature Engineering |
                | Ifrahim Yousuf | Model Training & Evaluation |

                ### Links

                - [GitHub Repository](https://github.com/Khizar525/edurisk-ai)
                - [Dataset (Kaggle)](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

                ---
                *Built for CSL 460 — Data Mining | Bahria University Karachi Campus*
                """
            )


if __name__ == "__main__":
    app.launch(share=True)
