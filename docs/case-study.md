# EduRisk AI — Engineering Case Study

How we built a production-oriented ML system that predicts student academic risk with explainable AI.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Research & Discovery](#research--discovery)
3. [Dataset](#dataset)
4. [System Design](#system-design)
5. [Feature Engineering](#feature-engineering)
6. [Model Selection & Training](#model-selection--training)
7. [Evaluation](#evaluation)
8. [Explainability with SHAP](#explainability-with-shap)
9. [Deployment Architecture](#deployment-architecture)
10. [Engineering Challenges](#engineering-challenges)
11. [Lessons Learned](#lessons-learned)
12. [Future Improvements](#future-improvements)

---

## Problem Statement

Every year, universities lose students to academic failure and mental health crises that could have been intercepted earlier. Traditional academic monitoring relies on GPA thresholds and attendance tracking — reactive measures that detect problems after they've escalated.

**The core question:** Can we predict which students are at high academic risk *before* they fail — and explain exactly why?

This project set out to build not just a model, but a complete ML system: from data collection to real-time inference, with full explainability for non-technical stakeholders (counselors, faculty, administrators).

### Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|----------|
| Predictive accuracy | > 80% | 85.58% |
| ROC-AUC | > 90% | 94.92% |
| Explainability | Per-prediction SHAP | ✅ Waterfall + interpretation |
| Deployment | REST API + web UI | ✅ FastAPI + Next.js |
| Testing | > 50 unit tests | 62 tests |

---

## Research & Discovery

We reviewed existing literature on student performance prediction and identified three key gaps:

1. **Most projects stop at the model** — they train, evaluate, and report metrics. There's no path from notebook to production.
2. **Explainability is rare** — even projects that do predict student risk rarely explain *why* a student is flagged. Counselors need actionable insights, not just a risk label.
3. **Single-interface limitation** — most tools offer either a notebook, a Gradio demo, or an API. Rarely all three.

### What We Built Differently

We designed EduRisk AI as a **full ML lifecycle demonstration**:

- Reproducible pipeline with no data leakage
- Four models compared with hyperparameter tuning
- SHAP explainability at both global and per-prediction levels
- Three interfaces: REST API, Next.js frontend, Gradio dashboard
- 62 unit tests across all modules
- Docker containerization

---

## Dataset

**Student Depression Dataset** from [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset).

| Property | Value |
|----------|-------|
| Records | 27,901 |
| Original Features | 27 |
| Selected Features | 11 |
| Target | 3-class Risk Level |

### Feature Selection Rationale

We reduced from 27 features to 11 based on three criteria:

1. **Predictive relevance** — features with demonstrated correlation to academic outcomes
2. **Actionability** — features that counselors could actually influence or discuss
3. **Data quality** — features with < 5% missing values

| Feature | Type | Why Selected |
|---------|------|-------------|
| Gender | Categorical | Demographic baseline |
| Age | Numerical | Developmental context |
| Academic Pressure | Ordinal | Direct stress indicator |
| CGPA | Numerical | Academic performance proxy |
| Study Satisfaction | Ordinal | Engagement signal |
| Sleep Duration | Categorical | Wellbeing indicator |
| Dietary Habits | Categorical | Lifestyle factor |
| Work/Study Hours | Numerical | Workload measure |
| Financial Stress | Ordinal | Socioeconomic stressor |
| Family History | Categorical | Genetic/environmental risk |
| Suicidal Thoughts | Categorical | Critical risk signal |

### Target Engineering

The original dataset had a binary `Depression` column. We engineered a composite `Risk_Level` target using weighted scoring:

```python
# Risk factors (positive contributors)
score += 3 if depressed else 0
score += 2 if academic_pressure >= 4 else 0
score += 2 if cgpa < 2.0 else 0
score += 1 if financial_stress >= 4 else 0
score += 3 if suicidal_thoughts else 0
score += 1 if sleep < 5 hours else 0

# Protective factors (negative contributors)
score -= 2 if cgpa >= 3.0 else 0
score -= 1 if study_satisfaction >= 4 else 0

# Threshold mapping
risk_level = 0 if score <= 1 else 1 if score <= 4 else 2
```

This produced a 3-class target: Low Risk (0), Medium Risk (1), High Risk (2).

---

## System Design

### Architecture Principles

1. **Separation of concerns** — data, training, inference, and presentation are independent layers
2. **No data leakage** — scaler is fit on training data only
3. **Configuration-driven** — all hyperparameters, paths, and constants in `src/config.py`
4. **Reproducibility** — fixed random seeds, deterministic pipelines
5. **Graceful degradation** — SHAP falls back if computation fails

### Module Structure

```
src/
├── config.py           # Single source of truth
├── data/               # Loading, validation
├── preprocessing/      # Imputation, encoding, scaling
├── features/           # Risk engineering
├── training/           # Models, tuner, trainer
├── evaluation/         # Metrics, plots, error analysis
├── explainability/     # SHAP utilities
├── inference/          # Predictor, logger
└── utils/              # Validators, helpers
```

Each module is independently testable. The `config.py` module uses Python dataclasses for type-safe configuration:

```python
@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
    scoring: str = "accuracy"
    rf_param_grid: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 200],
        "max_depth": [None, 15],
        "min_samples_split": [2, 5],
    })
```

---

## Feature Engineering

### Preprocessing Pipeline

```mermaid
flowchart LR
    A[Raw CSV] --> B[Drop Duplicates]
    B --> C[Impute Missing]
    C --> D[Label Encode]
    D --> E[Risk Engineering]
    E --> F[Train/Test Split]
    F --> G[Fit Scaler on Train]
    G --> H[Transform Both]
```

**Critical design decision:** The scaler is fit on training data only. Test data is transformed using training statistics. This prevents data leakage — a common mistake in student ML projects.

### Encoding Strategy

| Feature | Encoding | Rationale |
|---------|----------|-----------|
| Gender | LabelEncoder | Binary (Male/Female) |
| Sleep Duration | LabelEncoder | Ordinal (4 categories) |
| Dietary Habits | LabelEncoder | Ordinal (3 categories) |
| Family History | LabelEncoder | Binary (Yes/No) |
| Suicidal Thoughts | LabelEncoder | Binary (Yes/No) |

Label encoding was chosen over one-hot because:
1. Tree-based models (Random Forest, XGBoost) handle ordinal encoding well
2. SVM and MLP work with scaled numerical features
3. Reduces feature space from 11 to 11 (vs 16+ with one-hot)

---

## Model Selection & Training

### Models Compared

| Model | Type | Why Included |
|-------|------|-------------|
| Random Forest | Ensemble (bagging) | Robust, interpretable, fast |
| XGBoost | Ensemble (boosting) | State-of-the-art tabular performance |
| SVM (RBF) | Kernel method | Non-linear decision boundaries |
| MLP | Neural network | Universal approximator |

### Hyperparameter Tuning

**Default:** GridSearchCV with 3-fold cross-validation.

**Alternative:** Optuna Bayesian optimization (50 trials per model, toggle via config).

```python
# Switch tuning strategy
TRAINING.use_optuna = False   # GridSearchCV
TRAINING.use_optuna = True    # Optuna
```

### Training Results

| Model | Accuracy | ROC-AUC | 3-Fold CV |
|-------|----------|---------|-----------|
| **Random Forest** | **85.58%** | **94.92%** | 85.27 ± 0.39% |
| XGBoost | 85.24% | 95.02% | 85.88 ± 0.27% |
| MLP | 85.18% | 94.69% | 84.66 ± 0.46% |
| SVM | 82.12% | 93.10% | 82.40 ± 0.19% |

### Why Random Forest Won

XGBoost achieved a slightly higher ROC-AUC (95.02% vs 94.92%), but Random Forest was selected for deployment because:

1. **Higher overall accuracy** (85.58% vs 85.24%)
2. **Better calibration** — probability estimates are more reliable
3. **Faster inference** — critical for real-time API responses
4. **Exact SHAP attributions** — TreeExplainer on Random Forest produces consistent, mathematically guaranteed explanations

---

## Evaluation

### Per-Class Performance (Random Forest)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Risk | 0.87 | 0.88 | 0.88 | 3,203 |
| Medium Risk | 0.70 | 0.59 | 0.64 | 898 |
| High Risk | 0.91 | 0.97 | 0.94 | 1,480 |

### Error Analysis

The model's primary weakness is **Medium Risk recall (0.59)** — it misclassifies 41% of medium-risk students. This is expected because:

1. Medium Risk is the smallest class (898 samples vs 3,203 Low and 1,480 High)
2. The boundary between Low and Medium is inherently fuzzy
3. The composite scoring function creates some overlap

**Confusion pairs:**
- Medium → Low: 285 cases (most common error)
- Medium → High: 83 cases
- Low → Medium: 389 cases

---

## Explainability with SHAP

### Why SHAP?

We evaluated three explainability approaches:

| Method | Pros | Cons |
|--------|------|------|
| **SHAP** | Mathematical guarantees, global + local | Computationally expensive |
| LIME | Fast, intuitive | No mathematical guarantees, unstable |
| Permutation Importance | Simple, model-agnostic | Only global, no per-prediction |

SHAP won because it provides both global feature importance and per-prediction explanations with Shapley value guarantees.

### Implementation

```python
# TreeExplainer for tree-based models (exact, fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)

# KernelExplainer fallback for non-tree models
background = shap.sample(scaler.transform(zeros), 50)
explainer = shap.KernelExplainer(model.predict_proba, background)
```

### Output

For each prediction, we return:
- **Top risk factors** — features that increase risk (positive SHAP values)
- **Top protective factors** — features that decrease risk (negative SHAP values)
- **Human-readable interpretation** — "Strongly increases risk" / "Slightly decreases risk"

Example output:
```json
{
  "top_risk": [
    {"feature": "Suicidal Thoughts", "impact": 0.332, "direction": "Strongly increases risk"},
    {"feature": "Financial Stress", "impact": 0.117, "direction": "Moderately increases risk"}
  ],
  "top_protective": [
    {"feature": "CGPA", "impact": -0.112, "direction": "Moderately decreases risk"}
  ]
}
```

---

## Deployment Architecture

### Three Interfaces

| Interface | Port | Purpose | Technology |
|-----------|------|---------|------------|
| Next.js Frontend | 3000 | Production UI | React, TypeScript, Tailwind |
| FastAPI REST API | 8000 | Programmatic access | FastAPI, Pydantic |
| Gradio Dashboard | 7860 | Rapid prototyping | Gradio |

### Why Three Interfaces?

1. **Next.js** — for production deployment (Vercel), professional appearance, SHAP visualizations
2. **FastAPI** — for integration with other systems, Swagger documentation, Pydantic validation
3. **Gradio** — for instructor demos, quick testing, zero-setup usage

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
```

---

## Engineering Challenges

### 1. Data Leakage Prevention

**Challenge:** The scaler must not see test data during training.

**Solution:** Explicit split before scaling. Scaler is fit on `X_train` only, then transforms both `X_train` and `X_test`.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)  # ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. SHAP Compatibility Across Models

**Challenge:** TreeExplainer works for tree models but not SVM/MLP. KernelExplainer is slow.

**Solution:** Lazy-loaded explainer that auto-detects model type:

```python
is_tree = any(kw in model_type.lower() for kw in ["forest", "xgb", "gradient"])
if is_tree:
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.KernelExplainer(model.predict_proba, background)
```

### 3. Frontend-Backend Communication

**Challenge:** API returns emoji keys (`"✅ Low Risk": 0.95`) but frontend needs numeric keys for probability bars.

**Solution:** Added `probabilities_raw` field to API response with string-keyed numeric values (0-1 decimals).

### 4. Medium Risk Class Imbalance

**Challenge:** Only 898 medium-risk samples (3.2% of dataset).

**Solution:** Accepted as a limitation. The composite scoring function inherently produces this distribution. Addressed through per-class metrics reporting and error analysis documentation.

---

## Lessons Learned

### What Worked

1. **Configuration-driven design** — centralized config made it easy to switch between GridSearchCV and Optuna
2. **Modular architecture** — each module is independently testable and replaceable
3. **Multiple interfaces** — FastAPI + Next.js + Gradio covers different use cases
4. **SHAP explainability** — per-prediction explanations are critical for stakeholder trust

### What We'd Do Differently

1. **Start with the API first** — build the prediction service before the UI
2. **Add MLflow earlier** — experiment tracking from day one
3. **More EDA notebooks** — data exploration should be documented in notebooks, not just code
4. **Stratified sampling for Medium class** — could improve recall

### Surprises

1. **XGBoost didn't clearly win** — on this dataset, Random Forest matched it on accuracy and exceeded on interpretability
2. **SHAP was faster than expected** — TreeExplainer on 27K rows runs in < 1 second
3. **Next.js was easier than Gradio** — for building a polished UI, React + Tailwind was faster than customizing Gradio

---

## Future Improvements

| Priority | Improvement | Impact |
|----------|-------------|--------|
| High | MLflow experiment tracking | Reproducibility, comparison |
| High | Cloud deployment (Vercel + Render) | Public demo |
| Medium | PostgreSQL prediction logging | Production-grade logging |
| Medium | Longitudinal data collection | Temporal patterns |
| Low | A/B testing framework | Model comparison in production |
| Low | Authentication + multi-user | Multi-student support |

---

## Conclusion

EduRisk AI demonstrates that a student ML project can achieve production quality without production complexity. By combining:

- **Rigorous ML** (4 models, hyperparameter tuning, proper evaluation)
- **Explainable AI** (SHAP per-prediction explanations)
- **Production engineering** (FastAPI, Next.js, Docker, 62 tests)
- **Professional documentation** (architecture diagrams, case study, model card)

...we created a portfolio piece that stands out from the typical "train a model, publish a notebook" approach.

The key insight: **the model is the easiest part.** The real work is in the engineering around it — preprocessing, explainability, deployment, testing, and documentation.

---

*Built for CSL 460 — Data Mining, Bahria University Karachi Campus.*
*Team: M. Khizar Akram, Safwan Marwat, Syed Mughees, Ifrahim Yousuf.*
