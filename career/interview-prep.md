# Interview Preparation — EduRisk AI

Top 30 interview questions this project could trigger, with high-quality answers.

---

## Table of Contents

1. [Machine Learning Fundamentals](#machine-learning-fundamentals)
2. [Project Architecture & Design](#project-architecture--design)
3. [Model Selection & Evaluation](#model-selection--evaluation)
4. [Explainability (SHAP)](#explainability-shap)
5. [Engineering & Deployment](#engineering--deployment)
6. [Data & Feature Engineering](#data--feature-engineering)
7. [Behavioral / Trade-off Questions](#behavioral--trade-off-questions)

---

## Machine Learning Fundamentals

### 1. Why did you choose these four models?

**Answer:**

I chose Random Forest, XGBoost, SVM, and MLP because they represent four distinct families of classifiers:

| Model | Family | Why Included |
|-------|--------|-------------|
| Random Forest | Ensemble (bagging) | Robust baseline, interpretable |
| XGBoost | Ensemble (boosting) | State-of-the-art for tabular data |
| SVM | Kernel method | Non-linear decision boundaries |
| MLP | Neural network | Universal approximator |

This diversity ensures we're not biased toward one approach. If all four agreed, we'd have high confidence. If they disagreed, we'd investigate why.

**Follow-up:** "Why not logistic regression?"
Logistic regression would be a good baseline, but on a dataset with non-linear feature interactions (e.g., CGPA × Academic Pressure), tree-based models typically outperform linear models. I prioritized models that could capture these interactions.

---

### 2. How did you prevent data leakage?

**Answer:**

Data leakage occurs when information from the test set contaminates training. I prevented it in three ways:

1. **Split before scaling** — `train_test_split` happens first, then the scaler is fit on `X_train` only:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)  # ONLY training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Uses training statistics
```

2. **No future data** — the dataset is a cross-sectional survey, so there's no temporal leakage.

3. **No target leakage** — the `Depression` column is used to engineer the target (`Risk_Level`) and then dropped before training.

**Why it matters:** If the scaler saw test data, our accuracy estimates would be optimistically biased. In production, the model would receive data it hasn't seen the distribution of.

---

### 3. What is ROC-AUC and why did you use it?

**Answer:**

ROC-AUC measures the model's ability to distinguish between classes across all classification thresholds. It's the area under the Receiver Operating Characteristic curve.

I used it because:
1. **Threshold-independent** — accuracy depends on a single threshold (0.5), but ROC-AUC evaluates all thresholds
2. **Handles class imbalance** — our Medium Risk class has only 898 samples (3.2%). Accuracy can be misleading with imbalanced classes
3. **Model comparison** — it provides a single number to compare models regardless of their probability calibration

**Follow-up:** "What's the difference between ROC-AUC and PR-AUC?"
ROC-AUC plots True Positive Rate vs False Positive Rate. PR-AUC plots Precision vs Recall. For imbalanced datasets, PR-AUC is more informative because it focuses on the minority class. I included both in my evaluation (ROC curves + PR curves in `assets/results/`).

---

### 4. Why 3-fold CV instead of 5-fold?

**Answer:**

With 27,901 samples, even 3-fold CV gives each fold ~9,300 samples — more than enough for stable estimates. I chose 3-fold for two reasons:

1. **Computational efficiency** — GridSearchCV with 5-fold on 27K rows × 4 models × param grid combinations was taking > 30 minutes. Reducing to 3-fold cut runtime by ~40%.
2. **Statistical sufficiency** — At this dataset size, the variance between 3-fold and 5-fold estimates is negligible (< 0.1% difference in mean accuracy).

The standard deviation across folds (±0.39% for Random Forest) confirms the estimates are stable.

---

### 5. How do you handle the class imbalance in Medium Risk?

**Answer:**

The Medium Risk class has only 898 samples (3.2%) vs 3,203 Low and 1,480 High. I chose **not** to apply resampling (SMOTE, undersampling) for three reasons:

1. **The imbalance reflects reality** — the composite scoring function produces this distribution naturally. Artificial balancing would distort the true risk distribution.
2. **Tree models handle imbalance reasonably** — Random Forest and XGBoost have built-in class weighting (`class_weight='balanced'`)
3. **Per-class metrics expose the issue** — I report per-class precision/recall/F1, so the weakness is transparent rather than hidden behind overall accuracy

**What I'd do differently:** If Medium Risk recall were critical (e.g., for a counseling prioritization system), I'd use class weights or SMOTE and evaluate the trade-off.

---

## Project Architecture & Design

### 6. Why separate `src/` and `app/`?

**Answer:**

This separation enforces a clean boundary between **business logic** and **presentation**:

- `src/` — the ML library. Contains data loading, preprocessing, training, evaluation, inference. This is the core product.
- `app/` — the application layer. Contains Gradio UI (`main.py`) and FastAPI API (`api.py`). These are consumers of `src/`.

**Benefits:**
1. You can swap the UI without touching the ML code
2. `src/` is independently testable (62 tests)
3. Other projects can import `src/` as a library
4. It mirrors how production ML systems are structured (feature store → model service → API → UI)

---

### 7. Why FastAPI over Flask?

**Answer:**

| Feature | FastAPI | Flask |
|---------|---------|-------|
| Auto-generated docs | ✅ Swagger + ReDoc | ❌ Requires flask-restx |
| Request validation | ✅ Pydantic built-in | ❌ Manual validation |
| Async support | ✅ Native | ⚠️ Requires flask-async |
| Type safety | ✅ Type hints → validation | ❌ Runtime errors |
| Performance | ✅ Starlette (async) | ⚠️ WSGI (sync) |

For an ML API where request validation matters (wrong types → crashes), FastAPI's Pydantic integration eliminates entire categories of bugs. The auto-generated Swagger docs also made the API self-documenting for the frontend team.

---

### 8. Why keep Gradio alongside Next.js?

**Answer:**

They serve different purposes:

| Use Case | Gradio | Next.js |
|----------|--------|---------|
| Instructor demo | ✅ Zero setup | ❌ Needs npm install |
| Quick testing | ✅ Launch in 1 line | ❌ Needs backend running |
| Production UI | ❌ Limited customization | ✅ Full control |
| SHAP visualizations | ⚠️ Basic | ✅ Custom waterfall chart |
| Dark mode | ⚠️ Limited | ✅ Tailwind CSS |
| Deployment | ⚠️ Share link only | ✅ Vercel |

Gradio is a **prototyping tool**. Next.js is a **product**. Keeping both shows I understand the difference between demo and production.

---

### 9. Why use dataclasses for configuration?

**Answer:**

```python
@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
```

Benefits over dictionaries or YAML:
1. **Type safety** — IDE catches errors at write time
2. **Autocomplete** — `TRAINING.cv_folds` instead of `config['training']['cv_folds']`
3. **Defaults** — built-in, no external parsing
4. **Refactoring** — renaming a field updates all references
5. **No external dependency** — stdlib, no YAML/TOML parser needed

---

### 10. How would you scale this to 1M students?

**Answer:**

Current architecture handles 27K fine. For 1M:

1. **Training:** Switch from in-memory pandas to Dask or Spark. Or use incremental learning (`partial_fit` for SGDClassifier).
2. **Inference:** Add Redis caching for repeated predictions. Batch predictions for bulk assessment.
3. **API:** Deploy behind a load balancer (nginx/ALB). Add rate limiting.
4. **Database:** Replace CSV logging with PostgreSQL + async inserts.
5. **Model:** Consider lighter models (distilled Random Forest, ONNX runtime) for faster inference.

The modular architecture makes this feasible — you'd only change `src/training/` and `src/inference/`, not the entire system.

---

## Model Selection & Evaluation

### 11. Why did Random Forest win over XGBoost?

**Answer:**

XGBoost had slightly higher ROC-AUC (95.02% vs 94.92%), but Random Forest was better on three other dimensions:

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Accuracy | **85.58%** | 85.24% |
| ROC-AUC | 94.92% | **95.02%** |
| CV Stability | **±0.39%** | ±0.27% |
| SHAP Speed | **< 1s** | < 1s |
| Calibration | **Better** | Slightly overconfident |

The accuracy difference (0.34%) matters more than ROC-AUC in practice because we deploy at a fixed threshold. And Random Forest's TreeExplainer produces more consistent attributions.

**Key insight:** ROC-AUC alone doesn't determine the best model for deployment. You need to consider accuracy, calibration, inference speed, and explainability.

---

### 12. What would you do if Medium Risk recall was critical?

**Answer:**

If the use case demanded high Medium Risk recall (e.g., "never miss a student who needs counseling"), I'd:

1. **Lower the classification threshold** for Medium Risk specifically (trade precision for recall)
2. **Use class weights** — `class_weight={0: 1, 1: 5, 2: 1}` to penalize Medium misclassification
3. **Try SMOTE** — synthetic oversampling of the Medium class
4. **Ensemble** — combine predictions from multiple models
5. **Collect more data** — the best solution for class imbalance

The current 59% recall means 41% of medium-risk students are missed. In a real system, this would be unacceptable.

---

### 13. How do you choose between precision and recall?

**Answer:**

It depends on the cost of errors:

| Error Type | Cost | Prefer |
|------------|------|--------|
| False Positive (predict risk, student is fine) | Wasted counselor time | High precision |
| False Negative (predict safe, student is at risk) | Student falls through cracks | High recall |

For academic risk prediction, **false negatives are more dangerous** — missing a high-risk student has worse consequences than a false alarm. So I'd optimize for recall on the High Risk class (currently 97%, which is excellent).

For Medium Risk, the trade-off is harder. The 59% recall is a known weakness.

---

### 14. Why 80/20 split and not cross-validation for final evaluation?

**Answer:**

I use **both**:

1. **3-fold CV during training** — for model selection and hyperparameter tuning. This gives a robust estimate of each model's performance.
2. **80/20 holdout for final evaluation** — the test set is never seen during training or tuning. This simulates production performance.

The CV score (85.27%) and test accuracy (85.58%) are close, which confirms:
- No overfitting to the training set
- The test set is representative
- The model generalizes well

---

### 15. How do you handle a model that's overfitting?

**Answer:**

Signs of overfitting: high training accuracy, low test accuracy, large gap between train and validation curves.

My approach:
1. **Check learning curves** — if train/val curves diverge, the model is overfitting
2. **Reduce complexity** — lower `max_depth`, increase `min_samples_split` for trees
3. **Regularization** — add L1/L2 penalty, increase `alpha` for SVM/MLP
4. **More data** — the best regularizer
5. **Ensemble** — bagging (Random Forest) naturally reduces variance

In this project, the learning curves (`assets/results/learning_curves.png`) show convergence, confirming no significant overfitting.

---

## Explainability (SHAP)

### 16. Why SHAP over LIME?

**Answer:**

| Criterion | SHAP | LIME |
|-----------|------|------|
| Mathematical basis | Shapley values (game theory) | Local linear approximation |
| Consistency | ✅ Guaranteed | ❌ Can change with retraining |
| Global explanations | ✅ Yes | ⚠️ Aggregate only |
| Per-prediction | ✅ Yes | ✅ Yes |
| Computational cost | Higher | Lower |
| Theoretical guarantees | ✅ Yes | ❌ No |

SHAP is theoretically grounded — Shapley values are the *unique* solution satisfying efficiency, symmetry, dummy, and additivity axioms. LIME is an approximation that can produce different results across runs.

For a system that must explain predictions to non-technical stakeholders (counselors), mathematical guarantees matter.

---

### 17. What's the difference between TreeExplainer and KernelExplainer?

**Answer:**

| | TreeExplainer | KernelExplainer |
|---|---------------|-----------------|
| Applicable to | Tree-based models only | Any model |
| Speed | Fast (polynomial) | Slow (sampling-based) |
| Accuracy | Exact | Approximate |
| How it works | Exploits tree structure | Treats model as black box, uses kernel regression |

I use TreeExplainer for Random Forest and XGBoost (exact, fast). For SVM and MLP, I fall back to KernelExplainer with 50 background samples.

```python
is_tree = any(kw in model_type.lower() for kw in ["forest", "xgb"])
if is_tree:
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.KernelExplainer(model.predict_proba, background)
```

---

### 18. How do you interpret SHAP values for a counselor?

**Answer:**

I translate SHAP values into plain language:

| SHAP Value | Interpretation |
|------------|---------------|
| > +0.2 | "Strongly increases risk" |
| +0.1 to +0.2 | "Moderately increases risk" |
| +0.05 to +0.1 | "Slightly increases risk" |
| -0.05 to +0.05 | "Minimal impact" |
| -0.1 to -0.05 | "Slightly decreases risk" |
| < -0.1 | "Moderately decreases risk" |

For a counselor, the conversation would be:
> "This student's suicidal thoughts (+0.33) and financial stress (+0.12) are the biggest risk factors. Their CGPA (-0.11) is protective. I'd focus counseling on the mental health and financial support aspects."

---

### 19. What are SHAP's limitations?

**Answer:**

1. **Correlated features** — SHAP can split credit between correlated features unpredictably. If Academic Pressure and Financial Stress are correlated, their individual SHAP values may be unstable.
2. **Computational cost** — KernelExplainer is slow for large datasets (> 100K rows)
3. **Global vs local tension** — Global importance (mean |SHAP|) doesn't always align with per-prediction importance
4. **Assumes independence** — SHAP values assume features are independent, which isn't true in reality
5. **Model-specific** — TreeExplainer only works for tree models

I documented these limitations in the model card and case study.

---

### 20. How would you explain SHAP to a non-technical stakeholder?

**Answer:**

> "Think of it like a tax calculator. When you file your taxes, each income source and deduction contributes a specific amount to your final bill. SHAP does the same thing for predictions — it shows exactly how much each factor (sleep, stress, grades) contributes to the risk score.
>
> For this student, suicidal thoughts added +33 points to their risk, financial stress added +12 points, but their CGPA subtracted 11 points. The counselor can see exactly what to focus on."

---

## Engineering & Deployment

### 21. Why Docker?

**Answer:**

Docker solves the "it works on my machine" problem:

1. **Reproducibility** — same Python version, same dependencies, same environment
2. **Isolation** — doesn't conflict with system Python or other projects
3. **Deployment** — one command to start: `docker-compose up`
4. **Scaling** — can run multiple containers behind a load balancer

My Dockerfile uses `python:3.11-slim` (minimal image), installs dependencies first (layer caching), and runs as a non-root user.

---

### 22. How does the FastAPI request validation work?

**Answer:**

Pydantic models enforce schema at the API boundary:

```python
class PredictionRequest(BaseModel):
    gender: str = Field(..., description="Gender (Male/Female)")
    age: int = Field(..., ge=17, le=30, description="Age (17-30)")
    academic_pressure: int = Field(..., ge=1, le=5)
    cgpa: float = Field(..., ge=0.0, le=4.0)
```

If a request sends `age: 50`, FastAPI returns a 422 error with a clear message:
```json
{"detail": [{"loc": ["body", "age"], "msg": "ensure this value is less than or equal to 30"}]}
```

This eliminates entire categories of bugs: wrong types, out-of-range values, missing fields.

---

### 23. How would you add authentication?

**Answer:**

Current API is open (no auth). To add it:

1. **API keys** — simplest. Add `X-API-Key` header validation middleware
2. **JWT tokens** — for user-specific rate limiting and audit trails
3. **OAuth2** — for integration with university SSO systems

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.middleware("http")
async def validate_api_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    api_key = request.headers.get("X-API-Key")
    if api_key != VALID_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)
```

---

### 24. What's the CI/CD pipeline?

**Answer:**

GitHub Actions runs on every push and PR:

```yaml
- Python 3.10, 3.11, 3.12 matrix
- flake8 lint check
- black format check
- pytest with coverage
- Codecov upload
```

This catches:
- Syntax errors (flake8)
- Formatting issues (black)
- Test failures (pytest)
- Coverage drops (codecov)

For production, I'd add:
- Docker build + push to registry
- Deploy to Vercel/Render on merge to main
- Model performance monitoring (data drift detection)

---

### 25. How do you handle model versioning?

**Answer:**

Currently, models are saved as `.pkl` files in `models/`:

```
models/
├── best_model.pkl
├── scaler.pkl
├── encoders.pkl
└── feature_names.pkl
```

For production, I'd implement:

1. **MLflow** — experiment tracking, model registry, versioning
2. **Model registry** — `model_v1.pkl`, `model_v2.pkl` with metadata
3. **A/B testing** — route 10% of traffic to new model
4. **Rollback** — if new model degrades, switch back to previous version

---

## Data & Feature Engineering

### 26. Why label encoding instead of one-hot?

**Answer:**

| Encoding | Features Created | Best For |
|----------|-----------------|----------|
| Label Encoding | 1 (same) | Tree models, ordinal features |
| One-Hot Encoding | N categories | Linear models, nominal features |

I used label encoding because:
1. **Tree models handle it well** — Random Forest and XGBoost split on thresholds, so encoded values work fine
2. **Feature space stays small** — 11 features instead of 16+ (which would happen with one-hot)
3. **SVM/MLP get scaled features** — the scaler normalizes the encoded values

If I were using logistic regression or a linear SVM, I'd use one-hot encoding.

---

### 27. How did you choose the risk engineering weights?

**Answer:**

The weights were preserved from the original notebook (designed by Syed Mughees):

```python
depression_weight: 3       # Strongest signal
suicidal_thoughts_weight: 3 # Critical risk
academic_pressure_weight: 2
low_cgpa_weight: 2
financial_stress_weight: 1
low_sleep_weight: 1
high_cgpa_bonus: -2        # Protective
high_satisfaction_bonus: -1 # Protective
```

The weights reflect clinical intuition:
- Depression and suicidal thoughts are the strongest risk factors
- Academic pressure and low CGPA are moderate
- Financial stress and sleep are contributing factors
- High CGPA and satisfaction are protective

**Validation:** The resulting class distribution (Low: 76%, Medium: 14%, High: 10%) is realistic for a university population.

---

### 28. What would you do with more data?

**Answer:**

With more data, I'd:

1. **Add temporal features** — GPA trend over semesters, attendance patterns
2. **Add social features** — peer interactions, club participation
3. **Add environmental features** — distance from home, housing type
4. **Try deep learning** — with 100K+ samples, a tabular transformer (TabNet) could outperform trees
5. **Build a recommender** — suggest specific interventions based on risk profile

---

## Behavioral / Trade-off Questions

### 29. What was the hardest part of this project?

**Answer:**

The hardest part was **the gap between notebook and production**.

In a notebook, you can hardcode paths, skip error handling, and ignore edge cases. In production:
- The API must validate every input
- The predictor must handle missing encoders gracefully
- The frontend must show meaningful error messages
- The pipeline must be reproducible end-to-end

The biggest surprise was how much work the "last 10%" takes. Training a model is 10% of the effort. The other 90% is preprocessing, testing, documentation, deployment, and making it explainable.

---

### 30. If you had two more weeks, what would you improve?

**Answer:**

| Week 1 | Week 2 |
|--------|--------|
| MLflow experiment tracking | Cloud deployment (Vercel + Render) |
| PostgreSQL prediction logging | Model monitoring dashboard |
| EDA notebook for data exploration | A/B testing framework |

The highest-impact improvement would be **MLflow** — it would make model comparison automatic and reproducible. The second would be **cloud deployment** — a live demo URL is worth more than any amount of documentation.

---

## Quick Reference Card

| Topic | Key Points |
|-------|-----------|
| **Data leakage** | Split before scaling, scaler fit on train only |
| **Model selection** | Random Forest chosen for accuracy + interpretability |
| **Class imbalance** | Accepted as limitation, per-class metrics exposed |
| **SHAP** | TreeExplainer for trees, KernelExplainer for others |
| **Architecture** | src/ (ML) vs app/ (UI), config-driven, testable |
| **Why FastAPI** | Auto-docs, Pydantic validation, async |
| **Why 3-fold CV** | 27K samples, computational efficiency |
| **Hardest part** | Notebook → production gap (last 10% takes 90%) |
