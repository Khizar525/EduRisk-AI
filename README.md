<div align="center">

# EduRisk AI

### Explainable Machine Learning for Academic Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-41B883.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-E0003D.svg?style=for-the-badge&logo=python&logoColor=white)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-000000.svg?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-62%20passing-brightgreen.svg?style=for-the-badge)](#testing)

<br>

**EduRisk AI is an explainable machine learning platform that predicts academic risk from behavioral and academic indicators while providing transparent, per-prediction explanations through SHAP.**

[Quick Start](#quick-start) • [Architecture](#architecture) • [Screenshots](#screenshots) • [Results](#results) • [API](#rest-api)

</div>

---

### Pipeline Overview

```mermaid
flowchart LR
    A["Dataset\n27,901 students"] --> B["Preprocessing\nImpute → Encode → Scale"]
    B --> C["Training\n4 models compared"]
    C --> D["Best Model\nRandom Forest"]
    D --> E["FastAPI\nREST API"]
    D --> F["Next.js\nFrontend"]
    D --> G["SHAP\nExplainability"]
    E --> H["Prediction\n+ waterfall plot"]

    style A fill:#e3f2fd
    style D fill:#c8e6c9
    style H fill:#fff3e0
```

---

## Why This Project?

Most machine learning repositories stop at training a model. EduRisk AI was designed to demonstrate **production-oriented machine learning engineering** by combining:

- Reproducible preprocessing pipelines with no data leakage
- Explainable AI with SHAP (global + per-prediction)
- REST APIs with FastAPI
- Interactive web applications with Next.js and Tailwind CSS
- Modular software architecture across 8+ packages
- Automated testing (62 unit tests)
- Containerized deployment with Docker

| Stage | What EduRisk AI Does |
|-------|---------------------|
| **Data** | 27,901 student records, 11 selected features, 3-class risk target |
| **Preprocessing** | Leakage-free pipeline — scaler fit on train only |
| **Training** | 4 models compared with GridSearchCV + Optuna Bayesian tuning |
| **Evaluation** | Accuracy, ROC-AUC, per-class metrics, error analysis, calibration |
| **Explainability** | SHAP TreeExplainer — global importance + per-prediction waterfall |
| **Serving** | FastAPI REST API + Next.js frontend + Gradio dashboard |
| **Testing** | 62 unit tests across all modules |
| **Deployment** | Docker containerization, CI/CD with GitHub Actions |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js, TypeScript, Tailwind CSS, Framer Motion |
| **API** | FastAPI, Uvicorn, Pydantic validation |
| **ML** | Scikit-learn, XGBoost, SHAP, Optuna |
| **Data** | Pandas, NumPy, Kaggle API |
| **DevOps** | Docker, GitHub Actions, Pytest |

---

## Quick Start

```mermaid
flowchart TD
    A["Clone repo"] --> B["Install deps"]
    B --> C["Train model"]
    C --> D["Start API :8000"]
    D --> E["Start Frontend :3000"]
    E --> F["Open localhost:3000"]

    style A fill:#e3f2fd
    style C fill:#c8e6c9
    style F fill:#fff3e0
```

```bash
# Clone
git clone https://github.com/Khizar525/EduRisk-AI.git
cd EduRisk-AI

# Setup
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Train
python -m src.training.trainer

# Launch API
uvicorn app.api:app --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs

# Launch Frontend
cd frontend && npm install && npm run dev
# Frontend: http://localhost:3000
```

> Gradio is also available via `python -m app.main` as a lightweight alternative.

---

## Architecture

EduRisk AI separates training, inference, and presentation into independent layers — the same prediction engine serves the REST API, the Next.js frontend, and the Gradio dashboard.

```mermaid
graph TB
    subgraph "Data Layer"
        A[Kaggle API] --> B[Raw CSV]
        B --> C[Validation]
    end

    subgraph "Processing"
        C --> D[Imputation]
        D --> E[Encoding]
        E --> F[Risk Engineering]
    end

    subgraph "Training"
        F --> G[Split]
        G --> H[Scale - Train Only]
        H --> I[4 Models]
        I --> J[Tuning - GridSearch/Optuna]
        J --> K[Best Model]
    end

    subgraph "Inference"
        K --> L[Predictor]
        L --> M[SHAP Explainer]
        L --> N[Logger]
    end

    subgraph "Presentation"
        L --> O[Next.js Frontend]
        L --> P[FastAPI REST API]
        L --> Q[Gradio Dashboard]
        M --> R[Waterfall Plots]
    end

    style G fill:#ffcdd2
    style H fill:#c8e6c9
    style O fill:#e3f2fd
```

### Data Flow (No Data Leakage)

```mermaid
flowchart LR
    A[Load] --> B[Impute]
    B --> C[Encode]
    C --> D[Risk Target]
    D --> E[Split 80/20]
    E --> F[Fit Scaler on Train]
    F --> G[Transform Train]
    F --> H[Transform Test]
    G --> I[Train]
    I --> J[Evaluate]

    style E fill:#ffcdd2
    style F fill:#c8e6c9
```

> **Note on preprocessing order**: The pipeline performs deterministic label encoding and simple median/mode imputation before splitting. Because these transformations do not use target labels, they have minimal impact on evaluation. Feature scaling—the transformation most likely to introduce evaluation bias—is fitted exclusively on the training set and applied to the test set.

---

## Screenshots

<div align="center">

**Dashboard — Dark Mode UI**

![Dashboard](assets/images/screenshot-dashboard.png)

</div>

<div align="center">

**Prediction with SHAP Explainability**

![Prediction](assets/images/screenshot-prediction.png)

</div>

<div align="center">

**SHAP Feature Contributions**

![SHAP Waterfall](assets/images/screenshot-factors.png)

</div>

<div align="center">

**REST API — Swagger Documentation**

![Swagger](assets/images/screenshot-swagger.png)

</div>

---

## Results

> **Best Model — Random Forest**
>
> Accuracy: **85.58%** · ROC-AUC: **94.92%** · CV: **85.27 ± 0.39%**

Although XGBoost achieved a slightly higher ROC-AUC (95.02%), Random Forest was selected for deployment because it provided the best balance between accuracy, calibration, inference speed, and SHAP interpretability. TreeExplainer on Random Forest produces exact, consistent feature attributions — critical for a system that must explain its predictions to non-technical stakeholders.

| Model | Accuracy | ROC-AUC | 3-Fold CV |
|-------|----------|---------|-----------|
| **Random Forest** | **85.58%** | **94.92%** | **85.27 ± 0.39%** |
| XGBoost | 85.24% | 95.02% | 85.88 ± 0.27% |
| MLP | 85.18% | 94.69% | 84.66 ± 0.46% |
| SVM | 82.12% | 93.10% | 82.40 ± 0.19% |

![Model Comparison](assets/images/model-comparison.png)

### Per-Class Performance (Random Forest)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Low Risk | 0.87 | 0.88 | 0.88 |
| Medium Risk | 0.70 | 0.59 | 0.64 |
| High Risk | 0.91 | 0.97 | 0.94 |

![Confusion Matrix](assets/images/confusion-matrix.png)
![Per-Class Metrics](assets/images/per-class-metrics.png)

> See [docs/results.md](docs/results.md) for detailed analysis. See [MODEL_CARD.md](MODEL_CARD.md) for model documentation.

---

## REST API

The FastAPI backend exposes three endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Risk prediction with SHAP explanation |
| `GET` | `/health` | Model health check |
| `GET` | `/model/info` | Model metadata and feature names |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 22,
    "academic_pressure": 5,
    "cgpa": 1.5,
    "study_satisfaction": 1,
    "sleep_duration": "Less than 5 hours",
    "dietary_habits": "Unhealthy",
    "work_study_hours": 10,
    "financial_stress": 5,
    "family_history": "Yes",
    "suicidal_thoughts": "Yes"
  }'
```

### Example Response

```json
{
  "prediction": 2,
  "risk_level": "High Risk",
  "predicted_probability": "94.9%",
  "shap": {
    "top_risk": [{"feature": "Suicidal Thoughts", "impact": 0.332}],
    "top_protective": [{"feature": "CGPA", "impact": -0.112}]
  }
}
```

Interactive docs at `http://localhost:8000/docs`.

---

## Repository Structure

```
EduRisk-AI/
├── app/                    # Application layer
│   ├── main.py             # Gradio dashboard
│   └── api.py              # FastAPI REST API
├── frontend/               # Next.js + Tailwind frontend
│   └── src/components/     # 6 React components
├── src/                    # Core ML library
│   ├── config.py           # Centralized configuration
│   ├── data/               # Data loading and validation
│   ├── preprocessing/      # Imputation, encoding, scaling
│   ├── features/           # Risk engineering
│   ├── training/           # Model training and tuning
│   ├── evaluation/         # Metrics, plots, error analysis
│   ├── explainability/     # SHAP utilities
│   ├── inference/          # Prediction service and logging
│   └── utils/              # Validators and helpers
├── tests/                  # 62 unit tests
├── docs/                   # Technical documentation
├── career/                 # Personal portfolio & interview assets
├── assets/                 # Charts, screenshots, figures
├── models/                 # Saved model artifacts
├── data/                   # Raw and processed data
└── docker/                 # Containerization
```

---

## Dataset

**Student Depression Dataset** — [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

| Property | Value |
|----------|-------|
| Records | 27,901 |
| Features Used | 11 (selected from 27) |
| Target | 3-class Risk Level (Low / Medium / High) |

### Selected Features

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Student gender |
| Age | Numerical | Student age (17-30) |
| Academic Pressure | Ordinal (1-5) | Self-reported academic pressure |
| CGPA | Numerical | Cumulative GPA (0.0-4.0) |
| Study Satisfaction | Ordinal (1-5) | Satisfaction with study conditions |
| Sleep Duration | Categorical | Daily sleep hours |
| Dietary Habits | Categorical | General diet quality |
| Work/Study Hours | Numerical | Daily study/work hours (0-12) |
| Financial Stress | Ordinal (1-5) | Self-reported financial stress |
| Family History | Categorical | Family history of mental illness |
| Suicidal Thoughts | Categorical | History of suicidal ideation |

### Limitations

- **Self-reported data** — features rely on student self-assessment
- **Survey bias** — collected via online surveys
- **Demographic limitations** — may not generalize across all populations
- **Temporal snapshot** — single point in time, not longitudinal

---

## Design Decisions

### Why Random Forest over XGBoost?

XGBoost had slightly higher ROC-AUC (95.02% vs 94.92%), but Random Forest was selected because: (1) higher overall accuracy (85.58% vs 85.24%), (2) better calibration, (3) faster inference, and (4) TreeExplainer produces exact, consistent SHAP attributions.

### Why SHAP over LIME?

SHAP provides both global feature importance and per-prediction explanations with mathematical guarantees (Shapley values). TreeExplainer on tree models gives exact attributions in polynomial time.

### Why FastAPI alongside Gradio?

FastAPI provides automatic OpenAPI documentation, native async support, and Pydantic validation. Gradio serves as a rapid prototyping interface for demos.

### Why Next.js over React + Vite?

Next.js provides server-side rendering, file-based routing, and a mature ecosystem for deployment on Vercel. TypeScript integration and Tailwind CSS made building the dark-mode UI fast and maintainable.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html
```

**62 tests** covering preprocessing, training, evaluation, SHAP, inference, and API.

---

## Documentation

- [Architecture Guide](docs/architecture.md) — system design with Mermaid diagrams
- [ML Methodology](docs/methodology.md) — pipeline, feature selection, model details
- [Results & Metrics](docs/results.md) — performance analysis and error patterns
- [Case Study](docs/case-study.md) — full engineering narrative
- [Model Card](MODEL_CARD.md) — model details, fairness, and usage
- [Contributing Guide](CONTRIBUTING.md) — how to contribute

---

## Roadmap

- [x] Multi-model training with hyperparameter tuning
- [x] SHAP explainability with waterfall plots
- [x] FastAPI REST API with Swagger documentation
- [x] Next.js dark-mode frontend with animated visualizations
- [x] Docker containerization
- [x] CI/CD with GitHub Actions
- [ ] MLflow experiment tracking
- [ ] Cloud deployment (Vercel + Render)
- [ ] Continuous model monitoring

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Dataset**: [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) on Kaggle
- **Course**: CSL 460 — Data Mining, Bahria University Karachi Campus
- **Instructors**: Dr. Hussain (Course), Engr. Noor us Sabah (Lab)

---

<div align="center">

### Developed by

**M. Khizar Akram** — Team Lead

*Project architecture, data acquisition, EDA, preprocessing, feature engineering, FastAPI, frontend integration, deployment, documentation, and major contributions to model training and evaluation.*

<br>

### Contributors

| Name | Contribution |
|------|-------------|
| M. Khizar Akram | Project architecture, data acquisition, EDA, preprocessing, feature engineering, FastAPI, frontend integration, deployment, documentation, model training & evaluation |
| Safwan Marwat | Data collection, Kaggle integration, exploratory analysis |
| Syed Mughees | Preprocessing pipeline, feature engineering, label encoding |
| Ifrahim Yousuf | Model training, hyperparameter tuning, evaluation metrics |

[![GitHub](https://img.shields.io/badge/GitHub-Khizar525-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/Khizar525)

</div>
