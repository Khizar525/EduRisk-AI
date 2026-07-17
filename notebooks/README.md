# Notebooks

## Original Colab Notebook

This project started as a team Colab notebook for CSL 460 — Data Mining coursework at Bahria University.

The original notebook (`DM_Project.ipynb`) was a Google Colab notebook with:
- RandomForestClassifier, SVC, XGBClassifier, MLPClassifier
- GridSearchCV + StratifiedKFold (5-fold) cross-validation
- SHAP TreeExplainer + KernelExplainer
- Basic Gradio gr.Blocks() app

**Contributors to the original notebook:**
- Safwan Marwat — Data collection, Kaggle integration, exploratory analysis
- Syed Mughees — Preprocessing pipeline, feature engineering, label encoding
- Ifrahim Yousuf — Model training, hyperparameter tuning, evaluation metrics
- M. Khizar Akram — Project lead, integration, deployment

## Production Transformation

The Colab notebook was transformed into the production-ready **EduRisk AI** platform by M. Khizar Akram alone:

- Real dataset training (27,901 records, not synthetic)
- FastAPI REST API with 3 endpoints
- Next.js 14 dark-mode frontend with SHAP visualizations
- 62 unit tests across 8 modules
- Docker containerization
- Architecture docs, case study, model card
- CI/CD with GitHub Actions

The original Colab notebook is available on [Google Colab](https://colab.research.google.com/) for reference.
