# LinkedIn Launch Kit — EduRisk AI

Ready-to-post LinkedIn content for launching the project.

---

## Table of Contents

1. [Launch Post](#launch-post)
2. [Engineering Deep Dive](#engineering-deep-dive)
3. [Carousel Outline](#carousel-outline)
4. [Developer Article Outline](#developer-article-outline)
5. [Hashtag Strategy](#hashtag-strategy)
6. [Posting Schedule](#posting-schedule)

---

## Launch Post

*Copy-paste ready. Post with the dashboard screenshot or a short demo GIF.*

---

I just shipped EduRisk AI — a production-oriented ML system that predicts student academic risk and explains its predictions.

**The problem:** Universities lose students to academic failure that could have been caught earlier. Most ML projects stop at training a model. I wanted to build the full lifecycle.

**What I built:**

- Trained 4 classifiers (Random Forest, XGBoost, SVM, MLP) on 27,901 student records
- Best model: Random Forest — 85.58% accuracy, 94.92% ROC-AUC
- SHAP TreeExplainer for per-prediction explainability — every result comes with a waterfall plot showing exactly which factors increased or decreased risk
- FastAPI REST API with Pydantic validation and auto-generated Swagger docs
- Next.js dark-mode frontend with animated risk gauges and SHAP visualizations
- Docker containerization + CI/CD pipeline
- 62 unit tests across all modules

**The key insight:** The model is the easiest part. The real engineering is in preprocessing, explainability, deployment, and testing.

**What I learned:**

- Data leakage prevention matters more than model selection
- SHAP TreeExplainer gives exact attributions in < 1 second
- FastAPI's Pydantic validation eliminates entire categories of bugs
- The gap between notebook and production is where most projects fail

Built with Python, scikit-learn, XGBoost, SHAP, FastAPI, Next.js, and Docker.

🔗 github.com/Khizar525/edurisk-ai

#MachineLearning #DataScience #Python #SHAP #FastAPI #NextJS #Portfolio #ExplainableAI

---

## Engineering Deep Dive

*Longer-form post for engineering audiences. Post 2-3 days after the launch post.*

---

I built EduRisk AI — an ML system predicting student academic risk. Here's what I learned about the engineering behind the model.

**The preprocessing pipeline had 6 steps:**

1. Drop duplicates
2. Impute missing values
3. Label encode categoricals
4. Engineer composite risk target
5. Train/test split (80/20)
6. Fit StandardScaler on train ONLY

That last point is critical. Most student projects fit the scaler on the entire dataset before splitting — that's data leakage. Your test accuracy is lying to you.

**Risk engineering was the hardest part.**

The original dataset had a binary "Depression" column. I needed a 3-class target (Low/Medium/High). I designed a weighted composite score:

- Depression: +3 points
- Suicidal thoughts: +3 points
- Academic pressure ≥ 4: +2 points
- CGPA < 2.0: +2 points
- Financial stress ≥ 4: +1 point
- CGPA ≥ 3.0: -2 points (protective)
- Study satisfaction ≥ 4: -1 point (protective)

Score ≤ 1 → Low Risk
Score ≤ 4 → Medium Risk
Score > 4 → High Risk

**Why Random Forest beat XGBoost:**

XGBoost had slightly higher ROC-AUC (95.02% vs 94.92%), but Random Forest won because:
- Higher accuracy (85.58% vs 85.24%)
- Better calibration
- TreeExplainer produces exact, consistent SHAP attributions
- Faster inference for real-time API

**The SHAP implementation surprised me.**

TreeExplainer on Random Forest computes exact Shapley values in < 1 second for 27K samples. I expected it to be slow. It wasn't.

For each prediction, I return:
- Top 3 risk factors (positive SHAP values)
- Top 3 protective factors (negative SHAP values)
- Human-readable interpretation ("Strongly increases risk")

This turns a black-box prediction into an actionable conversation between counselor and student.

**The architecture boundary I enforced:**

```
src/  → ML library (pure Python, no UI)
app/  → Application layer (Gradio + FastAPI)
frontend/ → Next.js
```

This separation means you can swap the UI without touching the ML code. It also means `src/` is independently testable — 62 tests cover every module.

**What I'd do differently:**

1. Start with the API first, not the notebook
2. Add MLflow from day one
3. Build an EDA notebook before jumping to modeling
4. Collect more data for the Medium Risk class (only 898 samples)

---

## Carousel Outline

*10-slide carousel for LinkedIn. Each slide = one image + short text.*

### Slide 1 — Cover
**Title:** EduRisk AI
**Subtitle:** Explainable ML for Academic Risk Prediction
**Visual:** Logo or dashboard screenshot

### Slide 2 — Problem
**Title:** The Problem
**Text:** Universities lose students to academic failure that could have been caught earlier. Most ML projects stop at training a model.
**Visual:** Icon or illustration

### Slide 3 — What I Built
**Title:** What I Built
**Text:**
- 4 ML classifiers compared
- SHAP explainability
- FastAPI REST API
- Next.js frontend
- 62 unit tests
**Visual:** Architecture diagram

### Slide 4 — Results
**Title:** Results
**Text:**
- 85.58% Accuracy
- 94.92% ROC-AUC
- 27,901 students
- 11 features
**Visual:** Model comparison chart

### Slide 5 — Explainability
**Title:** SHAP Explainability
**Text:** Every prediction comes with a waterfall plot showing exactly which factors increased or decreased risk.
**Visual:** SHAP waterfall screenshot

### Slide 6 — Architecture
**Title:** Architecture
**Text:**
- Leakage-free preprocessing
- 4-model comparison
- Config-driven design
- Modular codebase
**Visual:** System diagram

### Slide 7 — Tech Stack
**Title:** Tech Stack
**Text:** Python, scikit-learn, XGBoost, SHAP, FastAPI, Next.js, TypeScript, Tailwind CSS, Docker, GitHub Actions
**Visual:** Tech logos or clean list

### Slide 8 — Engineering
**Title:** Engineering Highlights
**Text:**
- 62 unit tests
- Pydantic validation
- Docker containerization
- CI/CD pipeline
**Visual:** Test results or code snippet

### Slide 9 — Lessons
**Title:** What I Learned
**Text:**
- The model is the easy part
- Preprocessing is where projects fail
- Explainability builds trust
- Testing catches what reviews miss
**Visual:** Clean text slide

### Slide 10 — CTA
**Title:** Check It Out
**Text:** github.com/Khizar525/edurisk-ai
**Visual:** GitHub screenshot or QR code

---

## Developer Article Outline

*For publishing on Medium, Dev.to, or Hashnode.*

### Title Options

1. "I Built an ML System That Explains Its Predictions — Here's What I Learned"
2. "From Notebook to Production: Building EduRisk AI"
3. "Why I Chose Random Forest Over XGBoost (And SHAP Over LIME)"
4. "The Engineering Gap: What Most ML Projects Get Wrong"

### Structure

**Introduction** (200 words)
- Hook: "Most ML repositories stop at training a model"
- What EduRisk AI does
- What makes it different

**The Problem** (300 words)
- Student academic risk prediction
- Why it matters
- What existing solutions lack

**Data & Feature Engineering** (400 words)
- Dataset overview (27,901 records, 11 features)
- Risk target engineering with composite scoring
- Why label encoding over one-hot

**Model Training & Selection** (500 words)
- 4 models compared
- Hyperparameter tuning (GridSearchCV + Optuna)
- Why Random Forest won over XGBoost

**Explainability with SHAP** (400 words)
- SHAP vs LIME vs Permutation Importance
- TreeExplainer implementation
- Human-readable interpretations

**Deployment Architecture** (400 words)
- FastAPI with Pydantic validation
- Next.js frontend with Tailwind
- Docker containerization

**Lessons Learned** (300 words)
- Notebook → production gap
- Data leakage prevention
- The model is the easiest part

**Conclusion** (200 words)
- GitHub link
- What's next
- Call to action

**Total: ~2,700 words**

---

## Hashtag Strategy

### Primary (use every post)
```
#MachineLearning #DataScience #Python #Portfolio
```

### Secondary (rotate)
```
#SHAP #ExplainableAI #FastAPI #NextJS #XGBoost #ScikitLearn
```

### Tertiary (for reach)
```
#AI #DeepLearning #SoftwareEngineering #Tech #Coding #100DaysOfCode
```

### Niche (for targeted reach)
```
#MLEngineering #MLOps #DataSciencePortfolio #StudentSuccess #EdTech
```

### Post Rules
- Use 3-5 hashtags per post (not 30)
- Put hashtags at the end, not in the middle
- Mix popular (#MachineLearning) with niche (#MLEngineering)
- Don't repeat the same set across posts

---

## Posting Schedule

### Week 1 — Launch

| Day | Post | Type |
|-----|------|------|
| Monday | Launch post (main announcement) | Text + screenshot |
| Wednesday | Engineering deep dive | Text + code snippets |
| Friday | Carousel (10 slides) | Images |

### Week 2 — Follow-up

| Day | Post | Type |
|-----|------|------|
| Monday | Developer article (Medium/Dev.to) | Link post |
| Wednesday | SHAP explainability focus | Text + SHAP screenshot |
| Friday | "What I learned" reflection | Text only |

### Week 3 — Engagement

| Day | Post | Type |
|-----|------|------|
| Monday | Behind the scenes (architecture) | Text + diagram |
| Wednesday | Comparison post (RF vs XGBoost) | Text + chart |
| Friday | Thank you / lessons learned | Text only |

### Ongoing
- Respond to every comment within 2 hours
- Share to relevant LinkedIn groups
- Tag team members in the post
- Cross-post to Twitter/X with shorter copy
