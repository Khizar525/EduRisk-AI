# Mermaid Diagrams — Copy-Paste Guide

All diagrams below are ready to paste. Each one shows:
- **File**: Which file to edit
- **Line**: Where the diagram starts (replace the existing ```mermaid block)
- **Clean Mermaid**: The fixed diagram code

---

## DIAGRAM 1 — High-Level Architecture

**File:** `docs/architecture.md`
**Line:** 9 (replace lines 9-48)

````
```mermaid
graph TB
    subgraph data["Data Layer"]
        A[Kaggle API] --> B[Raw CSV]
        B --> C[Data Validation]
    end

    subgraph proc["Processing Layer"]
        C --> D[Imputation]
        D --> E[Label Encoding]
        E --> F[Risk Engineering]
        F --> G[Standard Scaling]
    end

    subgraph train["Training Layer"]
        G --> H[Train-Test Split]
        H --> I[Model Training]
        I --> J[Hyperparameter Tuning]
        J --> K[Cross-Validation]
        K --> L[Model Selection]
    end

    subgraph infer["Inference Layer"]
        L --> M[Best Model Artifact]
        M --> N[Prediction Service]
        N --> O[Prediction Logger]
        N --> P[SHAP Explainer]
    end

    subgraph ui["Presentation Layer"]
        N --> Q[Gradio Dashboard]
        P --> R[Waterfall Plots]
        P --> S[Interpretation]
        Q --> T[User Interface]
    end

    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style Q fill:#fff3e0
```
````

---

## DIAGRAM 2 — Data Flow (No Leakage)

**File:** `docs/architecture.md`
**Line:** 52 (replace lines 52-66)

````
```mermaid
flowchart LR
    A[Load CSV] --> B[Impute]
    B --> C[Label Encode]
    C --> D[Engineer Risk Level]
    D --> E[Train-Test Split]
    E --> F[Fit Scaler on Train]
    F --> G[Transform Train]
    F --> H[Transform Test]
    G --> I[Train Models]
    I --> J[Evaluate on Test]

    style E fill:#ffcdd2
    style F fill:#c8e6c9
```
````

---

## DIAGRAM 3 — Prediction Flow (Sequence Diagram)

**File:** `docs/architecture.md`
**Line:** 87 (replace lines 87-107)

````
```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio UI
    participant P as Predictor
    participant S as SHAP Explainer
    participant L as Logger

    U->>G: Enter student profile
    G->>P: predict_risk(inputs)
    P->>P: Validate inputs
    P->>P: Encode categorical features
    P->>P: Scale features
    P->>P: Model predict and predict_proba
    P->>S: shap_values(input_scaled)
    S-->>P: SHAP values
    P->>P: Generate interpretation
    P-->>G: Result dict
    G->>L: log(inputs, result)
    G-->>U: Risk level, gauge, SHAP plot, interpretation
```
````

---

## DIAGRAM 4 — Feature Selection Mindmap

**File:** `docs/methodology.md`
**Line:** 15 (replace lines 15-34)

````
```mermaid
mindmap
  root((Features))
    Demographic
      Gender
      Age
    Academic
      Academic Pressure
      CGPA
      Study Satisfaction
      Work Study Hours
    Lifestyle
      Sleep Duration
      Dietary Habits
    Environmental
      Financial Stress
    Clinical
      Family History of Mental Illness
      Suicidal Thoughts
```
````

> **Note:** Changed "Work/Study Hours" to "Work Study Hours" — the `/` character breaks Mermaid rendering.

---

## DIAGRAM 5 — Preprocessing Pipeline

**File:** `docs/methodology.md`
**Line:** 56 (replace lines 56-68)

````
```mermaid
flowchart TD
    A[Full Dataset] --> B[Impute - Full Data]
    B --> C[Label Encode - Full Data]
    C --> D[Engineer Risk Level]
    D --> E[Train-Test Split 80-20]
    E --> F[Fit Scaler on TRAIN ONLY]
    F --> G[Transform TRAIN]
    F --> H[Transform TEST]

    style E fill:#ffcdd2
    style F fill:#c8e6c9
```
````

> **Note:** Changed "80/20" to "80-20" — the `/` character breaks Mermaid rendering.

---

## DIAGRAM 6 — Error Patterns

**File:** `docs/results.md`
**Line:** 109 (replace lines 109-120)

````
```mermaid
graph LR
    A[True Low] -->|Misclassified| B[Predicted Medium]
    C[True Medium] -->|Misclassified| D[Predicted Low]
    C -->|Misclassified| E[Predicted High]
    F[True High] -->|Misclassified| G[Predicted Medium]

    style B fill:#fff3cd
    style D fill:#fff3cd
    style E fill:#f8d7da
    style G fill:#f8d7da
```
````

> **Note:** Removed colons from labels — "True: Low" → "True Low" to avoid parsing issues.

---

## DIAGRAM 7 — System Overview (README)

**File:** `README.md`
**Line:** 81 (replace lines 81-118)

````
```mermaid
graph TB
    subgraph data["Data Layer"]
        A[Kaggle API] --> B[Raw CSV]
        B --> C[Validation]
    end

    subgraph proc["Processing"]
        C --> D[Imputation]
        D --> E[Encoding]
        E --> F[Risk Engineering]
    end

    subgraph train["Training"]
        F --> G[Split]
        G --> H[Scale - Train Only]
        H --> I[4 Models]
        I --> J[Tuning]
        J --> K[Best Model]
    end

    subgraph infer["Inference"]
        K --> L[Predictor]
        L --> M[SHAP Explainer]
        L --> N[Logger]
    end

    subgraph ui["UI"]
        L --> O[Gradio Dashboard]
        M --> P[Waterfall Plots]
        M --> Q[Interpretation]
    end

    style G fill:#ffcdd2
    style H fill:#c8e6c9
    style O fill:#fff3e0
```
````

---

## DIAGRAM 8 — Data Flow (README)

**File:** `README.md`
**Line:** 122 (replace lines 122-136)

````
```mermaid
flowchart LR
    A[Load] --> B[Impute]
    B --> C[Encode]
    C --> D[Risk Target]
    D --> E[Split 80-20]
    E --> F[Fit Scaler on Train]
    F --> G[Transform Train]
    F --> H[Transform Test]
    G --> I[Train]
    I --> J[Evaluate]

    style E fill:#ffcdd2
    style F fill:#c8e6c9
```
````

> **Note:** Changed "80/20" to "80-20".

---

## DIAGRAM 9 — ML Pipeline (README)

**File:** `README.md`
**Line:** 197 (replace lines 197-206)

````
```mermaid
flowchart TD
    A[1. Data Collection] --> B[2. EDA]
    B --> C[3. Preprocessing]
    C --> D[4. Feature Engineering]
    D --> E[5. Model Training]
    E --> F[6. Evaluation]
    F --> G[7. SHAP Explainability]
    G --> H[8. Deployment]
```
````

---

## Quick Reference — All Files

| # | File | Line | Diagram Name |
|---|------|------|-------------|
| 1 | `docs/architecture.md` | 9 | High-Level Architecture |
| 2 | `docs/architecture.md` | 52 | Data Flow (No Leakage) |
| 3 | `docs/architecture.md` | 87 | Prediction Flow |
| 4 | `docs/methodology.md` | 15 | Feature Selection Mindmap |
| 5 | `docs/methodology.md` | 56 | Preprocessing Pipeline |
| 6 | `docs/results.md` | 109 | Error Patterns |
| 7 | `README.md` | 81 | System Overview |
| 8 | `README.md` | 122 | Data Flow (README) |
| 9 | `README.md` | 197 | ML Pipeline |

## Common Rendering Fixes Applied

1. **Subgraph labels** — Wrapped in quotes: `subgraph name["Label"]`
2. **Slash characters** — Removed from labels: `80/20` → `80-20`, `Work/Study Hours` → `Work Study Hours`
3. **Colon characters** — Removed from labels: `True: Low` → `True Low`
4. **Special characters** — Avoided `()`, `/`, `:` inside node labels
5. **Consistent quoting** — All subgraph names use `["Quoted Labels"]`
