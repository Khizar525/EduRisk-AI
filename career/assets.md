# Visual Assets Index

Complete catalog of all visual assets in the EduRisk AI repository.

---

## Brand Assets

| File | Type | Dimensions | Purpose | Status |
|------|------|-----------|---------|--------|
| `assets/images/logo.svg` | SVG | Scalable | Primary logo (shield + neural network) | ✅ Ready |
| `assets/images/logo.png` | PNG | — | Rasterized logo | ⚠️ Convert from SVG |
| `assets/images/banner.png` | PNG | 1280×640 | GitHub repository banner | ✅ Exists |
| `assets/images/og-image.svg` | SVG | 1200×630 | Open Graph social preview | ✅ Ready |
| `assets/images/og-image.png` | PNG | 1200×630 | OG image for web sharing | ⚠️ Convert from SVG |

---

## Application Screenshots

| File | Description | Section in README |
|------|-------------|-------------------|
| `assets/images/screenshot-dashboard.png` | Empty form — dark mode UI | Screenshots |
| `assets/images/screenshot-prediction.png` | High risk result with gauge + SHAP | Screenshots |
| `assets/images/screenshot-factors.png` | SHAP waterfall chart | Screenshots |
| `assets/images/screenshot-swagger.png` | FastAPI Swagger documentation | Screenshots |

**Recommended size**: 1200×800px (2:3 aspect ratio for mobile-friendly display)

---

## Model Evaluation Charts

| File | Description | Used In |
|------|-------------|---------|
| `assets/images/model-comparison.png` | Grouped bar chart — all 4 models | README Results |
| `assets/images/confusion-matrix.png` | Random Forest confusion matrix | README Results |
| `assets/images/per-class-metrics.png` | Precision/Recall/F1 per class | README Results |
| `assets/images/class-distribution.png` | Target class balance | docs/results.md |
| `assets/images/risk-breakdown.png` | Risk level distribution | docs/results.md |
| `assets/images/shap-importance.png` | Global SHAP feature importance | docs/results.md |

---

## Detailed Evaluation Figures

| File | Description | Used In |
|------|-------------|---------|
| `assets/results/roc_curves.png` | ROC curves for all 4 models | docs/results.md |
| `assets/results/pr_curves.png` | Precision-recall curves | docs/results.md |
| `assets/results/calibration_curves.png` | Probability calibration plots | docs/results.md |
| `assets/results/learning_curves.png` | Learning curves (train vs val) | docs/results.md |
| `assets/results/model_radar.png` | Radar chart — multi-metric comparison | docs/results.md |
| `assets/results/model_comparison.png` | Model comparison (alternate) | docs/results.md |
| `assets/results/class_distribution.png` | Class distribution (alternate) | docs/results.md |

---

## Per-Model Confusion Matrices

| File | Model |
|------|-------|
| `assets/results/cm_Random_Forest.png` | Random Forest |
| `assets/results/cm_XGBoost.png` | XGBoost |
| `assets/results/cm_MLP.png` | MLP |
| `assets/results/cm_SVM.png` | SVM |

---

## SHAP Analysis

| File | Description |
|------|-------------|
| `assets/results/shap_bar_rf.png` | SHAP bar chart — feature importance |
| `assets/results/shap_beeswarm_rf.png` | SHAP beeswarm — feature distribution |

---

## Mermaid Diagrams (In-Code)

| Location | Diagram | Purpose |
|----------|---------|---------|
| README.md | Architecture (graph TB) | System overview |
| README.md | Data Flow (flowchart LR) | No data leakage |
| README.md | Pipeline (flowchart TD) | 8-step methodology |
| docs/architecture.md | High-Level Architecture | Detailed system design |
| docs/architecture.md | Data Flow | Leakage prevention |
| docs/architecture.md | Prediction Flow (sequence) | User → UI → Predictor |
| docs/architecture.md | Deployment Architecture | Services + ports |
| docs/architecture.md | Inference Flow | Request lifecycle |
| docs/methodology.md | Pipeline diagram | Preprocessing steps |

---

## Asset Creation Checklist

### Done
- [x] SVG logo
- [x] OG image SVG
- [x] 4 application screenshots
- [x] 6 evaluation charts (matplotlib from real data)
- [x] 13 detailed evaluation figures
- [x] 4 per-model confusion matrices
- [x] 2 SHAP analysis charts
- [x] 9 Mermaid diagrams (in code)

### TODO
- [ ] Convert `logo.svg` → `logo.png` (for places that don't support SVG)
- [ ] Convert `og-image.svg` → `og-image.png` (for deployment)
- [ ] Generate favicon from logo (32×32)
- [ ] Upload banner as GitHub social preview
- [ ] Add screenshots to `docs/screenshots/` (currently empty)

---

## Conversion Commands

```bash
# SVG to PNG (requires Inkscape or rsvg-convert)
inkscape assets/images/logo.svg -w 512 -h 512 -o assets/images/logo.png
inkscape assets/images/og-image.svg -w 1200 -h 630 -o assets/images/og-image.png

# Or using rsvg-convert
rsvg-convert -w 512 -h 512 assets/images/logo.svg > assets/images/logo.png
rsvg-convert -w 1200 -h 630 assets/images/og-image.svg > assets/images/og-image.png

# Favicon from SVG
inkscape assets/images/logo.svg -w 32 -h 32 -o frontend/public/favicon.ico
```
