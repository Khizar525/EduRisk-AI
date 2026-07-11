# GitHub Social Preview — Setup Guide

## Repository Banner

**File**: `assets/images/banner.png`
**Dimensions**: 1280 × 640px
**Format**: PNG

### How to Set
1. Go to https://github.com/Khizar525/edurisk-ai
2. Click the ⚙️ gear icon on the right side of the repository description
3. Click "Edit repository social preview"
4. Upload `assets/images/banner.png`

### Content
- Dark background (#0a0a0f)
- EduRisk AI logo (left)
- Tagline: "Explainable ML for Academic Risk Prediction"
- Stats: 85.58% Accuracy · 94.92% ROC-AUC · 62 Tests

---

## Open Graph Image (For Web Sharing)

**File**: `assets/images/og-image.svg` (convert to PNG for production)
**Dimensions**: 1200 × 630px
**Format**: SVG → PNG

### Conversion Command
```bash
# Using Inkscape
inkscape assets/images/og-image.svg -w 1200 -h 630 -o assets/images/og-image.png

# Using rsvg-convert
rsvg-convert -w 1200 -h 630 assets/images/og-image.svg > assets/images/og-image.png
```

### Deploy to Frontend
```bash
cp assets/images/og-image.png frontend/public/og-image.png
```

### Meta Tags (Already Configured)
The Next.js layout (`frontend/src/app/layout.tsx`) includes:
- `og:title` — "EduRisk AI — Explainable Academic Risk Intelligence"
- `og:description` — Full description
- `og:image` — `/og-image.png`
- `og:url` — `https://edurisk-ai.vercel.app`
- `twitter:card` — `summary_large_image`

---

## Favicon

Replace `frontend/public/favicon.ico` with a 32×32 or 16×16 version of the logo.

### Generate from SVG
```bash
# Using ImageMagick
convert assets/images/logo.svg -resize 32x32 frontend/public/favicon.ico
```
