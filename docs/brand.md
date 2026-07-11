# EduRisk AI — Brand Guide

Visual identity system for the EduRisk AI project.

---

## Logo

### Primary Logo
- **File**: `assets/images/logo.svg`
- **Usage**: GitHub profile, documentation, presentations
- **Description**: Shield shape with neural network nodes and lightning bolt — represents protection through intelligence

### Wordmark
```
EduRisk AI
```
- Font: Inter (Bold, 800 weight)
- Color: `#14b8a6` (teal) for "EduRisk", `#e5e7eb` (light gray) for "AI"

### Logo Variants
| Variant | Background | Use Case |
|---------|-----------|----------|
| Primary | Dark (#111827) | Default, dark UI |
| Light | White (#ffffff) | Print, light backgrounds |
| Icon only | Any | Favicon, small spaces |

---

## Color Palette

### Primary Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| **Teal** | `#14b8a6` | 20, 184, 166 | Primary brand, CTAs, accents |
| **Teal Dark** | `#0d9488` | 13, 148, 136 | Hover states, emphasis |
| **Cyan** | `#38bdf8` | 56, 189, 248 | Secondary accent, links |

### Background Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| **Background** | `#0a0a0f` | 10, 10, 15 | Page background |
| **Surface** | `#111827` | 17, 24, 39 | Cards, panels |
| **Surface Hover** | `#1a2332` | 26, 35, 50 | Interactive hover |
| **Border** | `#1f2937` | 31, 41, 55 | Dividers, borders |
| **Border Light** | `#374151` | 55, 65, 81 | Emphasized borders |

### Text Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| **Text** | `#e5e7eb` | 229, 231, 235 | Primary text |
| **Text Dim** | `#9ca3af` | 156, 163, 175 | Secondary text |
| **Text Muted** | `#6b7280` | 107, 114, 128 | Tertiary text, labels |

### Semantic Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| **Green** | `#22c55e` | 34, 197, 94 | Low risk, success, positive |
| **Yellow** | `#f59e0b` | 245, 158, 11 | Medium risk, warning |
| **Red** | `#ef4444` | 239, 68, 68 | High risk, error, danger |
| **Orange** | `#f97316` | 249, 115, 22 | Accent, attention |

### Gradient

```css
background: linear-gradient(135deg, #14b8a6, #38bdf8);
```
- Used for: CTA buttons, logo, animated elements
- CSS class: `bg-gradient-to-r from-teal to-cyan`

---

## Typography

### Font Family
```css
font-family: "Inter", system-ui, -apple-system, sans-serif;
```

### Type Scale

| Element | Size | Weight | Usage |
|---------|------|--------|-------|
| H1 | 2rem (32px) | 800 (ExtraBold) | Page titles |
| H2 | 1.5rem (24px) | 700 (Bold) | Section headers |
| H3 | 1.125rem (18px) | 600 (SemiBold) | Card titles |
| Body | 1rem (16px) | 400 (Regular) | Paragraphs |
| Small | 0.875rem (14px) | 400 | Secondary text |
| Caption | 0.75rem (12px) | 500 (Medium) | Labels, badges |
| Micro | 0.625rem (10px) | 600 | Uppercase tracking labels |

### Letter Spacing
- Uppercase labels: `letter-spacing: 1.5px` (`tracking-[1.5px]`)
- Navigation: `letter-spacing: 0.5px`

---

## Icons

### Style
- **Outline/Stroke**: 1.5px stroke weight
- **Size**: 20x20px (standard), 16x16px (small), 24x24px (large)
- **Color**: Inherit from parent (`currentColor`)

### Icon Library
Custom SVG icons inline — no external icon library dependency.

Key icons used:
- ⚡ Lightning bolt (risk/alert) — header logo
- 🧪 Flask/beaker (AI analysis) — empty state
- 📊 Bar chart — analytics
- 🔒 Shield — protection/safety

---

## Spacing & Layout

### Grid
- **Max width**: 1280px (`max-w-7xl`)
- **Padding**: 24px horizontal (`px-6`)
- **Gap**: 32px between columns (`gap-8`)

### Border Radius
| Element | Radius |
|---------|--------|
| Cards | 16px (`rounded-2xl`) |
| Buttons | 12px (`rounded-xl`) |
| Badges | 8px (`rounded-lg`) |
| Inputs | 12px (`rounded-xl`) |
| Avatars | 50% (circle) |

### Shadows
```css
/* Subtle elevation */
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);

/* Teal glow (CTAs) */
box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2);

/* Teal glow (hover) */
box-shadow: 0 8px 24px rgba(20, 184, 166, 0.3);
```

---

## Animations

### Easing
```css
transition: all 200ms ease-out;  /* Standard */
transition: all 300ms ease-out;  /* Emphasis */
```

### Framer Motion Presets
| Name | Properties | Usage |
|------|-----------|-------|
| Fade In | opacity: 0 → 1 | Page load |
| Slide Up | y: 20 → 0, opacity: 0 → 1 | Results appear |
| Scale | scale: 0.95 → 1 | Card hover |

### Loading Spinner
- Teal ring with transparent gap
- 1-second spin animation
- 4px stroke width

---

## Application of Brand

### GitHub Repository
- **Banner**: 1280×640px, dark background with logo + tagline
- **Social Preview**: 1200×630px (Open Graph)
- **README**: Badge colors match palette

### Next.js Frontend
- All colors defined in `globals.css` via CSS custom properties
- Tailwind theme extended with brand tokens
- Dark mode only (no light mode)

### FastAPI Swagger
- Default Swagger UI (no customization)
- Could add custom CSS for brand colors in future

### Gradio Dashboard
- Uses Gradio's default dark theme
- Custom CSS for brand accents

---

## File Naming Convention

```
assets/
├── images/
│   ├── logo.svg              # Primary logo
│   ├── banner.png            # GitHub banner (1280x640)
│   ├── screenshot-*.png      # App screenshots
│   └── *.png                 # Generated charts
```

---

## Usage Rules

1. **Never use teal on dark backgrounds below 4.5:1 contrast** — use `#e5e7eb` text instead
2. **Red is reserved for high-risk/error states** — don't use for decorative purposes
3. **Gradients always go left-to-right, top-to-bottom** — never radial
4. **Logo must have clear space** equal to the height of the "E" in "EduRisk"
5. **Minimum logo size**: 32px height for icon, 120px width for wordmark
