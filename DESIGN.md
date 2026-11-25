---
name: Used Car Price Predictor
description: A clean product dashboard that prices a used car and shows its evidence.
colors:
  page: "#FAFAFA"
  surface: "#FFFFFF"
  subtle: "#F4F4F5"
  ink: "#18181B"
  muted: "#71717A"
  border: "#E4E4E7"
  border-strong: "#D4D4D8"
  control-border: "#8B8B94"
  grid: "#F0F0F2"
  primary: "#2F5BEA"
  primary-hover: "#2449C9"
  primary-ring: "rgba(47, 91, 234, 0.22)"
  primary-tint: "rgba(47, 91, 234, 0.12)"
  highlight: "#D97706"
  positive: "#15803D"
  negative: "#DC2626"
  caution-bg: "#FFFBEB"
  caution-border: "#FDE68A"
  caution-ink: "#B45309"
  error-bg: "#FEF2F2"
  error-border: "#FECACA"
  series-teal: "#0D9488"
  series-rose: "#BE185D"
  page-dark: "#0B0B0D"
  surface-dark: "#141417"
  subtle-dark: "#1B1B1F"
  ink-dark: "#EDEDEF"
  muted-dark: "#A1A1AA"
  border-dark: "#27272A"
  border-strong-dark: "#3F3F46"
  grid-dark: "#1E1E22"
  primary-dark: "#7C9BFF"
  highlight-dark: "#F59E0B"
  positive-dark: "#4ADE80"
  negative-dark: "#F87171"
  caution-ink-dark: "#FBBF24"
  series-teal-dark: "#2DD4BF"
  series-rose-dark: "#F472B6"
typography:
  display:
    fontFamily: "Geist, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "48px"
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: "-0.035em"
    fontFeature: "tnum"
  headline:
    fontFamily: "Geist, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "28px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.025em"
  lede:
    fontFamily: "Geist, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  title:
    fontFamily: "Geist, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    letterSpacing: "-0.005em"
  body:
    fontFamily: "Geist, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Geist, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "13px"
    fontWeight: 500
  caption:
    fontFamily: "Geist, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.55
  mono:
    fontFamily: "Geist Mono, ui-monospace, Cascadia Mono, monospace"
    fontSize: "0.95em"
    fontWeight: 400
    fontFeature: "tnum"
rounded:
  sm: "6px"
  md: "8px"
  pill: "999px"
spacing:
  s1: "4px"
  s2: "8px"
  s3: "12px"
  s4: "16px"
  s5: "24px"
  s6: "32px"
  s7: "48px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    height: "40px"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "{colors.surface}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    height: "40px"
  button-secondary-hover:
    backgroundColor: "{colors.subtle}"
    textColor: "{colors.ink}"
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "{spacing.s5}"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
  nav-link:
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "4px 12px"
  nav-link-active:
    backgroundColor: "{colors.subtle}"
    textColor: "{colors.ink}"
  note:
    backgroundColor: "{colors.subtle}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "12px 16px"
  note-caution:
    backgroundColor: "{colors.caution-bg}"
    textColor: "{colors.ink}"
  note-error:
    backgroundColor: "{colors.error-bg}"
    textColor: "{colors.ink}"
  status-pill:
    backgroundColor: "{colors.caution-bg}"
    textColor: "{colors.caution-ink}"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
  metric-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.headline}"
    rounded: "{rounded.md}"
    padding: "{spacing.s5}"
---

# Design System: Used Car Price Predictor

## Overview

**Creative North Star: "The Quiet Instrument Panel"**

The category standard, a clean product dashboard, played straight at the craft level of Linear, the Vercel dashboard and the Stripe dashboard. Zinc neutrals carry almost every pixel; one blue does the acting (buttons, selection, the main chart series) and one amber marks the single thing worth looking at on a view. Hierarchy comes from size, weight and space, not from colour or decoration. The estimate is the largest object on the estimator, and every chart sits in a bordered card with a finding-style title above it and a plain-English takeaway below it.

Density is calm, not sparse: 14px body text, 24px card padding, a 1200px content column, a two-column work area that collapses to one below 960px. Surfaces are flat, separated by 1px hairlines. Light and dark modes are the same system: every token has a dark counterpart and switches with Streamlit's own theme picker without a rerun.

What it refuses: the template look of a default Streamlit app (red accent, sidebar navigation, equal cards everywhere, weak hierarchy) and decorative colour.

**Key Characteristics:**
- Zinc neutral ramp, one blue action colour, one amber highlight per view.
- Geist throughout, tabular figures for every number.
- Flat surfaces, 1px borders, 8px cards, 6px controls.
- Top-tab navigation in a white header; no sidebar.
- Every chart framed as title, chart, one-sentence takeaway.

## Colors

A cool zinc greyscale with a saturated blue for action and a warm amber kept for one mark per view.

### Primary
- **Action Blue** (primary): primary buttons, focus rings, selected radio and segmented states, slider, links, the main series in every chart, and the brand mark tile. In dark mode, text, lines and links in this role brighten to primary-dark for contrast; filled buttons keep the light blue with white text.
- **Pressed Blue** (primary-hover): hover fill for primary buttons only.
- **Blue Wash** (primary-tint, primary-ring): the 80% range band, the what-if uncertainty band, the ±10% error band, the chosen-model row in tables, text selection, and the 3px focus halo.

### Secondary
- **Signal Amber** (highlight): the single highlight per data view. The estimate diamond on the range bar, the top brand on Market explorer, the weakest segment on Model performance, the diamond in the brand mark.

### Tertiary
- **Series Teal** and **Series Rose** (series-teal, series-rose): fourth and fifth chart series only, when a categorical chart needs them. Never UI chrome.

### Neutral
- **Paper** (page): page background.
- **Card White** (surface): cards, header bar, inputs, chart plot area.
- **Zinc Wash** (subtle): hover and active nav background, neutral note fill, secondary button hover.
- **Zinc Ink** (ink): headings, body, widget labels, the perfect-prediction line, the selected tab underline.
- **Zinc Slate** (muted): ledes, captions, axis ticks, inactive tabs and nav links.
- **Hairline** (border), **Firm Hairline** (border-strong), **Control Outline** (control-border): card and divider lines, hover outlines on fields, unselected radio rings (3:1 against the card).
- **Grid** (grid): chart gridlines.
- **Neutral series** (control-border value, slate slot): de-emphasised and "other" marks, negative effect bars.

### Status
- **Positive** and **Negative**: good-fit indicators and errors only. **Caution** (caution-bg, caution-border, caution-ink) and **Error** (error-bg, error-border) tints fill notes and the stale-result pill.

### Named Rules
**The One Amber Rule.** Amber appears at most once per view, on the data mark that answers the page's question. Pages that only explain (About and data) carry none.

**The Blue Acts Rule.** Blue means "this does something or this is the main series". Tabs and nav use ink, not blue, for their selected state.

**The Slot Rule.** Plotly figures never take hex values. They draw with placeholder colours #000001 to #000010 that Streamlit swaps for the active theme's ten chart slots: six series (primary, ink, highlight, teal, rose, slate), then muted, border, grid, surface.

## Typography

**Display Font:** Geist (with system-ui, Segoe UI, sans-serif)
**Body Font:** Geist
**Label/Mono Font:** Geist Mono (with ui-monospace, Cascadia Mono)

**Character:** One grotesque family at three weights (400, 500, 600) does all the work, tightened with negative tracking at large sizes. Mono is reserved for code and identifiers; numbers stay in Geist with tabular figures.

### Hierarchy
- **Display** (600, 48px, 1.05, -0.035em, tabular): the estimate only. 40px below 960px.
- **Headline** (600, 28px, 1.2, -0.025em): the one H1 per page and metric-card values. H1 drops to 24px below 480px.
- **Lede** (400, 16px, 1.5, muted, max 68ch): one sentence under the H1.
- **Title** (600, 14px): section and card titles, written as findings for charts.
- **Body** (400, 14px, 1.55): notes, tables, chart text and tick labels (Plotly at 14px).
- **Label** (500, 13px): widget labels, metric labels, status pill; nav and tabs use 500 at 14px.
- **Caption** (400, 13px, 1.55, muted): chart takeaways, provenance lines, metric captions.

### Named Rules
**The Tabular Rule.** Every price, metric, slider value and number input uses tabular figures.

**The Three Weights Rule.** 400, 500 and 600 only. No bold 700, no light weights.

## Layout

Top navigation in a white header with a bottom hairline; the brand mark and wordmark on the left, page tabs beside them, the theme menu on the right. Content sits in a 1200px column with 80px top padding and 48px bottom. The estimator splits roughly 40/60: inputs on the left grouped into Vehicle, Condition, and Engine and fuel (hairline plus 16px between groups), result cards stacked on the right; full-width cards follow underneath. Vertical rhythm between blocks is 16px; inside cards 12px.

Spacing uses the 4px scale (4, 8, 12, 16, 24, 32, 48). Below 960px every column row stacks to full width with 16px side gutters; below 640px the wordmark hides and only the mark remains; below 480px card padding drops to 16px.

## Elevation & Depth

Flat. Depth comes from tonal layering (page Paper, cards Card White, hover Zinc Wash) and 1px hairlines. The only shadows are functional: the 3px blue focus halo on buttons and fields. Overlays (select menus, tooltips) keep Streamlit's native shadow.

### Shadow Vocabulary
- **Focus halo** (`box-shadow: 0 0 0 3px var(--cp-primary-ring)`): keyboard focus on buttons and focused fields.

### Named Rules
**The Hairline Rule.** Cards are separated by a 1px border, never by a drop shadow.

## Shapes

Gently rounded rectangles: 8px for cards and panels, 6px for buttons, fields, notes, nav links and images, fully round only for the status pill and its dot. The brand mark is a 7px-rounded 28px tile. Empty states use a dashed border on a transparent card.

## Components

### Buttons
- **Shape:** gently curved (6px), 40px minimum height, weight 500.
- **Primary:** Action Blue fill, white text, full width under the input panel.
- **Hover / Focus:** fill darkens to Pressed Blue over 150ms (ease-out-expo `cubic-bezier(0.16, 1, 0.3, 1)`); focus shows the 3px blue halo, no outline.
- **Secondary:** white fill, hairline border; hover to Zinc Wash with Firm Hairline.
- **Disabled:** 50% opacity, not-allowed cursor.

### Cards / Containers
- **Corner Style:** 8px.
- **Background:** Card White in both modes (dark surface #141417).
- **Shadow Strategy:** none (see The Hairline Rule).
- **Border:** 1px Hairline.
- **Internal Padding:** 24px (16px below 480px).

### Metric card
A label (13px muted), a 28px semibold tabular value, and a muted caption; a row of them shares one height.

### Inputs / Fields
- **Style:** white field, 1px border, 6px radius, tabular number entry; widget labels 13px ink.
- **Hover:** border to Firm Hairline.
- **Focus:** border to Action Blue plus the 3px blue halo.

### Navigation
Top tabs: 14px weight 500 muted text with a Material icon, 6px rounded, 4px by 12px padding. Hover and current page get a Zinc Wash fill and ink text; focus-visible draws a 2px blue outline offset 2px. In-page tabs use muted text, ink when selected, with an ink underline and 24px gaps.

### Notes
Full 1px border and tint, never a side stripe. Neutral (Zinc Wash), caution (amber tints, caution-ink heading) and error (red tints, negative heading). Bold lead phrase, optional bullet list.

### Status pill
Fully rounded amber-tinted badge with a 6px dot, flagging a result that is out of date with the inputs.

### Range bar (signature)
The 80% range as a Blue Wash band with a blue edge on a hairline track, ink ticks at each end with muted "Low" and "High" lakh labels beneath, and the estimate as an amber diamond outlined in ink. It is the page's one amber mark; the estimate above it fades up 4px from a 4px blur over 520ms, the app's one authored motion (removed under reduced motion).

### Chart card
A bordered card: Title-style finding, the Plotly chart (no mode bar, Geist 14px, grid lines in Grid, axes in Hairline, plot on the card surface), then a muted one-sentence takeaway.

## Do's and Don'ts

### Do:
- **Do** take every GUI colour from theme.css tokens in CSS and from chart slot placeholders in Plotly, so both modes switch without a rerun.
- **Do** keep amber to one data mark per view, and none on explanatory pages.
- **Do** give every chart a finding-style title and a takeaway sentence beneath it.
- **Do** show prices with the rupee sign, Indian grouping and a lakh figure, in tabular figures.
- **Do** keep text at 4.5:1 or better on page and cards in both modes, and unselected control outlines at 3:1.

### Don't:
- **Don't** use drop shadows on cards; separate with a 1px hairline.
- **Don't** put a coloured stripe on one side of a note; use a full border and tint.
- **Don't** colour tabs or nav selection blue; ink carries selection there.
- **Don't** add a font weight beyond 400, 500 and 600, or a second display family.
- **Don't** use the A7 report palette (teal #0E7C7B, warm off-white #F7F5F0, DM Serif, Inter) in the GUI. Static matplotlib and SHAP report figures keep it by design (constants at the top of src/carprice/palette.py); they sit outside this system.
