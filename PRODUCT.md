# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: the examiners at the academic viva. They watch a projected screen or a laptop, judge the project's rigour and polish, and click around the app for a few minutes. Their job is to confirm that the four graded requirements (cleaning, EDA, leak-free modelling, evaluation with a GUI) hold, and that every number on screen can be traced to a source.

Secondary (not confirmed as a design driver): someone pricing a real used car in India.

## Product Purpose

A Used Car Price Prediction System. A user describes a used car (brand, model, age, kilometres driven, seller type, fuel, transmission, mileage, engine size, power, seats). The app returns an estimated selling price in rupees, an 80% likely range, the factors that moved the estimate, and plain-language cautions when the car falls in a group the model prices poorly. Success at the viva means the examiner can run an estimate, see why the price came out that way, and verify model quality without leaving the app.

## Positioning

Every figure shown is traceable: estimates come from a saved, tuned XGBoost pipeline, and every metric or claim comes from a file in `reports/tables/` or the model metadata. The app shows its own uncertainty (calibrated 80% range, segment cautions) instead of a single bare number.

## Operating Context

- A Streamlit multi-page app launched with `streamlit run app/Home.py`, demoed live at a viva.
- Pages: Price estimator (single car + batch CSV upload), Market explorer (filter the cleaned listings), Model performance (test metrics, residuals, segments), About and data (cleaning, method).
- It sits alongside notebooks, `reports/final_report.md` and `decisions.md`, which the examiner may cross-reference.

## Capabilities and Constraints

- Fixed stack per AGENTS.md A2: Python 3.11, Streamlit, plotly for GUI charts, matplotlib/seaborn for static EDA figures. No new dependencies without a logged reason.
- Prediction must return in under 2 seconds.
- Prices shown in Indian grouping with the rupee sign (₹ 5,50,000) and in lakhs (₹ 5.50 lakh), units next to every value.
- No invented numbers: every on-screen figure must come from `reports/tables/` or model metadata.
- Every page needs an empty, loading and error state in plain language; no Python tracebacks to users.
- Light and dark modes exist (D-036).
- Coding standards in AGENTS.md A4 apply (type hints, small functions, ruff clean, tests pass, decisions logged).

## Brand Commitments

- Name: "Used Car Price Predictor".
- Standing visual preference (2026-11-26): the category standard, a clean product dashboard played straight, with no themed costume. The craft bar is Linear, the Vercel dashboard and the Stripe dashboard. This replaced the original A7 palette and fonts; see `decisions.md` D-045 and AGENTS.md A7.

## Evidence on Hand

- `reports/tables/*.csv`: test metrics, interval coverage, segment errors, train-vs-test, etc.
- `models/` saved pipeline + metadata JSON.
- `data/interim`, `data/processed`: cleaned listings used by the market explorer.
- `reports/figures/`: static figures, including GUI screenshots referenced by README.
- No testimonials, users, or external validation exist; do not fabricate any.

## Product Principles

1. Show the evidence, not a claim: every number links back to a source file.
2. Honest uncertainty: the range and cautions sit next to the price, never hidden.
3. A viva demo in minutes: the main flow (describe a car, read the estimate and why) must work in one screen with no explanation.
4. Plain language over jargon: every chart says what it found in a sentence.

## Accessibility & Inclusion

Text contrast at least 4.5:1, never rely on colour alone, a text summary beneath every chart (carried over from the original A7 brief).
