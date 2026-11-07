# A1. Project context

You are building a Used Car Price Prediction System as an academic machine learning project. A user enters the specification of a used car (brand, model, age, kilometres driven, fuel, transmission, engine size, power, mileage, seats, seller type) and the system returns an estimated selling price in Indian rupees with a plausible range. The project is graded against four requirements:

1. Data is cleansed and pre-processed, with every cleaning decision justified.

2. A complete EDA process is demonstrated (univariate, bivariate, multivariate, with written interpretation).

3. Suitable algorithms are selected and models are built and tuned in a defensible, leak-free way.

4. Models are evaluated with proper metrics, prediction results are produced, and a GUI demonstrates the results.

The dataset is data/raw/cardekho_dataset.csv. It is read-only: never modify or overwrite it. All cleaning produces new files under data/processed/.

# A2. Fixed tech stack (do not substitute)

Use exactly the stack below. If you believe something else is better, write the proposal in decisions.md under "Open questions" and continue with the stack listed. Do not install extra libraries without logging the reason.

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Type hints on all public functions |
| Environment | venv + requirements.txt (pinned versions) | Create .venv, never install globally |
| Data | pandas, numpy | No chained assignment; no inplace=True |
| Static plots (EDA) | matplotlib + seaborn | One shared style module, see A7 |
| Interactive plots (GUI) | plotly | Same palette as the static plots |
| ML | scikit-learn, XGBoost, LightGBM | Everything inside sklearn.pipeline.Pipeline / ColumnTransformer |
| Tuning | Optuna (or RandomizedSearchCV if Optuna fails to install) | Fixed seeds, logged trials |
| Explainability | SHAP (TreeExplainer) plus permutation importance | SHAP is optional if installation fails; log it |
| Persistence | joblib | Save the full pipeline, not just the estimator |
| GUI | Streamlit (multi-page app) | Custom theme, see A7 |
| Notebooks | Jupyter, but only for EDA and the narrative evaluation | All reusable logic lives in src/, notebooks import it |
| Quality | pytest, ruff (lint and format), pre-commit optional | Tests must pass before a phase is declared done |
| Config | YAML file configs/config.yaml | Paths, seeds, split ratios, model grids. No magic numbers in code. |
| Version control | git with small commits | Message format in A4 |

# A3. Repository layout

Create this structure in Phase 0 and keep to it. New folders need a line in decisions.md.

## Folder structure

```text
used-car-price-predictor/

  AGENTS.md                 # this master brief

  decisions.md              # decision log (see A5)

  README.md                 # how to install, run, reproduce

  requirements.txt

  configs/config.yaml

  data/raw/                 # read-only original CSV

  data/interim/             # after cleaning

  data/processed/           # train/test splits

  notebooks/01_data_audit.ipynb

  notebooks/02_eda.ipynb

  notebooks/03_model_evaluation.ipynb

  src/carprice/**\_\_init\_\_**.py

  src/carprice/config.py    # loads YAML, exposes typed settings

  src/carprice/data.py      # load, validate, clean

  src/carprice/features.py  # feature engineering + ColumnTransformer

  src/carprice/train.py     # training, CV, tuning, saving

  src/carprice/evaluate.py  # metrics, residual analysis, segment reports

  src/carprice/predict.py   # load pipeline, predict, interval

  src/carprice/viz.py       # shared plotting style and helpers

  app/Home.py               # Streamlit entry point

  app/pages/                # Predict, Market Explorer, Model Performance, About

  app/components/           # reusable UI pieces, theme.css

  .streamlit/config.toml    # theme

  models/                   # saved pipelines + metadata json

  reports/figures/          # every figure used in the report

  reports/tables/           # every table as csv

  reports/final_report.md   # written report (also exported to docx/pdf)

  tests/

```

# A4. Coding standards (apply to every file, every phase)

The agent must treat these as hard rules. A phase that violates them is not done, even if the output looks right.

## Structure and readability

- Logic in src/, narrative in notebooks. A notebook may call functions and display results; it must not contain a function longer than 10 lines.

- Small functions, one job each, with type hints and a docstring stating inputs, outputs and any assumptions. Maximum roughly 40 lines per function.

- No magic numbers or hard-coded paths. Read them from configs/config.yaml through src/carprice/config.py. Use pathlib.Path, never string concatenation.

- Descriptive names (vehicle_age_years, not va). snake_case for functions and variables, PascalCase for classes.

- No dead code, no commented-out code, no print debugging. Use the logging module with a configured logger.

- No copy-paste. If the same snippet appears twice, extract it into a function.

## Correctness and reproducibility

- Set one global random seed (from config) and pass it to every splitter, model and sampler. Re-running a script must give identical numbers.

- No data leakage. Split first. Fit every imputer, scaler, encoder and selector on the training fold only, which in practice means inside a Pipeline. Never compute statistics on the full dataset and then split.

- Never mutate the raw data. Cleaning functions take a DataFrame and return a new one. Avoid inplace=True and chained assignment.

- Validate inputs at boundaries. data.py checks schema, dtypes and value ranges in a plain validate_schema() function that raises clear errors (do not add pandera or similar libraries).

- Assert, do not assume. After each cleaning step log rows before and after. If a step removes more than 2 percent of rows, stop and record why in decisions.md.

- Handle errors explicitly. No bare except:. Fail early with a helpful message.

## Testing, linting and version control

- Every module in src/ gets tests in tests/ (pytest). At minimum: cleaning rules, the feature pipeline output shape and column names, the predict function on a known row, and a check that the saved model reloads and predicts.

- Run ruff check and ruff format before declaring a phase done. Zero warnings.

- Commit at the end of each task, not only at the end of a phase. Message format: phase-N: imperative summary (for example phase-2: add bivariate price analysis).

- Pin dependency versions in requirements.txt after installing (pip freeze filtered to direct dependencies is acceptable).

## Communication

- Explain in plain language. Markdown cells in notebooks must say what is being checked, what was found and what it means for the next step. A chart without a written takeaway is incomplete.

- Do not invent numbers. Every figure quoted in a report or the GUI must come from a file in reports/tables/ or the saved model metadata.

- When uncertain, ask. If an instruction is ambiguous or a result looks wrong, stop and ask the student instead of guessing.

# A5. The decision log (decisions.md)

The agent must create decisions.md in Phase 0 and append to it whenever it makes a choice that a reasonable person could have made differently. This file is evidence for the viva and for the report, so it must be specific and honest.

## Required entry format

## D-014: Cap km_driven at 500,000 instead of dropping rows

- **Date / Phase:** 2026-11-08 / Phase 2

- **Context:** 12 rows have km_driven above 500,000 (max 3,800,000). Cars are 3 to 9 years old, so these are implausible.

- **Options considered:** (a) drop rows, (b) cap at the 99.9th percentile, (c) keep and log-transform.

- **Decision:** (b) cap at 500,000 and add a boolean flag column `km_was_capped`.

- **Reason:** keeps all rows, limits leverage on linear models, flag lets trees learn if it matters.

- **Evidence:** reports/tables/outlier_summary.csv, notebooks/01_data_audit.ipynb cell 23.

- **Impact / how to reverse:** change `cleaning.km_cap` in config.yaml and re-run `make clean`.

Minimum content: every cleaning rule, every dropped or merged column, the target transformation, the train/test split strategy, each candidate algorithm and why it was included or rejected, the tuning budget, the final model choice, the metric choices, every GUI design choice, and every deviation from this plan. The file also needs a top section "Assumptions" and a bottom section "Open questions for the student".

# A6. Working protocol for each phase

1. Read AGENTS.md, decisions.md and the phase instructions in full before writing code.

2. Write a short plan (5 to 10 lines) at the top of your reply, then execute it without waiting for approval.

3. Work task by task. After each task run the relevant tests and commit.

4. Stay in scope. Do not start work belonging to a later phase and do not refactor earlier phases unless a test fails.

5. At the end, run the full acceptance checklist for the phase. Fix failures yourself.

6. Finish with a Phase report containing: what was built (files), what was decided (new D-xxx ids), numbers worth knowing, problems found, and the exact command to reproduce the phase. Then stop and wait for the student to say "proceed".

# A7. UI and visual design system

The GUI and every chart must look deliberately designed, not like a default template. The agent must not use default Streamlit red/pink accents, rainbow chart palettes, emoji-heavy headings, purple-to-blue gradients, centred walls of text, or stock hero banners.

## Palette (use these exact values everywhere)

| Role | Hex | Used for |
|---|---|---|
| Background | #F7F5F0 | Page background (warm off-white) |
| Surface | #FFFFFF | Cards, input panels, tables |
| Ink (text) | #14213D | Headings and body text |
| Muted text | #6B7280 | Captions, axis labels, helper text |
| Primary (teal) | #0E7C7B | Buttons, selected states, main series in charts |
| Accent (amber) | #F2A541 | One highlight per view: the predicted price marker, key callout |
| Border | #E3DED3 | 1 px card and divider lines |
| Positive | #2E7D32 | Good-fit indicators |
| Negative | #C0392B | Errors and warnings only |
| Chart categorical | #0E7C7B, #14213D, #F2A541, #7A9E9F, #B5651D, #8D99AE | Maximum 6 series; use grey for "other" |
| Chart sequential | white #FFFFFF to teal #0E7C7B | Heatmaps and density; never a rainbow map |

## Typography, spacing and layout

- Fonts: headings in "DM Serif Display" or "Fraunces"; body and UI in "Inter"; numbers and metrics in "JetBrains Mono". Load via Google Fonts CSS import in app/components/theme.css; fall back to system sans-serif.

- Spacing scale: multiples of 4 px (4, 8, 12, 16, 24, 32, 48). Card padding 24 px, corner radius 12 px, 1 px border in the Border colour, no heavy shadows.

- Hierarchy: one H1 per page, short sentence-case labels, left-aligned text. Big numbers get a small muted caption underneath.

- Layout: inputs on the left (about 40 percent), result on the right (about 60 percent). Group inputs into "Vehicle", "Condition" and "Engine and fuel" sections.

- Numbers: show prices in Indian grouping with the rupee sign (for example ₹ 5,50,000) and also in lakhs (₹ 5.50 lakh). Keep units next to every value.

- Charts: no 3D, no pie charts for more than 3 slices, no chart junk. Title states the finding ("Prices fall about 8 percent per year of age"), axes are labelled with units, gridlines are light grey, direct labels where possible.

- Accessibility: contrast at least 4.5:1 for text, never rely on colour alone, every chart has a text summary beneath it.

- States: every page must have a sensible empty state, loading state and error message in plain language.

# A8. Definition of done (project level)

☐   A fresh clone plus pip install -r requirements.txt plus the commands in README.md reproduces the cleaned data, the model, the figures and the GUI.

☐   pytest passes, ruff check is clean.

☐   decisions.md has an entry for every non-trivial choice and no unexplained numbers.

☐   Every figure in reports/final_report.md exists in reports/figures/ and every table in reports/tables/.

☐   The GUI launches with streamlit run app/Home.py and returns a prediction in under 2 seconds.
