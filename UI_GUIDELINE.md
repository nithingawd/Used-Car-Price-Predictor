GOAL
Build a polished, multi-page Streamlit application that lets a non-technical person estimate a car price, understand the estimate, explore the market data, and inspect model quality. It must follow the design system in A7 exactly.
INPUTS
•	src/carprice/predict.py, models/, reports/tables/, reports/figures/
•	Palette, typography and layout rules from A7
•	data/interim/cars_clean.csv (for the Market Explorer only)
TASKS
7.1 Theme and shared components
1.	Create .streamlit/config.toml with the theme below, then app/components/theme.css (Google Fonts import, card class, metric class, spacing scale, hidden default footer and menu) injected once through a load_theme() helper.
2.	Build reusable components in app/components/: metric_card(), section_header(), range_bar() (Plotly), info_note(), format_inr() (Indian grouping) and format_lakh(). No page may re-implement these.
3.	Cache correctly: st.cache_resource for the predictor, st.cache_data for dataframes. Never reload the model on interaction.
.streamlit/config.toml
[theme]
primaryColor = "#0E7C7B"
backgroundColor = "#F7F5F0"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#14213D"
font = "sans serif"

7.2 Page 1: Price Estimator (Home.py)
1.	Left column, three grouped sections. Vehicle: brand, then model (options filtered by the chosen brand), vehicle age (slider 0 to 20), seller type. Condition: kilometres driven (number input with the sensible range from the data). Engine and fuel: fuel type, transmission, engine (cc), max power (bhp), mileage, seats.
2.	When a model is selected, pre-fill engine, power, mileage, seats and fuel with the most common values for that model (from a lookup built from training data and saved as models/model_specs.csv). The user can override any field.
3.	A primary button "Estimate price". Do not predict on every keystroke.
4.	Right column result card: the estimate as the biggest element (monospace font, rupees and lakh), the 80 percent range as a range_bar with an amber marker, a one-line plain-language message (for example "Similar 5-year-old Hyundai i20s sell for about 4.2 to 5.1 lakh"), and a low-confidence notice when the car is a rare model, very old, or in the luxury band.
5.	Below the result: three small panels. Why this price: top five factors from SHAP for this one car as a horizontal bar chart (positive in teal, negative in grey). Compare: this car against the median of the same brand and age from the data. What if: a line chart of predicted price against age (or km) holding other inputs fixed, with the current choice marked.
6.	A second tab "Batch estimate": upload a CSV with the same columns, validate it, show a preview and offer the predictions as a CSV download. Provide a downloadable template file.
7.	Invalid input shows a short, friendly message beside the field; the app never shows a Python traceback.
7.3 Page 2: Market Explorer
1.	Filter bar: brand multiselect, fuel, transmission, age range, price range.
2.	Four interactive Plotly charts that tell a story: price distribution (log x-axis option), median price by age with count-weighted trend, price vs power scatter coloured by fuel, top brands by median price with listing counts.
3.	A summary strip above the charts: number of listings, median price, median age, with the filter applied.
4.	Every chart has a finding-style title and a short text takeaway below.
7.4 Page 3: Model Performance
1.	Metric cards for the headline metrics from test_metrics.csv (MAE in rupees, within 10 percent accuracy, R-squared, RMSE), each with a muted caption that explains it in plain English.
2.	Interactive predicted vs actual chart (hover shows the car), residual distribution, and the model comparison table from cv_results.csv, with the selected model highlighted.
3.	Segment error chart (by price band, brand, fuel) with a short note on where the model is weaker.
4.	SHAP summary image and a plain-language paragraph on the main drivers.
5.	All numbers are read from reports/tables/; none are typed in the code.
7.5 Page 4: About and Data
1.	Short description of the dataset, a compact cleaning summary table (from cleaning_log.csv), how the model works in four sentences, a limitations box, and the model card (version, training date, library versions).
7.6 Design self-review (mandatory before reporting)
Run the app, capture a screenshot of every page at desktop width and at about 800 px width (use the available browser or screenshot tooling; if none exists, say so and review the CSS and layout code line by line instead). Then answer the checklist below in the phase report. Fix every "no".
Design check	Pass criterion
Palette discipline	Only the A7 colours appear; exactly one amber highlight per view; no default Streamlit red
Typography	Serif display heading, Inter for UI, monospace for numbers; at most three font sizes per page
Hierarchy	The estimated price is the largest element on the page; one H1 per page; labels are short and sentence-case
Spacing and alignment	All spacing on the 4 px scale; cards share padding and radius; no orphaned widgets or ragged columns
Emoji and decoration	No emoji in headings or buttons, no gradients, no stock banner images, no heavy shadows
Charts	Finding-style titles, axis units, light gridlines, palette colours only, text takeaway below each chart
Responsiveness	Usable at about 800 px width: columns stack, nothing is cut off
Empty, loading and error states	Each page has all three, written in plain language
Accessibility	Contrast at least 4.5:1, colour never the only signal, every input labelled

PHASE-SPECIFIC RULES
•	No logic duplicated from src/: the GUI calls PricePredictor and shared helpers only.
•	No hard-coded metric values or colours outside theme.css and a single app/components/palette.py.
•	Prediction response time under 2 seconds including SHAP.
•	Keyboard navigable, labels on every input, contrast at least 4.5:1.
DELIVERABLES
File or artefact	What it must contain
app/Home.py and app/pages/ (3 pages)	Price Estimator with batch tab, Market Explorer, Model Performance, About and Data
app/components/	Theme CSS, palette, reusable components, formatters
.streamlit/config.toml	Theme matching A7
models/model_specs.csv	Default specs per brand and model for pre-filling
reports/figures/gui_*.png	Screenshots of each page for the report
tests/test_app_helpers.py	Formatter tests (for example 550000 becomes ₹ 5,50,000), model-spec lookup tests, and a smoke test using Streamlit AppTest

ACCEPTANCE CHECKLIST (AGENT MUST TICK EVERY BOX BEFORE REPORTING)
☐	streamlit run app/Home.py starts with no warnings and produces a prediction for a default car.
☐	Brand selection filters the model list; selecting a model pre-fills the specs.
☐	Result shows point estimate, range, explanation and what-if chart.
☐	Batch upload works and bad files give a clear message.
☐	Every page uses only the A7 palette and fonts.
☐	All design self-review answers are "yes".
DECISIONS TO RECORD IN DECISIONS.MD
•	Page structure and why
•	How SHAP is computed in the app and its latency
•	Treatment of rare and luxury cars in the UI
•	Any design deviation from A7
Stop here
Write the phase report described in A6 and wait for the student to say "proceed". Do not begin the next phase.