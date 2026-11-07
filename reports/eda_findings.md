# Exploratory Data Analysis: Key Findings and Modelling Implications

This document presents the core empirical findings derived from the exploratory data analysis of the cleaned Cardekho used car dataset (15,244 records, 13 features) stored in `data/interim/cars_clean.csv`. Every finding is supported by high-resolution visualizations in `reports/figures/` and quantitative evidence in `reports/tables/`.

---

## Part 1: Numbered Summary of Key Findings

### 1. Extreme Right-Skew of Price Mandates Logarithmic Transformation
- **Finding:** The distribution of raw used car selling prices is severely right-skewed with a sample skewness coefficient of **10.11**. While the median vehicle price is **₹5.59 lakh** and 75% of vehicles retail below **₹8.25 lakh**, the upper tail stretches to **₹3.95 crore** (mean ₹7.75 lakh, standard deviation ₹8.95 lakh).
- **Evidence:** Applying a natural logarithmic transform ($\log(1 + y)$) compresses the tail and reduces skewness to **0.56**, producing a near-Gaussian distribution well-suited for linear and tree-based loss functions.
- **Supporting Artefacts:** Figure `reports/figures/eda_01_selling_price_raw_vs_log.png`, Table `reports/tables/eda_numeric_summary.csv`.

---

### 2. Engine Power and Displacement Are the Strongest Continuous Predictors
- **Finding:** Maximum rated power (`max_power`) exhibits the single strongest monotonic association with selling price, boasting a Spearman rank correlation coefficient of **$\rho = 0.69$** (Pearson $r = 0.75$ with log price). Engine displacement (`engine`) closely follows with **$\rho = 0.58$**.
- **Evidence:** Scatter plots and binned median trends reveal a steep, non-linear exponential valuation curve: vehicles generating under 80 bhp trade at a median of ~₹3.5 lakh, whereas premium engines producing over 180 bhp consistently command valuations exceeding ₹20 lakh.
- **Supporting Artefacts:** Figures `reports/figures/eda_20_scatter_price_vs_power.png`, `reports/figures/eda_21_scatter_price_vs_engine.png`, Table `reports/tables/eda_spearman_correlation.csv`.

---

### 3. Substantial Premium for Automatic Transmissions Across All Power Tiers
- **Finding:** Automatic transmission vehicles command an overall median price of **₹13.00 lakh**, compared to just **₹4.75 lakh** for manual counterparts—a gross ratio of **2.74x**.
- **Evidence:** To ensure this premium is not merely an artefact of larger engines, a controlled Mann-Whitney U test was performed within each power quartile (Q1: 38–74 bhp, Q2: 74–88 bhp, Q3: 88–118 bhp, Q4: 118–626 bhp). In every tier, automatics maintained a statistically significant premium ($p < 0.001$) with substantial rank-biserial effect sizes ranging from **$r_{rb} = 0.26$ to $0.61$**.
- **Supporting Artefacts:** Figures `reports/figures/eda_15_price_by_transmission.png`, `reports/figures/eda_24_interaction_power_transmission.png`, Table `reports/tables/eda_hypothesis_tests.csv`.

---

### 4. Vehicle Age Follows Predictable Exponential Depreciation
- **Finding:** Vehicle age (`vehicle_age`) is the primary downward valuation driver ($\rho = -0.60$ with selling price). Depreciation is steepest during the first four years of vehicle life, moderating into a steady linear decay in log space.
- **Evidence:** Across the top volume manufacturers (Maruti, Hyundai, Honda, Toyota, Ford), vehicles experience an estimated **8% to 12% average annual price loss** across years 1 through 7. Toyota exhibits exceptional value retention, maintaining a median price of ₹10.5 lakh across its fleet.
- **Supporting Artefacts:** Figures `reports/figures/eda_18_scatter_price_vs_age.png`, `reports/figures/eda_22_depreciation_curves.png`, Table `reports/tables/eda_depreciation_rates.csv`.

---

### 5. Diesel Powertrains Command a Sizable Market Premium Over Petrol
- **Finding:** Diesel vehicles trade at a median price of **₹7.00 lakh** (mean ₹9.28 lakh), significantly above petrol vehicles at **₹4.50 lakh** (mean ₹6.29 lakh). CNG vehicles trade at a median of **₹3.60 lakh**, and LPG at **₹2.12 lakh**.
- **Evidence:** A Kruskal-Wallis test across fuel types confirms significant distributional variance ($H = 1756.2, p < 0.001$). Faceted age analysis shows that while diesel models maintain a higher base value across all model years due to higher original showroom MSRPs, their annual percentage depreciation slope closely mirrors that of petrol vehicles.
- **Supporting Artefacts:** Figures `reports/figures/eda_14_price_by_fuel.png`, `reports/figures/eda_25_interaction_age_fuel.png`, Tables `reports/tables/eda_fuel_medians.csv`, `reports/tables/eda_hypothesis_tests.csv`.

---

### 6. Extreme Brand Hierarchy and Volume Concentration
- **Finding:** The market exhibits extreme brand tiering. The top two manufacturers—Maruti (4,948 cars, 32.5%) and Hyundai (2,982 cars, 19.6%)—account for **52.0%** of all market listings.
- **Evidence:** Medians span an order of magnitude: budget commuter brands (Maruti: ₹4.5 lakh, Tata: ₹4.7 lakh) sit at the base; mid-market stalwarts (Toyota: ₹10.5 lakh, MG: ₹15.8 lakh, Isuzu: ₹16.0 lakh) occupy the middle; European luxury marques (Audi: ₹21.0 lakh, Mercedes-Benz: ₹23.0 lakh, BMW: ₹24.0 lakh) occupy the upper tier; and exotic supercars (Porsche: ₹47.5 lakh, Ferrari: ₹3.95 crore) form the extreme ceiling.
- **Supporting Artefacts:** Figures `reports/figures/eda_08_cat_brand.png`, `reports/figures/eda_13_price_by_brand.png`, Table `reports/tables/eda_brand_medians.csv`.

---

### 7. Commercial Dealership Premium Over Private Individual Sellers
- **Finding:** Cars sold through commercial dealers command a median price of **₹6.50 lakh** (both standard Dealers and Trustmark Certified Dealers), compared to **₹4.50 lakh** for direct Individual sellers.
- **Evidence:** A Mann-Whitney U test confirms this difference is statistically significant ($U = 3.96 \times 10^7, p < 0.001$, rank-biserial effect size $r_{rb} = 0.26$). Dealers predominantly inventory newer, lower-mileage vehicles and higher-trim automatic models, adding certified refurbishment value.
- **Supporting Artefacts:** Figure `reports/figures/eda_16_price_by_seller.png`, Tables `reports/tables/eda_seller_medians.csv`, `reports/tables/eda_hypothesis_tests.csv`.

---

### 8. Moderate Odometer Correlation with Non-Linear Plateau
- **Finding:** Odometer mileage (`km_driven`) displays a moderate negative rank correlation ($\rho = -0.37$).
- **Evidence:** Valuations drop sharply during the initial 0 to 60,000 km interval. Beyond 120,000 km, the price degradation curve flattens significantly, indicating that once a vehicle enters high-mileage territory, condition, mechanical health, and vehicle age outweigh marginal odometer gains.
- **Supporting Artefacts:** Figure `reports/figures/eda_19_scatter_price_vs_km.png`, Table `reports/tables/eda_spearman_correlation.csv`.

---

### 9. Collinearity Between Engine Displacement and Max Power
- **Finding:** Engine displacement and maximum power exhibit substantial collinearity ($r = 0.78, \rho = 0.79$).
- **Evidence:** Variance Inflation Factor (VIF) analysis shows `engine` with a VIF of **6.41** ($R^2 = 0.8439$) and `max_power` with a VIF of **4.56** ($R^2 = 0.7806$). Other numerical features (`vehicle_age`: 1.43, `km_driven`: 1.43, `mileage`: 1.96, `seats`: 2.27) show negligible inflation ($VIF < 2.5$).
- **Supporting Artefacts:** Table `reports/tables/eda_vif_summary.csv`.

---

### 10. Multi-Seater Utility Vehicle Premium
- **Finding:** 7-seater vehicles command a median selling price of **₹8.80 lakh**, significantly outpacing standard 5-seaters (median **₹5.25 lakh**) and 4-seaters (median **₹3.90 lakh**).
- **Evidence:** 7-seaters in India encompass popular premium family MPVs and ladder-frame SUVs (e.g. Toyota Innova, Mahindra XUV500, Ford Endeavour) equipped with larger diesel powertrains.
- **Supporting Artefacts:** Figures `reports/figures/eda_12_cat_seats.png`, `reports/figures/eda_17_price_by_seats.png`, Table `reports/tables/eda_seats_medians.csv`.

---

## Part 2: Implications for Downstream Machine Learning Modeling

1. **Target Variable Transformation:**
   - The regression target must be modelled in log-space ($\log(1 + y)$ or `TransformedTargetRegressor`). Predicting raw prices directly would cause linear and distance models to optimize almost exclusively on high-value outliers, compromising accuracy on the 80% mass market.
2. **Feature Engineering Opportunities (Phase 3):**
   - *Power-to-Displacement Ratio:* $\text{power\_per\_litre} = \frac{\text{max\_power}}{\text{engine} / 1000}$, capturing engine tuning efficiency and turbocharging.
   - *Annual Odometer Usage Rate:* $\text{km\_per\_year} = \frac{\text{km\_driven}}{\text{vehicle\_age}}$, distinguishing heavy commercial fleet use from light private driving.
   - *Brand Equity Tiers:* Grouping 32 automotive marques into structured tiers (Budget, Mid, Premium, Luxury, Ultra-Luxury) to resolve categorical sparsity among low-frequency brands.
   - *Power $\times$ Transmission Interaction:* Explicitly modeling the interaction term between horsepower and automatic gearboxes.
3. **Collinearity Mitigation for Linear Baseline:**
   - Unregularized OLS will suffer variance instability due to the 6.41 VIF between `engine` and `max_power`. L2-regularized Ridge regression and tree-based gradient boosters (XGBoost, LightGBM) must be prioritized.
4. **Categorical Encoding Strategy:**
   - High-cardinality features (`model` with 119 levels and `brand` with 32 levels) require One-Hot Encoding for frequent levels combined with Target / Frequency encoding or tree-native categorical handling to prevent dimensionality explosion.

