# Five-minute demo script

A step-by-step walk through the GUI with the exact inputs to type and what to say. The two cars used are real cars from the locked test set (`reports/predictions/worked_examples.csv`), so the prices you will see are the ones quoted here.

## Before the demo (2 minutes, not part of the five)

1. Open a terminal in the project folder and start the app:

   ```bash
   python tasks.py app          # or: streamlit run app/Home.py   (or: make app)
   ```

   On a fresh copy of the project, run `python tasks.py all` once first: it rebuilds the cleaned data, the split and the model (see `reports/reproducibility_check.md` for how long it takes).
2. The browser opens at `http://localhost:8501`. Wait until the **Price estimator** page shows a price on the right.
3. Set the browser zoom so the form and the result fit side by side (90% works on most laptops).
4. Keep `reports/viva_notes.md` open in another window for questions.

---

## 0:00 to 0:30: What this is

**Do:** Stay on the Price estimator page.

**Say:** "This app estimates the listing price of a used car in India from eleven details anyone can read off the car. It was trained on about twelve thousand CarDekho listings and tested once on cars it never saw. Everything on screen comes from saved results, nothing is typed in by hand. There are four pages along the top: the estimator, a market explorer, the model's test performance, and an about page."

---

## 0:30 to 1:45: A typical car, priced well

**Do:** In the left-hand form, enter exactly:

| Field | Value |
|---|---|
| Brand | Honda |
| Model | City |
| Vehicle age (years) | 3 |
| Seller type | Dealer |
| Kilometres driven | 40000 |
| Fuel type | Petrol |
| Transmission | Automatic |
| Engine size (cc) | 1498 |
| Max power (bhp) | 119.35 |
| Mileage (kmpl) | 18.4 |
| Seats | 5 |

Choosing the model fills in typical engine, power and mileage; overwrite them with the values above. Then press **Estimate price**.

**Expect:** about **₹ 10,72,861** (₹ 10.73 lakh), likely between ₹ 8,86,946 and ₹ 12,99,304.

**Say:**

- "The big number is the estimate; the bar shows an 80% range. On the test set that range contained the true price for 80% of cars."
- "This exact car is in the test set: it was listed at ₹ 11,50,000, so the model is about 7% low and well inside its range."
- Point at **Why this price**: "These are the five inputs that moved this car's price most compared with an average car, computed for this car alone. Being young and fairly powerful pushes the price up."
- Point at **Compare**: "This compares it with the median Honda of the same age in the data."
- In **What if**, click **Age**, then **Kilometres**: "Holding everything else fixed, this is how the estimate changes with age, or with kilometres. Age matters far more than kilometres, which matches the market analysis."

---

## 1:45 to 2:45: A car the model gets badly wrong, and why

**Do:** Change the form to:

| Field | Value |
|---|---|
| Brand | Volkswagen |
| Model | Polo |
| Vehicle age (years) | 3 |
| Seller type | Dealer |
| Kilometres driven | 18523 |
| Fuel type | Petrol |
| Transmission | Automatic |
| Engine size (cc) | 1798 |
| Max power (bhp) | 189 |
| Mileage (kmpl) | 17.21 |
| Seats | 4 |

Press **Estimate price**.

**Expect:** about **₹ 21,47,143** (₹ 21.47 lakh), with a **Lower confidence** note saying that luxury and high-priced cars are priced less accurately.

**Say:**

- "This is the worst prediction in the whole test set. The real listing was ₹ 8,75,000, so the model is more than double."
- Point at **Why this price**: "Look at what drove it: power. 189 bhp is sports-car territory for a Polo, and the model trusts its inputs, so it prices it like a performance car."
- "Either the listing has a wrong specification, or the car has a problem the data do not record, such as condition or accident history. The model cannot know."
- "Notice the app does not hide this. Because the estimate is over ₹ 20 lakh, it warns, in words, that this kind of estimate is less reliable. On the test set, cars above ₹ 20 lakh were priced within 10% only 41% of the time."

---

## 2:45 to 3:30: Market explorer

**Do:** Click **Market explorer** in the top bar. In the brand filter choose **Maruti** and **Hyundai**.

**Say:**

- "These are all the cleaned listings, filtered live. The strip at the top gives the number of listings, the median price and median age for the filter."
- Read one chart title aloud (each title states its finding), for example the price-by-age chart: "Prices fall by roughly this much per year of age."
- Clear the brand filter and move the price slider down to the budget range: "The same charts update for any slice of the market."

---

## 3:30 to 4:30: How good is the model?

**Do:** Click **Model performance**.

**Say:**

- Point at the four cards: "On 3,049 cars the model never saw, the average miss is ₹ 94,648, and 51.0% of cars are within 10% of the listing price."
- Point at the predicted-against-actual chart: "Each dot is a test car; hover to see which one. Dots on the dashed line are priced exactly."
- In **Segment by**, click **Vehicle age**: "Here is the honest part. The highlighted bar is the weakest group: cars 12 years and older. Condition matters most for old cars, and the data do not record it."
- Point at the comparison table: "Ten algorithms were compared on training data only. The tuned XGBoost was chosen by a rule fixed in advance; the three boosting models were effectively tied, and all beat the linear models."

---

## 4:30 to 5:00: Close

**Do:** Click **About and data** and scroll to the limitations box.

**Say:** "The about page lists the cleaning steps, how the model works and its limitations, and the model card records exactly which commit and settings produced it. In short: a useful first opinion for mainstream cars, with honest warnings where it is weak. Thank you."

---

## If something goes wrong

| Problem | What to do |
|---|---|
| A "Missing file." note on the page | Stop the app, run the command the note names (usually `python tasks.py all`), start it again |
| A list of problems appears instead of a price | A value is outside what the model was trained on; each line says what is allowed. Correct it and press again |
| The page is slow on the first click | The first estimate loads the model; later ones are fast |
| Port already in use | `streamlit run app/Home.py --server.port 8502` |
| No internet (fonts) | The app still works; it falls back to system fonts |
