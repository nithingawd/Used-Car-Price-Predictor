# Worked examples: ten real test cars

Ten cars from the locked test set, chosen by fixed rules (D-031), not by hand. Eight are the *typical* car of a category: the one whose error is closest to that category's median error. The last two are the single best and worst predictions in the test set. Influence percentages are each feature's TreeSHAP effect on this car's predicted price relative to an average car. All numbers come from `worked_examples.csv` in this folder.

### 1. Cheap petrol hatchback: Datsun Redigo

- **Inputs:** 5 years, 79,000 km, Petrol, Manual, 799 cc, 53.64 bhp, 22.7 kmpl, 5 seats, Dealer seller
- **Actual listing price:** ₹ 2,35,000 (₹ 2.35 lakh)
- **Predicted:** ₹ 2,63,272 (₹ 2.63 lakh), 80% range ₹ 2,17,650 to ₹ 3,18,840
- **Why:** Close: the estimate is +12.03% from the listing and inside the 80% range; the biggest influences were model (-38%) and max power (bhp) (-15%), and it is a well-represented model (55 training cars), so the gap is ordinary variation between similar listings.

### 2. Family diesel SUV or MPV: Maruti Ertiga

- **Inputs:** 5 years, 41,974 km, Diesel, Manual, 1248 cc, 88.5 bhp, 24.52 kmpl, 7 seats, Dealer seller
- **Actual listing price:** ₹ 8,11,000 (₹ 8.11 lakh)
- **Predicted:** ₹ 7,18,623 (₹ 7.19 lakh), 80% range ₹ 5,94,094 to ₹ 8,70,300
- **Why:** Close: the estimate is -11.39% from the listing and inside the 80% range; the biggest influences were model (+11%) and vehicle age (years) (+8%), and it is a well-represented model (267 training cars), so the gap is ordinary variation between similar listings.

### 3. Old car (12+ years): Hyundai Santro

- **Inputs:** 13 years, 15,000 km, Petrol, Manual, 1086 cc, 62.1 bhp, 17.92 kmpl, 5 seats, Individual seller
- **Actual listing price:** ₹ 1,20,000 (₹ 1.20 lakh)
- **Predicted:** ₹ 1,42,660 (₹ 1.43 lakh), 80% range ₹ 1,17,938 to ₹ 1,72,770
- **Why:** Close: the estimate is +18.88% from the listing and inside the 80% range; the biggest influences were vehicle age (years) (-40%) and model (-38%), and old cars differ in condition, which the data does not record.

### 4. Luxury car: Jaguar Xe

- **Inputs:** 4 years, 91,795 km, Diesel, Automatic, 1999 cc, 177 bhp, 13.5 kmpl, 5 seats, Dealer seller
- **Actual listing price:** ₹ 29,50,000 (₹ 29.50 lakh)
- **Predicted:** ₹ 33,04,602 (₹ 33.05 lakh), 80% range ₹ 27,31,951 to ₹ 40,02,088
- **Why:** Close: the estimate is +12.02% from the listing and inside the 80% range; the biggest influences were model (+142%) and max power (bhp) (+37%), and only 2 cars of this model were in the training data.

### 5. CNG car: Maruti Eeco

- **Inputs:** 8 years, 72,000 km, Cng, Manual, 1196 cc, 61.7 bhp, 20.88 kmpl, 5 seats, Dealer seller
- **Actual listing price:** ₹ 3,25,000 (₹ 3.25 lakh)
- **Predicted:** ₹ 2,98,398 (₹ 2.98 lakh), 80% range ₹ 2,46,689 to ₹ 3,61,380
- **Why:** Accurate: the estimate is -8.19% from the listing and inside the 80% range; the biggest influences were model (-32%) and max power (bhp) (-12%), and this is a well-represented model with 98 training cars.

### 6. Automatic petrol city car: Honda City

- **Inputs:** 3 years, 40,000 km, Petrol, Automatic, 1498 cc, 119.35 bhp, 18.4 kmpl, 5 seats, Dealer seller
- **Actual listing price:** ₹ 11,50,000 (₹ 11.50 lakh)
- **Predicted:** ₹ 10,72,861 (₹ 10.73 lakh), 80% range ₹ 8,86,946 to ₹ 12,99,304
- **Why:** Accurate: the estimate is -6.71% from the listing and inside the 80% range; the biggest influences were vehicle age (years) (+26%) and max power (bhp) (+13%), and this is a well-represented model with 606 training cars.

### 7. High-kilometre car: Hyundai I20

- **Inputs:** 6 years, 150,000 km, Diesel, Manual, 1396 cc, 88.73 bhp, 21.19 kmpl, 5 seats, Individual seller
- **Actual listing price:** ₹ 6,20,000 (₹ 6.20 lakh)
- **Predicted:** ₹ 5,20,167 (₹ 5.20 lakh), 80% range ₹ 4,30,028 to ₹ 6,29,956
- **Why:** Close: the estimate is -16.10% from the listing and inside the 80% range; the biggest influences were kilometres driven (-9%) and model (-6%), and it is a well-represented model (702 training cars), so the gap is ordinary variation between similar listings.

### 8. Nearly new car (0-1 year): Renault Duster

- **Inputs:** 1 year, 150,000 km, Diesel, Automatic, 1461 cc, 108.5 bhp, 19.87 kmpl, 5 seats, Individual seller
- **Actual listing price:** ₹ 6,00,000 (₹ 6.00 lakh)
- **Predicted:** ₹ 6,58,459 (₹ 6.58 lakh), 80% range ₹ 5,44,355 to ₹ 7,97,436
- **Why:** Accurate: the estimate is +9.74% from the listing and inside the 80% range; the biggest influences were kilometres driven (-15%) and transmission (+13%), and this is a well-represented model with 175 training cars.

### 9. Best prediction in the test set: Maruti Alto

- **Inputs:** 10 years, 72,000 km, Petrol, Manual, 796 cc, 46.3 bhp, 19.7 kmpl, 5 seats, Dealer seller
- **Actual listing price:** ₹ 1,85,000 (₹ 1.85 lakh)
- **Predicted:** ₹ 1,84,967 (₹ 1.85 lakh), 80% range ₹ 1,52,914 to ₹ 2,24,007
- **Why:** Accurate: the estimate is -0.02% from the listing and inside the 80% range; the biggest influences were model (-41%) and vehicle age (years) (-23%), and this is a well-represented model with 623 training cars.

### 10. Worst prediction in the test set: Volkswagen Polo

- **Inputs:** 3 years, 18,523 km, Petrol, Automatic, 1798 cc, 189 bhp, 17.21 kmpl, 4 seats, Dealer seller
- **Actual listing price:** ₹ 8,75,000 (₹ 8.75 lakh)
- **Predicted:** ₹ 21,47,143 (₹ 21.47 lakh), 80% range ₹ 17,75,067 to ₹ 26,00,329
- **Why:** Miss: the estimate is +145.39% from the listing and outside the 80% range; the biggest influences were max power (bhp) (+89%) and vehicle age (years) (+37%), and the listing price is unusual for these specifications.
