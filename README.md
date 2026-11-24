# Workforce Scheduling Optimisation & A&E Breach Prediction

**Linear programming for fair, low-cost staff scheduling, plus machine learning to flag patients at risk of breaching the NHS 4-hour A&E target**

MSc Business Analytics group project, University of Manchester

`Python` · `PuLP` · `scikit-learn` · `pandas` · `seaborn` · `Logistic Regression`

---

## Part 1: Staff scheduling with linear programming
`01_staff_scheduling_lp.ipynb` (runs without any external data)

**Problem:** schedule 6 student operators to provide exactly 14 hours of cover per weekday. The schedule must respect each person's availability and minimum weekly hours (8 h for BSc students, 7 h for MSc students) while keeping labour cost low.

**Models**
| Scenario | Objective | Weekly cost |
|---|---|---|
| Baseline | Minimise labour cost | **£1,755** |
| Fairness I | Minimise workload deviation, with a budget cap of +1.8% | £1,786 |
| Fairness II | Minimise workload deviation, with no budget cap | £1,790 (+2.0%) |
| Skill-based | Minimise cost, with at least 6 h/day each of Programming and Troubleshooting skills | £1,762 (+0.4%) |

**Insight:** the cost-optimal schedule gives the cheapest operators up to 20 hours, while others work only 7. Spending **about £31 more per week (+1.8%)** gives a much fairer workload. Fairness turns out to be a cheap design choice, not an unavoidable trade-off.

## Part 2: A&E 4-hour breach prediction
`02_aed_breach_prediction.ipynb`

**Problem:** identify, at the moment of arrival, which A&E patients are likely to stay longer than 4 hours, so that staff can step in earlier.

**Approach**
- Exploratory analysis of a 400-patient sample: arrival patterns, length of stay, case mix (HRG codes) and breach rates by age, hour and workload.
- **Logistic regression** with standardised numeric features, one-hot encoded categories and `class_weight='balanced'`, because only about 2% of patients breach.
- Evaluated with 5-fold stratified cross-validation and a held-out test set.

**Results**
- ROC-AUC **0.78**; recall on breaches **0.78** on the test set.
- Precision is low. The model is deliberately tuned as an **early-warning tool**: missing a breach costs more than a false alarm.
- **Key risk drivers (odds ratios):** arrival in the early hours (around 03:00, OR ≈ 12.5), crowding at arrival (OR ≈ 2.2), number of investigations (OR ≈ 1.7) and Sunday arrivals (OR ≈ 1.9).
- Breaches are driven mainly by **operational pressure, not patient demographics**.

## Integrated management system
`src/management_system.py` is an interactive command-line tool that brings together:
- the scheduling model (total cost, schedule by day or by operator),
- the A&E charts and the breach prediction model,
- a **data-management module** to look up, filter, add, modify, delete and save patient records, with every action written to an **audit log** (`user_actions.log`).

```bash
python src/management_system.py
```

## 📁 Repository structure
```
├── 01_staff_scheduling_lp.ipynb     # LP: baseline, fairness scenarios, skill constraints
├── 02_aed_breach_prediction.ipynb   # A&E EDA + logistic regression
├── src/
│   └── management_system.py         # integrated interactive system with audit logging
└── requirements.txt
```

## 📂 Data availability
The A&E dataset (`AED4weeks.csv`) was provided for coursework and is **not included**. Part 1 needs no external data.

---
**Author:** Yi-Wen (Sophia) Su · [LinkedIn](https://www.linkedin.com/in/yi-wen-su-523a54433)
