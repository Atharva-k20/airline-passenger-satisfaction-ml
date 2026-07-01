# ✈️ Airline Passenger Satisfaction Prediction

Predicting passenger satisfaction from flight experience data to help airlines proactively reduce churn — built end-to-end from raw, messy data to a tuned, cross-validated Random Forest model.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Problem Statement

Airlines lose loyal customers when dissatisfaction goes undetected until it's too late. This project builds a binary classification model to predict whether a passenger is **Satisfied** or **Neutral/Dissatisfied**, enabling airlines to identify at-risk passengers and intervene before they churn.

## 📊 Results

| Model | Accuracy | ROC-AUC | F1 Score | Train Time |
|---|---|---|---|---|
| **Random Forest (Tuned)** ⭐ | **96.47%** | **0.9946** | **0.9586** | 33.4s |
| Gradient Boosting | 94.30% | 0.9879 | 0.9335 | 38.3s |
| Decision Tree | 94.85% | 0.9479 | 0.9409 | 2.0s |
| Logistic Regression | 87.11% | 0.9292 | 0.8532 | 0.4s |

**Overfitting check:** 5-fold stratified CV mean = 0.9939 (std: 0.0002) vs Test ROC-AUC = 0.9946 → gap of **0.0007**, confirming the model generalizes well rather than memorizing training data.

**Recall on Dissatisfied class: 0.98** — the model correctly catches 98% of all genuinely unhappy passengers, prioritizing the business cost of a missed at-risk customer over a false alarm.

## 🔑 Key Insight

> Online Boarding — not seat comfort, not food, not flight delay — is the single strongest predictor of satisfaction (correlation: 0.50, feature importance: 0.169). The first digital touchpoint sets the tone for the entire passenger journey.

**Top 5 satisfaction drivers (Random Forest feature importance):**
1. Online Boarding (0.169)
2. In-flight Wifi Service (0.153)
3. Class (0.104)
4. Type of Travel (0.096)
5. In-flight Entertainment (0.057)

## 📁 Dataset

- **129,880 passenger records** × 21 features (after cleaning)
- Target distribution: 56.6% Neutral/Dissatisfied, 43.4% Satisfied (mild imbalance — handled via `class_weight='balanced'`, no SMOTE needed)
- Source: `DS-DATA.csv`

**Data quality issues found and fixed:**
- `Arrival Delay`: 393 missing values → imputed with median (robust to outliers)
- `Flight Distance`: stored as mixed-type (object) due to 3 corrupted entries → converted via `pd.to_numeric(errors='coerce')`, then median-imputed
- `Departure Delay` dropped due to high multicollinearity with `Arrival Delay` (correlation ≈ 0.99) — kept Arrival Delay since it's closer to the passenger's actual experience

## 🛠 Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Jupyter`

## 🔬 Methodology

1. **Data Cleaning** — missing value imputation, dtype correction, multicollinearity removal
2. **EDA** — target distribution, feature distributions, outlier detection (IQR), correlation analysis
3. **Feature Engineering** — Label Encoding (binary features), Ordinal Encoding (Class: Economy < Economy Plus < Business)
4. **Preprocessing** — stratified 80-20 train-test split, StandardScaler (fit on train only — no data leakage)
5. **Modeling** — benchmarked 4 algorithms (Logistic Regression → Decision Tree → Random Forest → Gradient Boosting), increasing in complexity
6. **Tuning** — RandomizedSearchCV (20 iterations, 3-fold CV) on a 20% sample for efficiency, validated on full test set
7. **Validation** — 5-fold stratified cross-validation to confirm no overfitting
8. **Business Translation** — converted feature importance into actionable airline recommendations

## 💡 Business Recommendations

1. **Invest in digital boarding** — highest-impact, highest-ROI improvement area
2. **Improve in-flight Wifi reliability** — second highest driver of satisfaction
3. **Upgrade Economy / Economy Plus experience** — these segments are disproportionately dissatisfied
4. **Build a first-time flyer loyalty program** — first-time passengers show significantly lower satisfaction and higher churn risk than returning customers

## 📂 Project Structure

```
airline-passenger-satisfaction-ml/
├── Airline_Satisfaction_Prediction.ipynb   # full analysis notebook
├── Airline_Satisfaction_Presentation.pptx  # 5-slide project summary
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
git clone https://github.com/Atharva-k20/airline-passenger-satisfaction-ml.git
cd airline-passenger-satisfaction-ml
pip install -r requirements.txt
jupyter notebook Airline_Satisfaction_Prediction.ipynb
```

## 👤 Author

**Atharva Khot** — Data Scientist | M.Sc. AI & ML
[LinkedIn](https://www.linkedin.com/in/atharva-khot-37b53228a/) · [GitHub](https://github.com/Atharva-k20) · atharvakhot58@gmail.com
