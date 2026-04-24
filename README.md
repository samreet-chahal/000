#   Telco Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer is likely to churn, built with **Streamlit** and **scikit-learn**.

---

##  Demo

Enter a customer's gender, tenure, and monthly charge — the app predicts churn probability instantly.

---

##  Project Structure

```
├── app.py                  # Streamlit web application
├── churn_analysis.ipynb    # EDA, model training, and evaluation
├── model.pkl               # Trained ML model
├── scaler.pkl              # Feature scaler (StandardScaler)
├── telco_churn.csv         # Dataset
├── requirements.txt        # Python dependencies
└── README.md
```

---

##  Model

| Detail | Value |
|---|---|
| Dataset | Telco Customer Churn |
| Features used | `tenure`, `MonthlyCharges`, `gender` |
| Target | `Churn` (Yes / No) |
| Encoding | Male = 1, Female = 0 |

---

##  How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/samreet-chahal/000.git
cd 000
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

##  Requirements

```
streamlit
scikit-learn
joblib
numpy
pandas
```

---

##  Dataset

The dataset is the publicly available [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

##  Contributing

Pull requests are welcome. For major changes, please open an issue first.
