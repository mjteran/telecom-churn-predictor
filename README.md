# Telecom Customer Churn Predictor

Predict customer churn for a telecom company with an end-to-end ML pipeline and an interactive Streamlit dashboard.

---

## 🔍 Project Overview

Customer churn is a major challenge for telecom companies.  
This project:

- Cleans and engineers an extended IBM Telco Churn dataset.  
- Benchmarks several models and selects **Logistic Regression (L1)** for the best recall + interpretability.  
- Serves real-time predictions and insights through a public **Streamlit** app.

---

## 🤖 Model Exploration & Selection

| Model Tested | F1 / Recall (val) | Notes |
|--------------|-------------------|-------|
| Logistic Regression (L1) | **0.93 / 0.91** | *Chosen – best recall & interpretability* |
| K-Nearest Neighbors | 0.86 / 0.79 | Baseline non-parametric |
| Decision Tree | 0.82 / 0.81 | Simple tree |
| SVM | 0.85 / 0.84 | High-dim space |
| Random Forest | 0.91 / 0.87 | Ensemble |
| XGBoost | 0.93 / 0.89 | Boosted trees |

**Why Linear?**  
Logistic Regression matches the top F1, gives the highest recall, avoids overfitting, and remains fully explainable to stakeholders.

*Scaler:* `RobustScaler` | *Extra features:* GeoCluster (K-Means), frequency-encoded city, one-hot/ordinal/boolean encodings.

---

## 📒 Notebook & Slides

- **EDA + Modeling notebook:** [`notebooks/churn_modeling_exploration.ipynb`](notebooks/churn_modeling_exploration.ipynb)  
- **Slide deck:** <https://docs.google.com/presentation/d/15nHl9ydwYCIzIfBo-hEmODlNJwKqVoqxpElqfxdKs_o/edit?usp=sharing>

---

## 📁 Repository Structure

```text
telecom-churn-predictor/
├── app.py                       # Streamlit dashboard
├── Churn_modeling.py            # Data cleaning, feature engineering, model training
├── requirements.txt             # Python dependencies
├── model/                       # Serialized model + helper objects
│   ├── churn_prediction_lr.pkl
│   ├── rob_scaler.pkl
│   ├── city_freq_dict.pkl
│   ├── city_to_cluster.pkl
│   └── means_churn_inputs.pkl
├── data/
│   └── df_EDA.csv               # Aggregated data for dashboard visuals
├── notebooks/
│   └── churn_modeling_exploration.ipynb
└── README.md
```

---

## 🌐 Live Demo

▶ **Try it now:** <https://mjteran-telecom-churn-predictor.streamlit.app>

---

## 🚀 Run Locally

```bash
# 1 – clone and enter repo
git clone https://github.com/<your-username>/telecom-churn-predictor.git
cd telecom-churn-predictor

# 2 – install dependencies (use a venv if you like)
pip install -r requirements.txt

# 3 – launch Streamlit
streamlit run app.py
```
(Opens <http://localhost:8501> in your browser).

---

## 📊 Dashboard Highlights

| Section | What you can do |
|---------|-----------------|
| **Predict Churn** | Fill a form → get churn probability & label. |
| **Insights** | Explore churn by contract, tenure, city, charges, satisfaction, CLTV, etc. |

---

## 🛠 Requirements

```text
pandas
scikit-learn
joblib
plotly
streamlit
```

(Automatically installed via `requirements.txt`.)

---

## 👩‍💻 Author

**Maria Jose Teran** — Data Analyst & ML Enthusiast  
GitHub • [@mjteran](https://github.com/mjteran) | LinkedIn • [majoteran92](https://linkedin.com/in/majoteran92)
