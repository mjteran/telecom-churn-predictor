# Telecom Customer Churn Predictor

Predict customer churn for a telecom company with an end-to-end ML pipeline and an interactive Streamlit dashboard.

---

## 🔍 Project Overview

Customer churn is a major challenge for telecom companies.  This project:
- Cleans and engineers an extended IBM Telco Churn dataset.  
- Benchmarks several models and selects **Logistic Regression (L1)** for the best recall + interpretability.  
- Serves real-time predictions and insights through a public **Streamlit** app.

---

## 📊 Dashboard Highlights

| Section | What you can do |
|---------|-----------------|
| **Churn Predictor** | Fill a form → get churn probability & label. |
| **Visualizations** | Explore churn by contract, tenure, city, charges, satisfaction, CLTV, etc. |

---

## ▶ **Try it now:**

### 🌐 Live Demo
<https://mjteran-telecom-churn-predictor.streamlit.app>

### 🚀 Run Locally

```bash
# 1 – clone and enter repo
git clone https://github.com/<your-username>/telecom-churn-predictor.git
cd telecom-churn-predictor

# 2 – install dependencies (use a venv if you like)
pip install -r requirements.txt

# 3 – launch Streamlit
streamlit run churn_app.py
```
(Opens <http://localhost:8501> in your browser).

### 🛠 Requirements

```text
pandas
scikit-learn
joblib
plotly
streamlit
```

(Automatically installed via `requirements.txt`.)

---

## 📒 Notebook & Slides

- **EDA + Modeling notebook:** [`notebooks/churn_modeling_exploration.ipynb`](notebook/churn_modeling_exploration.ipynb)  
- **Slide deck:** <https://docs.google.com/presentation/d/15nHl9ydwYCIzIfBo-hEmODlNJwKqVoqxpElqfxdKs_o/edit?usp=sharing>

---

## 📁 Repository Structure

```text
telecom-churn-predictor/
├── churn_app.py                 # Streamlit dashboard
├── requirements.txt             # Python dependencies
├── model/                       # Serialized model + helper objects
│   ├── Churn_modeling.py            # Data cleaning, feature engineering, model training
│   ├── churn_prediction_lr.pkl
│   ├── rob_scaler.pkl
│   ├── city_freq_dict.pkl
│   ├── city_to_cluster.pkl
│   └── means_churn_inputs.pkl
├── data/
│   ├── train.csv                # Raw training split
│   ├── test.csv                 # Raw test split
│   ├── validation.csv           # Raw validation 
│   └── df_EDA.csv               # Aggregated data used for dashboard visuals only
├── notebooks/
│   └── churn_modeling_exploration.ipynb
└── README.md
```

---

## 🤖 Model Exploration & Selection

| Model  | Accuracy | F1 (churn) | Recall (churn) | Hyper-parameter Tuning |
|--------|----------|------------|----------------|------------------------|
| **Logistic Regression (L1)** | **0.9645** | **0.93** | **0.91** | Grid Search (C, penalty) |
| K-Nearest Neighbors | 0.9304 | 0.86 | 0.79 | Elbow Method to pick optimal *k* |
| Decision Tree | 0.9489 | 0.90 | 0.86 | Cross-validation on `max_depth` |
| Support Vector Machine | 0.9574 | 0.92 | 0.89 | Grid Search on C / kernel |
| Random Forest | 0.9574 | 0.92 | 0.87 | Grid Search (n_estimators, depth, etc.) |
| XGBoost | 0.9617 | 0.93 | 0.90 | Grid Search (learning rate, depth, estimators) |

**Why Logistic Regression?**  
Logistic Regression matches the top F1, gives the highest recall, and avoids overfitting. It offered the best performance with simpler interpretation and faster computation.

*Scaler:* `RobustScaler` | *Extra features:* GeoCluster (K-Means), frequency-encoded city, one-hot/ordinal/boolean encodings.

---

## 👩‍💻 Author

**Maria Jose Teran** — Data Analyst & ML Learner  
GitHub • [@mjteran](https://github.com/mjteran) | LinkedIn • [majoteran92](https://linkedin.com/in/majoteran92)
