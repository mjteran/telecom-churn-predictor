# Telecom Customer Churn Predictor

This project predicts customer churn for a telecom company using machine learning and presents the results through an interactive dashboard built with Streamlit.

---

## 🔍 Project Description

Customer churn is a major challenge for telecom companies. This app helps identify customers who are likely to leave the service. It uses:

* A machine learning model (Logistic Regression)
* A Streamlit app for predictions and insights
* Visualizations based on customer data (demographics, services, billing, and satisfaction)

---

## 📁 Project Structure

```
telecom-churn-predictor/
├── churn_app.py             # Streamlit dashboard
├── Churn_modeling.py        # Model training and feature engineering
├── requirements.txt         # Required Python libraries
├── README.md                # Project overview (this file)
├── model/                   # Saved model and encoder objects
│   ├── churn_prediction_lr.pkl
│   ├── city_freq_dict.pkl
│   ├── city_to_cluster.pkl
│   ├── means_churn_inputs.pkl
│   └── rob_scaler.pkl
├── data/
│   └── df_EDA.csv           # Processed data for dashboard visualizations
├── notebooks/
│   └── churn_modeling_exploration.ipynb  # Jupyter notebook with EDA + ML
```

---

## 🚀 Run the App Locally

1. Clone this repository:

```bash
git clone https://github.com/your-username/telecom-churn-predictor.git
cd telecom-churn-predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the dashboard:

```bash
streamlit run churn_app.py
```

---

## 🌐 Live App

You can access the dashboard here:
🔗 [Streamlit App](https://mjteran-telecom-churn-predictor.streamlit.app)

---

## 📊 Dashboard Features

* A **churn prediction form** where you input customer information
* Visual insights on:

  * Churn rate by contract, city, and monthly charges
  * Satisfaction score vs churn
  * CLTV by gender
  * Main churn reasons

---

## 📒 Notebooks & Presentation

Explore the full modeling process in:

* [Modeling Notebook](notebooks/churn_modeling_exploration.ipynb)
* [Project Presentation (Google Slides)](https://docs.google.com/presentation/d/15nHl9ydwYCIzIfBo-hEmODlNJwKqVoqxpElqfxdKs_o/edit?usp=sharing)

It includes:

* Data exploration
* Feature engineering
* Model evaluation
* Metrics and comparisons

---

## 📈 Model Details

* Models tested: Logistic Regression, K-Nearest Neighbors, Decision Tree, SVM, Random Forest, XGBoost
* Final Model: **Logistic Regression** with L1 regularization, chosen for its strong recall, simplicity, and interpretability
* Scaler: RobustScaler
* Features include demographics, services, billing info, and geoclusters
* Validation metrics:

  * Accuracy \~96%
  * F1-score (churn): 0.93
  * Recall: 0.91

---

## 👩‍💻 Author

**Maria Jose Teran**
Data Analyst & ML Learner
GitHub: [@mjteran](https://github.com/mjteran)
LinkedIn: [linkedin.com/in/majoteran92](https://linkedin.com/in/majoteran92)

---

## 📷 Screenshot

*Add a screenshot of your dashboard UI and update the path below:*

![Dashboard Screenshot](dashboard_preview.png)
