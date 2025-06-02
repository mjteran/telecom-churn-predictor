import plotly.express as px
import streamlit as st
import pandas as pd
import joblib

# Author: Maria Jose Teran

# --- Load model and files ---
model_lr = joblib.load("model/churn_prediction_lr.pkl")
city_freq_dict = joblib.load("model/city_freq_dict.pkl")
means_dict = joblib.load("model/means_churn_inputs.pkl")
city_to_cluster = joblib.load("model/city_to_cluster.pkl")
rob_scaler = joblib.load("model/rob_scaler.pkl")

# --- Functions ---
default_freq = min(city_freq_dict.values())
age_bins = [18, 30, 45, 60, 100]
def get_age_group(x):
    return pd.cut([x], bins=age_bins)[0]

def get_mean_from_dict(d, key):
    return d.get(key, 0)

# --- Layout ----
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# Sidebar navigation
with st.sidebar.expander("Telecom Churn Navigation Menu"):
    page = st.radio("Choose a page", ["Predict Churn", "Insights & Visualizations"])

### Visualizations page
if page == "Insights & Visualizations":
    st.title("Customer Insights & Visualizations")
    st.markdown("""
    ### Analysis Overview
    This analysis explores **customer churn** in a telecom company using machine learning. By identifying customers likely to leave, the company can improve retention strategies and reduce losses.

    Dataset used: **IBM Telco Churn Dataset (extended version)**, which includes customer demographics, service usage, support, billing, and satisfaction data.
    This dataset includes over 7,000 customers from California USA, and 50+ features.""")
    st.markdown("---")
    st.markdown("Explore patterns related to churn, customer satisfaction, contracts, etc.")

    # Load EDA data
    df_eda = pd.read_csv("data/df_EDA.csv")

    # 1. Churn distribution
    st.subheader("Churn Percentage")
    churn_counts = df_eda['Churn'].value_counts()
    fig1 = px.pie(values=churn_counts.values, names=['Stayed', 'Churned'], title='Overall Churn Rate', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1)
    st.markdown("""
    #### Insights:
    - Approximately 26.5% of customers have churned, indicating over a quarter of the user base has left the service.
    - The majority (73.5%) of customers have remained, showing room for retention strategies but highlighting potential churn risk.
    """)
    # 2. Churn by Contract Type
    st.subheader("Churn by Contract Type")
    df_eda['Contract'] = pd.Categorical(df_eda['Contract'], categories=["Month-to-Month", "One Year", "Two Year"], ordered=True)
    fig2 = px.histogram(df_eda.sort_values("Contract"), x="Contract", color="Churn", barmode="group", title="Churn vs Contract Type", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2.update_traces(texttemplate='%{y}', textposition='outside')
    st.plotly_chart(fig2)
    st.markdown("""
    #### Insights:
    - Month-to-Month contracts have the highest churn rate, indicating these customers are more likely to leave.  
    - One Year and especially Two Year contracts show much lower churn, highlighting stronger customer retention.
    """)

    # 3. Monthly Charges vs Churn
    st.subheader("Monthly Charges Distribution by Churn")
    fig3 = px.box(df_eda, x="Churn", y="Monthly Charge", color="Churn", title="Monthly Charges and Churn", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig3)
    st.markdown("""
    #### Insights:
    - Customers who churn tend to have higher monthly charges on average.
    - Non-churning customers have a wider range but typically pay less monthly.
    """)

    # 4. Tenure vs Churn
    st.subheader("Tenure and Churn")
    fig4 = px.histogram(df_eda, x="Tenure in Months", color="Churn", nbins=30, barmode="overlay",
                        title="Tenure Distribution by Churn", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig4)
    st.markdown("""
    #### Insights:
    - Customers with shorter tenure (less than 12 months) exhibit significantly higher churn rates.
    - Retention increases with tenure: long-term customers (over 50 months) are much less likely to churn.
    """)

    # 5. Geographic Churn Rate
    st.subheader("Geographic Churn Rate")
    df_map = df_eda.reset_index()
    df_map_grouped = df_map.groupby(['City', 'Latitude', 'Longitude']).agg(
        Customer_Count=('Customer ID', 'count'),
        Churn_Rate=('Churn', 'mean')).reset_index()
    fig5 = px.scatter_geo(df_map_grouped,
                          lat='Latitude',
                          lon='Longitude',
                          size='Customer_Count',
                          color='Churn_Rate',
                          hover_name='City',
                          scope='usa',
                          title='Churn Rate and Customer Count by City',
                          color_continuous_scale='plasma')

    fig5.update_layout(height=700, width=1000)
    fig5.update_geos(landcolor="lightgray", showland=True)
    st.plotly_chart(fig5)
    st.markdown("""
    #### Insights:
    - Churn rates vary across cities, with lighter colors showing where more customers are leaving.
    - Larger customer bases do not always mean higher churn, suggesting that churn is not only driven by the number of users in each city.
    """)

    #6. Sunburst chart of Churn Reasons
    st.subheader("Churn Reasons Breakdown")
    churn_reason = df_eda[df_eda['Churn'] == 1][['Churn Category', 'Churn Reason']].value_counts().reset_index(name='Count')
    fig6 = px.sunburst(
        churn_reason,
        path=['Churn Category', 'Churn Reason'],
        values='Count',
        title='Churn Reasons by Category',
        color_discrete_sequence=px.colors.qualitative.Pastel)
    fig6.update_traces(
        textinfo='label+percent entry',
        insidetextorientation='radial')
    fig6.update_layout(height=800)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
    #### Insights:
    - Most customers churned due to competitors offering better deals or devices, accounting for nearly half of the reasons.  
    - Negative attitudes from support or service staff and overall dissatisfaction were also significant drivers.
    """)

    #7. Lineplot Satisfaction Score vs Churn Score
    st.subheader("Satisfaction Score compared with Churn Score")
    df_avg = df_eda.groupby(['Churn Score', 'Churn']).agg({'Satisfaction Score': 'mean'}).reset_index()
    fig7 = px.line(df_avg, x='Churn Score', y='Satisfaction Score', color='Churn',
                  title='Avg Satisfaction Score by Churn Score', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig7.add_hline(y=df_avg[df_avg['Churn'] == 0]['Satisfaction Score'].mean(), line_dash="dash", line_color="black")
    fig7.add_hline(y=df_avg[df_avg['Churn'] == 1]['Satisfaction Score'].mean(), line_dash="dash", line_color="black")
    fig7.update_layout(height=500, width=900)
    st.plotly_chart(fig7)
    st.markdown("""
    #### Insights:
    - Customers who stayed had an average satisfaction score close to 4, showing steady service approval.  
    - Those who churned showed much lower satisfaction levels, with an average below 2, confirming dissatisfaction as a key churn driver.
    """)

    #8. CLTV Distribution by Churn Status and Gender
    st.subheader("CLTV vs Gender")
    fig8 = px.box(
        df_eda,
        x="Churn",
        y="CLTV",
        color="Gender",
        title="CLTV Distribution by Churn Status and Gender",
        labels={"Churn": "Churned", "CLTV": "Customer Lifetime Value"},
        color_discrete_map={"Male": "dodgerblue", "Female": "deeppink"})
    fig8.update_layout(height=600)
    st.plotly_chart(fig8)
    st.markdown("""
    #### Insights:
    - Both men and women who stayed had higher average CLTV compared to those who churned.  
    - Regardless of gender, churned customers show a drop in lifetime value, suggesting retention directly impacts profitability.
    """)

### Predict Churn page
if page == "Predict Churn":
    st.title("Telecom Churn Predictor")
    st.markdown("This tool allows you to **predict the likelihood that a customer will churn** (i.e., leave the telecom service). \
    Please fill in the customer's information in the form below. The model will analyze service usage, demographics, and payment behavior to make a prediction.")

    st.markdown("### ")
    col1, col2, col3 = st.columns(3)

    # Column 1: Personal Info
    with col1:
        st.markdown("#### 👤 Customer Info")
        age = st.number_input("Age", 18, 100, 30)
        age_group = get_age_group(age)
        gender = st.selectbox("Gender", ["", "Male", "Female"])
        city = st.selectbox("City", [""] + sorted(city_freq_dict.keys()))
        married = st.selectbox("Married", ["", "Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["", "Yes", "No"])
        if dependents == "Yes":
            num_dependents = st.number_input("Number of Dependents", 1, 10, 1)
        elif dependents == "No":
            num_dependents = 0
        else:
            num_dependents = ""

    # Column 2: Services
    with col2:
        st.markdown("#### 🧾 Services Subscribed")
        phone_service = st.selectbox("Phone Service", ["", "Yes", "No"])
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["", "Yes", "No"])
        elif phone_service == "No":
            multiple_lines = "No"
        else:
            multiple_lines = ""

        internet_service = st.selectbox("Internet Service", ["", "Yes", "No"])
        if internet_service == "Yes":
            internet_type = st.selectbox("Internet Type", ["", "Cable", "DSL", "Fiber Optic"])
        elif internet_service == "No":
            internet_type = "None"
        else:
            internet_type = ""

        unlimited_data = st.selectbox("Unlimited Data", ["", "Yes", "No"])

        st.markdown("*Streaming Services*")
        streaming_tv = st.checkbox("TV")
        streaming_movies = st.checkbox("Movies")
        streaming_music = st.checkbox("Music")

        st.markdown("*Tech Support*")
        tech_support = st.checkbox("Premium Tech Support")
        device_protection = st.checkbox("Device Protection Plan")
        online_security = st.checkbox("Online Security protection Plan")
        online_backup = st.checkbox("Cloud Storage Plan")


    # Column 3: Financial Info
    with col3:
        st.markdown("#### 💰 Financial Info")
        contract = st.selectbox("Contract Type", ["", "Month-to-Month", "One Year", "Two Year"])
        tenure = st.slider("Time with services (in months)", 0, 72, 12)
        monthly_charge = st.number_input("Monthly Charge", 0.0, 200.0, 50.0)
        offer = st.selectbox("Offer", ["", "None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
        payment_method = st.selectbox("Payment Method", ["", "Bank Withdrawal", "Credit Card"])
        paperless_billing = st.selectbox("Paperless Billing", ["", "Yes", "No"])

    st.markdown("---")
    st.markdown("Mark your level of satisfaction with the service overall:")
    col_a, _ = st.columns([1, 2])
    with col_a:
        satisfaction = st.slider("Satisfaction Level", 1, 5, 3)
        referred = st.selectbox("Have you ever referred a friend?", ["", "Yes", "No"])
        if referred == "Yes":
            num_referred = st.number_input("Number of people referred", 1, 10, 1)
        elif referred == "No":
            num_referred = 0
        else:
            num_referred = ""

    # --- Validate Inputs ---
    required_fields = [gender, married, dependents, city, contract, phone_service, multiple_lines,
                       internet_type, unlimited_data, streaming_tv, streaming_movies, streaming_music,
                       tech_support, device_protection, offer, payment_method, paperless_billing, referred]

    # --- Prediction ---
    if st.button("🔍 Predict Churn"):
        if "" in required_fields:
            st.warning("Please fill in all required fields before predicting ⚠️.")
        else:
            input_data = {
                'Age': age,
                'Senior Citizen': 1 if age >= 65 else 0,
                'Under 30': 1 if age < 30 else 0,
                'Gender': 1 if gender == "Male" else 0,
                'Married': 1 if married == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0,
                'Number of Dependents': num_dependents,
                'City_FE': city_freq_dict.get(city, default_freq),
                'Contract': {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}.get(contract, 0),
                'Tenure in Months': tenure,
                'Phone Service': 1 if phone_service == "Yes" else 0,
                'Multiple Lines': 1 if multiple_lines == "Yes" else 0,
                'Internet Type_None': 1 if internet_type == "None" else 0,
                'Internet Type_Cable': int(internet_type == "Cable"),
                'Internet Type_DSL': int(internet_type == "DSL"),
                'Internet Type_Fiber Optic': int(internet_type == "Fiber Optic"),
                'Unlimited Data': 1 if unlimited_data == "Yes" else 0,
                'Streaming TV': int(streaming_tv),
                'Streaming Movies': int(streaming_movies),
                'Streaming Music': int(streaming_music),
                'Premium Tech Support': int(tech_support),
                'Device Protection Plan': int(device_protection),
                'Online Security': int(online_security),
                'Online Backup': int(online_backup),
                'Satisfaction Score': satisfaction,
                'Monthly Charge': monthly_charge,
                'Offer_None': int(offer == "None"),
                'Offer_Offer A': int(offer == "Offer A"),
                'Offer_Offer B': int(offer == "Offer B"),
                'Offer_Offer C': int(offer == "Offer C"),
                'Offer_Offer D': int(offer == "Offer D"),
                'Offer_Offer E': int(offer == "Offer E"),
                'Payment Method_Bank Withdrawal': int(payment_method == "Bank Withdrawal"),
                'Payment Method_Credit Card': int(payment_method == "Credit Card"),
                'Paperless Billing': 1 if paperless_billing == "Yes" else 0,
                'Referred a Friend': 1 if referred == "Yes" else 0,
                'Number of Referrals': int(num_referred),
                # Obtain from means
                'Total Refunds': get_mean_from_dict(means_dict["mean_refunds_by_offer"], offer),
                'Avg Monthly GB Download': get_mean_from_dict(means_dict["mean_gb_download"], (unlimited_data, internet_type, age_group)),
                'Total Extra Data Charges': get_mean_from_dict(means_dict["mean_extra_data"], (unlimited_data, internet_type, age_group)),
                'Avg Monthly Long Distance Charges': get_mean_from_dict(means_dict["mean_long_distance"], (phone_service, multiple_lines)),
                'Total Long Distance Charges': get_mean_from_dict(means_dict["mean_total_long_distance"],(phone_service, multiple_lines)),
                'Total Revenue': tenure * monthly_charge - get_mean_from_dict(means_dict["mean_refunds_by_offer"], offer),
                'GeoCluster': city_to_cluster.get(city, 0)  # Use 0 as default if not found
            }

            df = pd.DataFrame([input_data])
            df = df[model_lr.feature_names_in_]

            df_scaled = rob_scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)

            pred = model_lr.predict(df)[0]
            proba = model_lr.predict_proba(df)[0][1]
            label = "⚠️ Likely to Churn" if pred == 1 else "✅ Unlikely to Churn"

            st.subheader("📈 Prediction Result")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Churn Probability:** {proba:.2%}")