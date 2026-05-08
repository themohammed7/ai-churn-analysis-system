import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import random

# ==========================================
# 1. PAGE CONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

EXPECTED_COLS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 
    'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 
    'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def apply_theme(theme_name):
    themes = {
        "Light Default": {"bg": "#FFFFFF", "text": "#31333F"},
        "Dark Mode": {"bg": "#0E1117", "text": "#FAFAFA"},
        "Ocean Blue": {"bg": "#E0F7FA", "text": "#006064"},
        "Forest Green": {"bg": "#E8F5E9", "text": "#1B5E20"}
    }
    st.markdown(f"<style>.stApp {{ background-color: {themes[theme_name]['bg']}; }} .stMarkdown, h1, h2, h3, p, label {{ color: {themes[theme_name]['text']} !important; }} </style>", unsafe_allow_html=True)

def simulate_platform_load():
    platforms = ["AWS S3 Bucket", "Google Cloud Storage", "Azure Blob", "Enterprise Database API"]
    chosen = random.choice(platforms)
    with st.spinner(f"Establishing secure connection to {chosen}..."):
        time.sleep(random.uniform(0.5, 1.0))
    my_bar = st.progress(0, text=f"Downloading from {chosen}...")
    for p in range(100):
        time.sleep(random.uniform(0.005, 0.02))
        my_bar.progress(p + 1, text=f"Downloading from {chosen}... {p + 1}%")
    time.sleep(0.3)
    my_bar.empty()
    st.toast(f'Data loaded from {chosen}!', icon='✅')

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#ff4b4b" if probability > 0.5 else "#00cc96"},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': [{'range': [0, 30], 'color': "rgba(0, 204, 150, 0.2)"},
                      {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.2)"},
                      {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.2)"}],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=80)
st.sidebar.title("Settings")
apply_theme(st.sidebar.selectbox("🎨 UI Theme", ["Light Default", "Dark Mode", "Ocean Blue", "Forest Green"]))
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["📊 Analytics Dashboard", "👤 Single Customer Profiling", "📂 Batch Risk Processing"])

# ==========================================
# 4. PAGE 1: DASHBOARD
# ==========================================
if page == "📊 Analytics Dashboard":
    st.title("📊 Telco Analytics Dashboard")
    uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
    
    if uploaded_file:
        simulate_platform_load()
        df = pd.read_csv(uploaded_file)
        
        # Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(df):,}")
        c2.metric("Monthly Revenue", f"${df['MonthlyCharges'].sum():,.2f}")
        c3.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
        c4.metric("Avg Tenure", f"{df['tenure'].mean():.1f} Months")
        
        st.markdown("---")
        # Visualizations
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(px.histogram(df, x="Contract", color="Churn", barmode="group", color_discrete_sequence=["#1f77b4", "#ff4b4b"], title="Customer Churn by Contract Type"), use_container_width=True)
        with col_chart2:
            st.plotly_chart(px.box(df, x="Churn", y="MonthlyCharges", color="Churn", color_discrete_sequence=["#1f77b4", "#ff4b4b"], title="Impact of Monthly Charges on Churn"), use_container_width=True)

# ==========================================
# 5. PAGE 2: SINGLE PREDICTION
# ==========================================
elif page == "👤 Single Customer Profiling":
    st.title("👤 Individual Risk Profiling")
    st.markdown("Enter customer details below. Hover over the **(?)** icon for column descriptions.")
    
    if not model:
        st.error("Model missing. Ensure .pkl files exist.")
    else:
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Demographics")
                gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's identified gender.")
                senior = st.selectbox("Senior Citizen", ["No", "Yes"], help="Is customer 65 or older?")
                partner = st.selectbox("Has Partner", ["No", "Yes"], help="Customer has a spouse/partner?")
                dependents = st.selectbox("Has Dependents", ["No", "Yes"], help="Customer lives with children/dependents?")
                
            with col2:
                st.subheader("Account Details")
                tenure = st.number_input("Tenure (Months)", 0, 100, 12, help="Total months with the company.")
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], help="Current length of subscription term.")
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, help="Amount billed to customer monthly.")
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0, help="Total lifetime billed amount.")
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], help="Preferred method for monthly billing.")
                
            with col3:
                st.subheader("Services Subscribed")
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], help="Type of internet connection.")
                phone = st.selectbox("Phone Service", ["Yes", "No"], help="Subscribed to home phone service?")
                security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], help="Subscribed to network security?")
                backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], help="Subscribed to cloud backup?")
                support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], help="Subscribed to premium support?")
            
            submit = st.form_submit_button("Generate Risk Analysis", use_container_width=True)
            
        if submit:
            # Prepare Data Array
            input_dict = {col: 0 for col in EXPECTED_COLS}
            input_dict['SeniorCitizen'], input_dict['tenure'] = (1 if senior == 'Yes' else 0), tenure
            input_dict['MonthlyCharges'], input_dict['TotalCharges'] = monthly_charges, total_charges
            
            if gender == 'Male': input_dict['gender_Male'] = 1
            if partner == 'Yes': input_dict['Partner_Yes'] = 1
            if dependents == 'Yes': input_dict['Dependents_Yes'] = 1
            if phone == 'Yes': input_dict['PhoneService_Yes'] = 1
            if internet == 'Fiber optic': input_dict['InternetService_Fiber optic'] = 1
            elif internet == 'No': input_dict['InternetService_No'] = 1
            if contract == 'One year': input_dict['Contract_One year'] = 1
            elif contract == 'Two year': input_dict['Contract_Two year'] = 1
            if payment != 'Bank transfer (automatic)': input_dict[f'PaymentMethod_{payment}'] = 1
            
            for s_name, s_val in zip(['OnlineSecurity', 'OnlineBackup', 'TechSupport'], [security, backup, support]):
                if s_val == 'Yes': input_dict[f'{s_name}_Yes'] = 1
                elif s_val == 'No internet service': input_dict[f'{s_name}_No internet service'] = 1

            input_df = pd.DataFrame([input_dict])
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
            
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            r1, r2 = st.columns([1.5, 1])
            with r1:
                st.subheader("AI Prediction Insight:")
                if prediction == 1: 
                    st.error("🚨 **CRITICAL FLIGHT RISK DETECTED**")
                    st.write("This profile exhibits behavior strongly correlated with churning. We recommend immediate intervention, such as a promotional discount or targeted tech support check-in.")
                else: 
                    st.success("✅ **STABLE CUSTOMER**")
                    st.write("This customer shows high loyalty indicators. Standard marketing flows are sufficient.")
            with r2:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

# ==========================================
# 6. PAGE 3: BATCH PREDICTION
# ==========================================
elif page == "📂 Batch Risk Processing":
    st.title("📂 Batch Risk Processing")
    st.markdown("Process entire databases to visualize your highest risk segments.")
    
    batch_file = st.file_uploader("Upload Customer Roster (CSV)", type=["csv"])
    
    if batch_file and model:
        simulate_platform_load()
        df_clean = pd.read_csv(batch_file)
        
        # Save columns for visualizations before encoding
        vis_df = df_clean.copy() 
        c_ids = df_clean.pop('customerID') if 'customerID' in df_clean.columns else pd.Series(range(1, len(df_clean)+1))
        
        if 'TotalCharges' in df_clean.columns: df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
        if 'Churn' in df_clean.columns: df_clean.drop('Churn', axis=1, inplace=True)

        try:
            X_batch = pd.get_dummies(df_clean, drop_first=True)
            for col in EXPECTED_COLS:
                if col not in X_batch.columns: X_batch[col] = 0
            X_batch = X_batch[EXPECTED_COLS]
            
            X_batch[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_batch[['tenure', 'MonthlyCharges', 'TotalCharges']])
            predictions, probabilities = model.predict(X_batch), model.predict_proba(X_batch)[:, 1]
            
            results_df = pd.DataFrame({'CustomerID': c_ids, 'Risk (%)': (probabilities * 100).round(2), 'Status': ['Will Churn' if p == 1 else 'Will Stay' for p in predictions]})
            
            # Interactive Visualizations for Batch
            st.markdown("### Risk Distribution Analysis")
            col_viz1, col_viz2 = st.columns([1, 1.5])
            
            with col_viz1:
                st.plotly_chart(px.pie(results_df, names='Status', color='Status', color_discrete_map={'Will Churn':'#ff4b4b', 'Will Stay':'#00cc96'}, hole=0.4, title="Overall Fleet Risk"), use_container_width=True)
            
            with col_viz2:
                # Merge prediction back to original data for interactive plotting
                vis_df['Predicted Status'] = results_df['Status']
                fig = px.scatter(vis_df, x='tenure', y='MonthlyCharges', color='Predicted Status', color_discrete_map={'Will Churn':'#ff4b4b', 'Will Stay':'#00cc96'}, opacity=0.7, title="Risk Matrix: Tenure vs Monthly Charges")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Detailed Customer Roster")
            st.dataframe(results_df.style.applymap(lambda x: 'background-color: rgba(255, 75, 75, 0.2)' if x == 'Will Churn' else 'background-color: rgba(0, 204, 150, 0.2)', subset=['Status']), height=300)
            st.download_button("📥 Export High-Risk Target List (CSV)", data=results_df[results_df['Status'] == 'Will Churn'].to_csv(index=False).encode('utf-8'), file_name='intervention_targets.csv', mime='text/csv')
            
        except Exception as e:
            st.error(f"Format mismatch. Ensure columns match the training dataset. Detail: {e}")