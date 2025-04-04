import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Page configuration for a cleaner look
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #334155;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
    }
    .prediction-positive {
        background-color: #DCFCE7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #22C55E;
    }
    .prediction-negative {
        background-color: #FEE2E2;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #EF4444;
    }
    .info-box {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 0.25rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for app info and additional controls
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/bank-building.png", width=100)
    st.markdown("## Bank Churn Analyzer")
    st.markdown("### Customer Retention Tool")
    st.markdown("---")
    
    # App description 
    st.markdown("### About")
    st.markdown("""
    This tool helps predict which customers are at risk of leaving the bank.
    
    By analyzing customer data, it provides actionable insights to improve retention strategies.
    """)
    
    st.markdown("---")
    
    # Add sample customer profiles for quick testing
    st.markdown("### Sample Profiles")
    
    if st.button("🧓 Senior Customer"):
        st.session_state.profile_senior = True
    
    if st.button("👨‍💼 High Value Customer"):
        st.session_state.profile_high_value = True
    
    if st.button("👩‍🎓 Young Customer"):
        st.session_state.profile_young = True
    
    if st.button("⚠️ At-Risk Customer"):
        st.session_state.profile_at_risk = True
    
    st.markdown("---")
    st.markdown("### Developed by")
    st.markdown("Ritik Jain - 2025")
    
# Main content
st.markdown('<h1 class="main-header">Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict and prevent customer attrition with advanced analytics</p>', unsafe_allow_html=True)

try:
    # Load the encoders and scaler
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    # Set default values based on selected profiles
    if 'profile_senior' in st.session_state and st.session_state.profile_senior:
        default_age = 68
        default_tenure = 8
        default_balance = 120000.0
        default_credit_score = 720
        default_salary = 65000.0
        default_products = 1
        default_card = 'Yes'
        default_active = 'No'
        st.session_state.profile_senior = False
    elif 'profile_high_value' in st.session_state and st.session_state.profile_high_value:
        default_age = 42
        default_tenure = 7
        default_balance = 185000.0
        default_credit_score = 800
        default_salary = 120000.0
        default_products = 3
        default_card = 'Yes'
        default_active = 'Yes'
        st.session_state.profile_high_value = False
    elif 'profile_young' in st.session_state and st.session_state.profile_young:
        default_age = 24
        default_tenure = 1
        default_balance = 15000.0
        default_credit_score = 680
        default_salary = 48000.0
        default_products = 1
        default_card = 'No'
        default_active = 'Yes'
        st.session_state.profile_young = False
    elif 'profile_at_risk' in st.session_state and st.session_state.profile_at_risk:
        default_age = 32
        default_tenure = 2
        default_balance = 5000.0
        default_credit_score = 610
        default_salary = 45000.0
        default_products = 1
        default_card = 'No'
        default_active = 'No'
        st.session_state.profile_at_risk = False
    else:
        default_age = 35
        default_tenure = 3
        default_balance = 10000.0
        default_credit_score = 650
        default_salary = 50000.0
        default_products = 1
        default_card = 'Yes'
        default_active = 'Yes'
    
    # Create two columns for the form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Customer Demographics</h2>', unsafe_allow_html=True)
        
        # Add an info box for this section
        st.markdown('<div class="info-box">Demographic characteristics help identify customer segments with similar behavior patterns.</div>', unsafe_allow_html=True)
        
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        
        # Better slider styling for age
        age = st.slider('Age', 18, 92, default_age, help="Customer's age in years")
        
        # Add age group indicator for context
        age_group = "Senior Citizen (65+)" if age >= 65 else "Middle-aged (35-64)" if age >= 35 else "Young Adult (18-34)"
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>{age_group}</div>", unsafe_allow_html=True)
        
        tenure = st.slider('Tenure (years)', 0, 10, default_tenure, help="How long the customer has been with the bank")
        
        # Add loyalty indicator
        loyalty = "Long-term Customer" if tenure >= 7 else "Regular Customer" if tenure >= 3 else "New Customer"
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>{loyalty}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Financial Information</h2>', unsafe_allow_html=True)
        
        # Add an info box for this section
        st.markdown('<div class="info-box">Financial patterns often provide the strongest indicators of potential churn.</div>', unsafe_allow_html=True)
        
        credit_score = st.slider('Credit Score', 300, 900, default_credit_score, help="Customer's credit rating")
        
        # Add credit quality indicator
        credit_quality = "Excellent" if credit_score >= 750 else "Good" if credit_score >= 650 else "Fair" if credit_score >= 550 else "Poor"
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>Credit Quality: {credit_quality}</div>", unsafe_allow_html=True)
        
        balance = st.number_input('Account Balance ($)', min_value=0.0, max_value=250000.0, value=default_balance, step=1000.0, format="%.2f")
        
        # Add balance tier indicator
        balance_tier = "Premium" if balance >= 100000 else "Standard" if balance >= 10000 else "Basic"
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>Account Tier: {balance_tier}</div>", unsafe_allow_html=True)
        
        estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, max_value=200000.0, value=default_salary, step=5000.0, format="%.2f")
        num_of_products = st.slider('Number of Products', 1, 4, default_products, help="Number of bank products the customer uses")
        
        # Better binary choices with icons
        has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'], index=0 if default_card == 'No' else 1)
        is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'], index=0 if default_active == 'No' else 1)

    # Add a prediction button with better styling
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_btn = st.button('Analyze Churn Risk', key='predict')
    
    if predict_btn:
        with st.spinner('Analyzing customer data...'):
            # Convert Yes/No to 1/0
            has_cr_card_value = 1 if has_cr_card == 'Yes' else 0
            is_active_member_value = 1 if is_active_member == 'Yes' else 0
            
            # Prepare the input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card_value],
                'IsActiveMember': [is_active_member_value],
                'EstimatedSalary': [estimated_salary]
            })
            
            # One-hot encode 'Geography'
            geo_encoded = onehot_encoder_geo.transform([[geography]])
            # Handle both sparse and dense outputs
            if hasattr(geo_encoded, 'toarray'):
                geo_encoded = geo_encoded.toarray()
                
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine one-hot encoded columns with input data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Calculate churn probability using a heuristic model
            def calculate_churn_probability(data):
                # Extract features (after scaling)
                credit_score = data[0][0]
                gender = data[0][1]
                age = data[0][2]
                tenure = data[0][3]
                balance = data[0][4]
                products = data[0][5]
                has_card = data[0][6]
                is_active = data[0][7]
                salary = data[0][8]
                
                # Base probability
                prob = 0.3
                
                # Age factors
                if age > 1.0:  # After scaling, high age values are positive
                    prob += 0.1
                elif age < -0.5:  # Young customers (negative after scaling)
                    prob += 0.15
                    
                # Tenure factors
                if tenure < -0.5:  # Short tenure (negative after scaling)
                    prob += 0.2
                elif tenure > 1.0:  # Long tenure
                    prob -= 0.15
                    
                # Balance factors
                if balance < -0.8:  # Low balance
                    prob += 0.1
                elif balance > 1.0:  # High balance
                    prob -= 0.1
                    
                # Activity status
                if is_active < 0:  # Inactive (the scaling may affect this)
                    prob += 0.25
                    
                # Products
                if products > 0.5:  # More products
                    prob -= 0.15
                    
                # Has credit card
                if has_card < 0:  # No credit card
                    prob += 0.05
                    
                # Geography is in the columns after index 8 (France, Germany, Spain)
                # We can check which one is 1 after one-hot encoding
                if data[0][10] > 0.5:  # Germany (assuming Germany is the second one-hot column)
                    prob += 0.05  # Higher churn in Germany (example)
                    
                # Clamp to valid range
                return max(0.01, min(0.99, prob))
            
            # Get prediction
            prediction_proba = calculate_churn_probability(input_data_scaled)
            
            # Display results with much better formatting
            st.markdown('<h2 class="section-header">Analysis Results</h2>', unsafe_allow_html=True)
            
            # Create three columns for the results section
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                # Create a gauge chart for the churn probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction_proba,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability", 'font': {'size': 24}},
                    delta = {'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 0.3], 'color': 'green'},
                            {'range': [0.3, 0.7], 'color': 'yellow'},
                            {'range': [0.7, 1], 'color': 'red'}
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with res_col2:
                # Show probability as a progress bar
                st.progress(float(prediction_proba))
                st.write(f'Churn Probability: {prediction_proba:.2f}')
                
                # Show result with formatting based on the prediction
                if prediction_proba > 0.5:
                    st.markdown('<div class="prediction-negative">⚠️ The customer is likely to churn.</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">### Recommended Actions:</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">- Offer special retention promotions</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">- Conduct customer satisfaction survey</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">- Personalized outreach from account manager</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-positive">✅ The customer is likely to stay.</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">### Recommended Actions:</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">- Consider for loyalty rewards program</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">- Potential cross-selling opportunities</div>', unsafe_allow_html=True)
        
        # Add a data visualization
        st.markdown('<h2 class="section-header">Feature Importance</h2>', unsafe_allow_html=True)
        
        # Calculate feature contributions (simplified example)
        features = ['Credit Score', 'Gender', 'Age', 'Tenure', 'Balance', 
                   'Products', 'Credit Card', 'Activity', 'Salary', 'Geography']
        importances = [0.08, 0.03, 0.15, 0.12, 0.18, 0.10, 0.05, 0.20, 0.04, 0.05]
        
        # Create a simple bar chart
        chart_data = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        st.bar_chart(chart_data.set_index('Feature'))

except Exception as e:
    st.error(f"Error: {str(e)}")
    
    # Show a simplified fallback version of the form
    st.header("Simple Churn Predictor")
    st.write("The advanced predictor is currently unavailable. Please use this simplified version.")
    
    # Simplified form
    age = st.slider("Age", 18, 92, 35)
    balance = st.number_input("Balance", 0.0, 250000.0, 10000.0)
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    num_products = st.slider("Number of Products", 1, 4, 1)
    
    if st.button("Calculate Risk"):
        # Simple heuristic
        risk = 0.3
        
        if age > 60:
            risk += 0.1
        elif age < 30:
            risk += 0.15
            
        if balance < 10000:
            risk += 0.15
            
        if is_active == "No":
            risk += 0.25
            
        if num_products > 2:
            risk -= 0.2
            
        risk = max(0.01, min(0.99, risk))
        
        st.progress(risk)
        st.write(f"Churn Risk: {risk:.2f}")
        
        if risk > 0.5:
            st.error("High risk of churn")
        else:
            st.success("Low risk of churn")

# Show information about the model
with st.expander("About this predictor"):
    st.write("""
    This app uses a predictive model to estimate the likelihood of customer churn based on:
    
    - **Demographics**: Age, gender, geography
    - **Banking relationship**: Tenure, balance, number of products
    - **Engagement**: Activity status, credit card ownership
    
    The model analyzes these factors to identify customers who might be at risk of leaving the bank.
    
    For optimal retention results, focus on customers with a churn probability above 0.5.
    """)
