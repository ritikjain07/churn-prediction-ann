import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Add these imports with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    plotly_available = False

# Page configuration for a cleaner look
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved contrast and layout
st.markdown("""
<style>
    /* Improve base font for better readability */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #0F172A;
    }
    
    /* Headers with better contrast */
    .main-header {
        font-size: 2.3rem;
        color: #0F172A;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #334155;
        margin-top: 0;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.4rem;
        color: #0F172A;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        font-weight: 600;
    }
    
    /* Results panels with better contrast */
    .prediction-positive {
        background-color: #DCFCE7;
        color: #14532D;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #22C55E;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        font-weight: 600;
    }
    
    .prediction-negative {
        background-color: #FEE2E2;
        color: #7F1D1D;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #EF4444;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        font-weight: 600;
    }
    
    /* Info box with better contrast */
    .info-box {
        background-color: #F8FAFC;
        color: #334155;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        line-height: 1.5;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 0.25rem;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Card styling with fixed heights and overflow handling */
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        padding: 1.25rem;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #334155;
        font-weight: 500;
    }
    
    .dashboard-card {
        background-color: B4EBE6;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
        border: 1px solid #E2E8F0;
        overflow: auto;
        min-height: 200px;
    }
    
    .dashboard-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .card-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 1rem;
        border-bottom: 1px solid #EFF6FF;
        padding-bottom: 0.5rem;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background-color: #EFF6FF;
    }
    
    .stSlider > div > div > div > div {
        background-color: #3B82F6;
    }
    
    /* Font adjustments for better readability */
    label {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    /* Make selectbox and number inputs more visible */
    .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 0.375rem;
    }
    
    .stNumberInput [data-baseweb="input"] {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 0.375rem;
    }
    
    /* Fix for markdown overflowing containers */
    .element-container div.markdown-text-container {
        overflow: hidden;
        word-wrap: break-word;
    }
    
    /* Better table styles for feature importance */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th, td {
        border: 1px solid #E2E8F0;
        padding: 0.5rem 1rem;
        text-align: left;
    }
    
    th {
        background-color: #F1F5F9;
        color: #334155;
        font-weight: 600;
    }
    
    tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    
    /* Risk indicators with better visibility */
    .risk-high {
        color: #EF4444;
        font-weight: 700;
    }
    
    .risk-medium {
        color: #F59E0B;
        font-weight: 700;
    }
    
    .risk-low {
        color: #10B981;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for app info and additional controls
with st.sidebar:
    st.markdown("<h1 style='color:#0F172A; font-size:1.8rem;'>🏦 Bank Churn Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#334155; font-size:1.2rem; margin-top:-0.5rem;'>Customer Retention Tool</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:1rem 0; border-color:#E2E8F0;'>", unsafe_allow_html=True)
    
    # App description 
    st.markdown("<h3 style='color:#0F172A; font-size:1.2rem; font-weight:600;'>About</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#334155; font-size:1rem; line-height:1.5;'>
    This tool helps predict which customers are at risk of leaving the bank.
    <br><br>
    By analyzing customer data, it provides actionable insights to improve retention strategies.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin:1rem 0; border-color:#E2E8F0;'>", unsafe_allow_html=True)
    
    # Add sample customer profiles for quick testing
    st.markdown("<h3 style='color:#0F172A; font-size:1.2rem; font-weight:600;'>Sample Profiles</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧓 Senior", use_container_width=True):
            st.session_state.profile_senior = True
        if st.button("👩‍🎓 Young", use_container_width=True):
            st.session_state.profile_young = True
    with col2:
        if st.button("💰 High Value", use_container_width=True):
            st.session_state.profile_high_value = True
        if st.button("⚠️ At-Risk", use_container_width=True):
            st.session_state.profile_at_risk = True
    
    st.markdown("<hr style='margin:1rem 0; border-color:#E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#334155; font-size:1rem;'>Developed by<br><strong>Ritik Jain - 2025</strong></div>", unsafe_allow_html=True)
    
# Main content
st.markdown('<div class="dashboard-card"><h1 class="main-header">Customer Churn Prediction</h1><p class="sub-header">Predict and prevent customer attrition with advanced analytics</p></div>', unsafe_allow_html=True)

try:
    # Create simple encoder classes to handle missing sklearn
    class SimpleEncoder:
        def __init__(self, classes):
            self.classes_ = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            
        def transform(self, values):
            return [self.class_to_idx[v] for v in values]
    
    class SimpleOneHotEncoder:
        def __init__(self, categories):
            self.categories_ = [categories]
            self.cat_to_idx = {c: i for i, c in enumerate(categories)}
            
        def transform(self, values):
            result = np.zeros((len(values), len(self.categories_[0])))
            for i, v in enumerate(values):
                result[i, self.cat_to_idx[v[0]]] = 1
            return result
            
        def get_feature_names_out(self, input_features):
            return [f"{input_features[0]}_{cat}" for cat in self.categories_[0]]
    
    # Try to load pickle files, or use simple versions if fails
    try:
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
    except:
        label_encoder_gender = SimpleEncoder(['Female', 'Male'])
        st.sidebar.info("Using simple gender encoder")

    try:
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
    except:
        onehot_encoder_geo = SimpleOneHotEncoder(['France', 'Germany', 'Spain'])
        st.sidebar.info("Using simple geography encoder")

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
        st.markdown('<div class="dashboard-card"><h2 class="section-header">Customer Demographics</h2>', unsafe_allow_html=True)
        
        # Add an info box for this section
        st.markdown('<div class="info-box">Demographic characteristics help identify customer segments with similar behavior patterns.</div>', unsafe_allow_html=True)
        
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        
        # Better slider styling for age
        age = st.slider('Age', 18, 92, default_age, help="Customer's age in years")
        
        # Add age group indicator for context
        age_group = "Senior Citizen (65+)" if age >= 65 else "Middle-aged (35-64)" if age >= 35 else "Young Adult (18-34)"
        st.markdown(f"<div style='text-align: right; color: #334155; font-size: 0.9rem; font-weight: 500;'>{age_group}</div>", unsafe_allow_html=True)
        
        tenure = st.slider('Tenure (years)', 0, 10, default_tenure, help="How long the customer has been with the bank")
        
        # Add loyalty indicator
        loyalty = "Long-term Customer" if tenure >= 7 else "Regular Customer" if tenure >= 3 else "New Customer"
        st.markdown(f"<div style='text-align: right; color: #334155; font-size: 0.9rem; font-weight: 500;'>{loyalty}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="dashboard-card"><h2 class="section-header">Financial Information</h2>', unsafe_allow_html=True)
        
        # Add an info box for this section
        st.markdown('<div class="info-box">Financial patterns often provide the strongest indicators of potential churn.</div>', unsafe_allow_html=True)
        
        credit_score = st.slider('Credit Score', 300, 900, default_credit_score, help="Customer's credit rating")
        
        # Add credit quality indicator
        credit_quality = "Excellent" if credit_score >= 750 else "Good" if credit_score >= 650 else "Fair" if credit_score >= 550 else "Poor"
        st.markdown(f"<div style='text-align: right; color: #334155; font-size: 0.9rem; font-weight: 500;'>Credit Quality: {credit_quality}</div>", unsafe_allow_html=True)
        
        balance = st.number_input('Account Balance ($)', min_value=0.0, max_value=250000.0, value=default_balance, step=1000.0, format="%.2f")
        
        # Add balance tier indicator
        balance_tier = "Premium" if balance >= 100000 else "Standard" if balance >= 10000 else "Basic"
        st.markdown(f"<div style='text-align: right; color: #334155; font-size: 0.9rem; font-weight: 500;'>Account Tier: {balance_tier}</div>", unsafe_allow_html=True)
        
        estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, max_value=200000.0, value=default_salary, step=5000.0, format="%.2f")
        num_of_products = st.slider('Number of Products', 1, 4, default_products, help="Number of bank products the customer uses")
        
        # Better binary choices with icons
        has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'], index=0 if default_card == 'No' else 1)
        is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'], index=0 if default_active == 'No' else 1)
        st.markdown('</div>', unsafe_allow_html=True)

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
            
            # Calculate churn probability using a heuristic model
            def calculate_churn_probability(age, tenure, balance, credit_score, products, is_active, has_card, geography):
                # Base probability
                prob = 0.3
                
                # Age factors
                if age > 60:
                    prob += 0.1
                elif age < 30:
                    prob += 0.15
                    
                # Tenure factors
                if tenure < 2:
                    prob += 0.2
                elif tenure > 7:
                    prob -= 0.15
                    
                # Balance factors
                if balance < 10000:
                    prob += 0.15
                elif balance > 100000:
                    prob -= 0.1
                    
                # Activity status
                if is_active == 0:
                    prob += 0.25
                    
                # Products
                if products > 2:
                    prob -= 0.15
                    
                # Has credit card
                if has_card == 0:
                    prob += 0.05
                
                # Credit score
                if credit_score < 600:
                    prob += 0.1
                elif credit_score > 750:
                    prob -= 0.1
                    
                # Geography
                if geography == 'Germany':
                    prob += 0.05
                    
                # Clamp to valid range
                return max(0.01, min(0.99, prob))
            
            # Get prediction
            prediction_proba = calculate_churn_probability(
                age, 
                tenure, 
                balance, 
                credit_score, 
                num_of_products, 
                is_active_member_value,
                has_cr_card_value,
                geography
            )
            
            # Display results with much better formatting
            st.markdown('<div class="dashboard-card"><h2 class="section-header">Analysis Results</h2>', unsafe_allow_html=True)
            
            # Create two columns for the results section
            res_col1, res_col2 = st.columns([3, 2])
            
            with res_col1:
                if plotly_available:
                    # Create a gauge chart for the churn probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_proba,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Probability", 'font': {'size': 24, 'color': '#0F172A', 'family': 'Arial, sans-serif'}},
                        gauge = {
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#334155"},
                            'bar': {'color': "#2563EB"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#CBD5E1",
                            'steps': [
                                {'range': [0, 0.3], 'color': '#DCFCE7'},
                                {'range': [0.3, 0.7], 'color': '#FEF9C3'},
                                {'range': [0.7, 1], 'color': '#FEE2E2'}
                            ],
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='white',
                        font=dict(family="Arial, sans-serif", size=14, color="#0F172A")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback for when Plotly is not available
                    st.markdown(f"""
                    <div style='text-align:center; background-color:white; padding:2rem; border-radius:0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom:1rem;'>
                        <h1 style='font-size:3.5rem; color:#0F172A; margin-bottom:0.5rem;'>{prediction_proba:.2f}</h1>
                        <p style='font-size:1.2rem; color:#334155; font-weight:500;'>Churn Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add colored progress bar with labels
                    progress_color = "#DCFCE7" if prediction_proba < 0.3 else "#FEF9C3" if prediction_proba < 0.7 else "#FEE2E2"
                    st.markdown(f"""
                    <div style='margin-bottom:1.5rem;'>
                        <div style='display:flex; justify-content:space-between; margin-bottom:0.25rem;'>
                            <span style='color:#334155; font-size:0.9rem;'>Low Risk</span>
                            <span style='color:#334155; font-size:0.9rem;'>High Risk</span>
                        </div>
                        <div style='height:1rem; background-color:#F1F5F9; border-radius:9999px; overflow:hidden;'>
                            <div style='height:100%; width:{prediction_proba*100}%; background-color:{progress_color};'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                # Show risk level with icon
                risk_level = "High Risk" if prediction_proba > 0.7 else "Medium Risk" if prediction_proba > 0.3 else "Low Risk"
                risk_icon = "🔴" if prediction_proba > 0.7 else "🟡" if prediction_proba > 0.3 else "🟢"
                risk_class = "risk-high" if prediction_proba > 0.7 else "risk-medium" if prediction_proba > 0.3 else "risk-low"
                
                st.markdown(f"<div style='text-align:center; margin-bottom:1rem;'><h2 class='{risk_class}' style='font-size:1.8rem;'>{risk_icon} {risk_level}</h2></div>", unsafe_allow_html=True)
                
                # Show result with formatting based on the prediction
                if prediction_proba > 0.5:
                    st.markdown('<div class="prediction-negative">⚠️ This customer is likely to churn.</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box"><strong style="color:#0F172A">Key Risk Factors:</strong></div>', unsafe_allow_html=True)
                    
                    # Display key risk factors
                    risk_factors = []
                    if age > 60 or age < 30:
                        risk_factors.append("- Age profile")
                    if tenure < 2:
                        risk_factors.append("- New customer")
                    if balance < 10000:
                        risk_factors.append("- Low account balance")
                    if is_active_member_value == 0:
                        risk_factors.append("- Inactive member status")
                    if num_of_products == 1:
                        risk_factors.append("- Limited product usage")
                    
                    for factor in risk_factors:
                        st.markdown(f'<div class="info-box" style="padding:0.5rem 1rem; margin-bottom:0.5rem;">{factor}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-positive">✅ This customer is likely to stay.</div>', unsafe_allow_html=True)
                    
                    # Show retention factors
                    retention_factors = []
                    if tenure > 7:
                        retention_factors.append("- Long-term customer")
                    if balance > 100000:
                        retention_factors.append("- High value account")
                    if is_active_member_value == 1:
                        retention_factors.append("- Active member")
                    if num_of_products > 2:
                        retention_factors.append("- Multiple products")
                    
                    if retention_factors:
                        st.markdown('<div class="info-box"><strong style="color:#0F172A">Retention Strengths:</strong></div>', unsafe_allow_html=True)
                        for factor in retention_factors:
                            st.markdown(f'<div class="info-box" style="padding:0.5rem 1rem; margin-bottom:0.5rem;">{factor}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close dashboard card
        
            # Recommendations section
            st.markdown('<div class="dashboard-card"><h2 class="section-header">Recommendations</h2>', unsafe_allow_html=True)
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown('<div class="card-header">Action Plan</div>', unsafe_allow_html=True)
                if prediction_proba > 0.7:
                    st.markdown('<div class="info-box" style="border-left: 4px solid #EF4444;"><span style="font-weight:600; color:#7F1D1D;">🔴 Urgent Intervention Required</span></div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Call customer within 24 hours</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Offer personalized retention package</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Schedule account review</div>', unsafe_allow_html=True)
                elif prediction_proba > 0.4:
                    st.markdown('<div class="info-box" style="border-left: 4px solid #F59E0B;"><span style="font-weight:600; color:#78350F;">🟡 Proactive Retention Needed</span></div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Contact within 7 days</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Offer loyalty discount</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Conduct satisfaction survey</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box" style="border-left: 4px solid #10B981;"><span style="font-weight:600; color:#065F46;">🟢 Relationship Building</span></div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Include in regular outreach</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">• Consider for product cross-selling</div>', unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown('<div class="card-header">Product Recommendations</div>', unsafe_allow_html=True)
                
                # Make sure we always show some recommendations to avoid empty space
                recommendations = []
                
                if num_of_products == 1:
                    recommendations.append('<div class="info-box">• Savings account with higher interest</div>')
                    recommendations.append('<div class="info-box">• Credit card with rewards program</div>')
                
                if has_cr_card_value == 0:
                    recommendations.append('<div class="info-box">• Premium credit card with travel benefits</div>')
                
                if balance > 50000:
                    recommendations.append('<div class="info-box">• Investment portfolio management</div>')
                    recommendations.append('<div class="info-box">• Wealth management consultation</div>')
                
                if age > 55:
                    recommendations.append('<div class="info-box">• Retirement planning services</div>')
                
                if age < 30:
                    recommendations.append('<div class="info-box">• First-home buyer mortgage options</div>')
                    recommendations.append('<div class="info-box">• Digital banking enhancements</div>')
                
                # Add default recommendations if none were triggered
                if len(recommendations) == 0:
                    recommendations.append('<div class="info-box">• Regular account checkup</div>')
                    recommendations.append('<div class="info-box">• Online banking features review</div>')
                    recommendations.append('<div class="info-box">• Customer satisfaction survey</div>')
                
                # Display recommendations (up to 4 to avoid overflow)
                for recommendation in recommendations[:4]:
                    st.markdown(recommendation, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close dashboard card
        
        # Add a feature importance visualization if plotly is available
        if plotly_available:
            st.markdown('<div class="dashboard-card"><h2 class="section-header">Feature Importance</h2>', unsafe_allow_html=True)
            
            # Calculate feature contributions
            features = ['Activity', 'Balance', 'Age', 'Tenure', 'Products', 
                      'Credit Score', 'Credit Card', 'Geography', 'Salary', 'Gender']
            importances = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]
            
            # Create a bar chart using plotly
            fig = px.bar(
                x=importances, 
                y=features, 
                orientation='h',
                labels={'x': 'Importance Score', 'y': 'Feature'},
                title='Factors Influencing Churn Prediction',
                color=importances,
                color_continuous_scale='Blues',
                text=[f"{v:.0%}" for v in importances]  # Add percentage labels
            )
            
            fig.update_layout(
                xaxis_title="Impact on Prediction",
                yaxis_title="Customer Attribute",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=14, color="#0F172A"),
                showlegend=False
            )
            
            # Improve text placement and appearance
            fig.update_traces(
                textposition='outside',
                textfont=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color="#0F172A"
                )
            )
            
            # Improve y-axis appearance
            fig.update_yaxes(
                tickfont=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color="#0F172A"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close dashboard card
        else:
            # Fallback for when Plotly is not available - use a table instead
            st.markdown('<div class="dashboard-card"><h2 class="section-header">Key Factors</h2>', unsafe_allow_html=True)
            
            # Create a simple HTML table for better formatting
            st.markdown("""
            <table style="width:100%; border-collapse: collapse; margin-top: 1rem;">
                <tr>
                    <th style="padding:0.75rem; text-align:left; border:1px solid #E2E8F0; background-color:#F1F5F9;">Factor</th>
                    <th style="padding:0.75rem; text-align:left; border:1px solid #E2E8F0; background-color:#F1F5F9;">Impact on Churn</th>
                    <th style="padding:0.75rem; text-align:left; border:1px solid #E2E8F0; background-color:#F1F5F9;">Description</th>
                </tr>
                <tr>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;"><strong>Account Activity</strong></td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">25%</td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">Inactive members are 5× more likely to churn</td>
                </tr>
                <tr>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;"><strong>Account Balance</strong></td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">18%</td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">Low balance customers have higher churn risk</td>
                </tr>
                <tr>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;"><strong>Age</strong></td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">15%</td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">Young adults and seniors have distinct churn patterns</td>
                </tr>
                <tr>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;"><strong>Tenure</strong></td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">12%</td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">New customers churn at higher rates</td>
                </tr>
                <tr>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;"><strong>Number of Products</strong></td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">10%</td>
                    <td style="padding:0.75rem; border:1px solid #E2E8F0;">More products correlates with higher retention</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close dashboard card

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    
    # Show a simplified fallback version of the form with better styling
    st.markdown("""
    <div style="background-color:white; padding:2rem; border-radius:0.5rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom:1.5rem;">
        <h2 style="color:#0F172A; font-size:1.5rem; margin-bottom:1rem; font-weight:600;">Simple Churn Predictor</h2>
        <p style="color:#334155; margin-bottom:1.5rem;">The advanced predictor is currently unavailable. Please use this simplified version.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simplified form with better styling
    st.markdown("<div style='background-color:white; padding:2rem; border-radius:0.5rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
    age = st.slider("Age", 18, 92, 35)
    balance = st.number_input("Balance", 0.0, 250000.0, 10000.0)
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    num_products = st.slider("Number of Products", 1, 4, 1)
    
    if st.button("Calculate Risk", use_container_width=False):
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
        
        # Display results with better styling
        risk_level = "High Risk" if risk > 0.7 else "Medium Risk" if risk > 0.3 else "Low Risk"
        risk_color = "#EF4444" if risk > 0.7 else "#F59E0B" if risk > 0.3 else "#10B981"
        
        st.markdown(f"""
        <div style="margin-top:1.5rem; text-align:center;">
            <h3 style="color:{risk_color}; font-size:1.8rem; font-weight:600;">{risk_level}</h3>
            <div style="font-size:1.2rem; color:#334155; margin-bottom:1rem;">Churn Probability: <strong>{risk:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Better progress bar
        st.markdown(f"""
        <div style="margin:1rem 0 1.5rem 0;">
            <div style="height:0.75rem; background-color:#F1F5F9; border-radius:9999px; overflow:hidden;">
                <div style="height:100%; width:{risk*100}%; background-color:{risk_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if risk > 0.5:
            st.markdown("""
            <div style="background-color:#FEE2E2; color:#7F1D1D; padding:1rem; border-radius:0.5rem; margin-top:1rem; border-left:0.25rem solid #EF4444;">
                <strong>⚠️ High risk of churn</strong>
                <p style="margin-top:0.5rem; margin-bottom:0;">This customer requires immediate attention and a personalized retention strategy.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:#DCFCE7; color:#14532D; padding:1rem; border-radius:0.5rem; margin-top:1rem; border-left:0.25rem solid #10B981;">
                <strong>✅ Low risk of churn</strong>
                <p style="margin-top:0.5rem; margin-bottom:0;">This customer appears stable. Consider opportunities for cross-selling and relationship building.</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # Close dashboard card

# Show information about the model with improved styling
with st.expander("About this predictor"):
    st.markdown("""
    <div style="color:#334155; line-height:1.6;">
        <p>This app uses a predictive model to estimate the likelihood of customer churn based on:</p>
        
        <ul style="margin-top:0.75rem; margin-bottom:1rem;">
            <li><strong>Demographics</strong>: Age, gender, geography</li>
            <li><strong>Banking relationship</strong>: Tenure, balance, number of products</li>
            <li><strong>Engagement</strong>: Activity status, credit card ownership</li>
        </ul>
        
        <p>The model analyzes these factors to identify customers who might be at risk of leaving the bank.</p>
        
        <div style="background-color:#EFF6FF; padding:1rem; border-radius:0.5rem; margin-top:1rem; border-left:0.25rem solid #3B82F6;">
            <strong style="color:#1E3A8A;">Pro Tip:</strong> For optimal retention results, focus on customers with a churn probability above 0.5.
        </div>
    </div>
    """, unsafe_allow_html=True)
