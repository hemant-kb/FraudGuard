
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from hf import explain_prediction

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="wide")

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    .fraud-alert {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
        color: #000;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .safe-alert {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        color: #000;
        font-size: 1.2rem;
        font-weight: bold;
    }
    @media (prefers-color-scheme: dark) {
        .fraud-alert { background-color: #3d1a1a; color: #fff; }
        .safe-alert { background-color: #1a3d1a; color: #fff; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load trained model, threshold, feature names, and frequency maps"""
    with open('fraud_model_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('optimal_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('frequency_maps.pkl', 'rb') as f:
        freq_maps = pickle.load(f)
    
    # Extract only category and state frequencies
    category_freq_map = freq_maps['category']
    state_freq_map = freq_maps['state']
    
    return pipeline, threshold, features, category_freq_map, state_freq_map

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius = 6371 km


def engineer_features(user_inputs, category_freq_map, state_freq_map):
    """
    Transform user inputs into model features
    Uses real frequencies for category and state
    
    Args:
        user_inputs: dict with user-provided values
        category_freq_map: dict of category frequencies
        state_freq_map: dict of state frequencies
    
    Returns:
        DataFrame: Features ready for model preprocessing
    """
    
    # Extract user inputs
    amount = user_inputs['amount']
    category = user_inputs['category']
    state = user_inputs['state']
    hour = user_inputs['hour']
    day_of_week = user_inputs['day_of_week']
    minutes_since_last = user_inputs['minutes_since_last']
    user_lat = user_inputs['user_lat']
    user_long = user_inputs['user_long']
    merchant_lat = user_inputs['merchant_lat']
    merchant_long = user_inputs['merchant_long']
    city_population = user_inputs['city_population']
    gender = user_inputs['gender']
    avg_transaction_amount = user_inputs['avg_transaction_amount']
    
    # Calculate derived features
    amt_log = np.log1p(amount)
    user_amt_mean = avg_transaction_amount
    user_amt_std = avg_transaction_amount * 0.3
    amt_zscore = (amount - user_amt_mean) / (user_amt_std + 1e-6)
    
    time_since_last_txn = minutes_since_last * 60
    geo_distance = haversine_distance(user_lat, user_long, merchant_lat, merchant_long)
    
    is_weekend = 1 if day_of_week in [5, 6] else 0
    txns_last_1hr = 1 if minutes_since_last <= 60 else 0
    txns_last_10min = 1 if minutes_since_last <= 10 else 0
    
    # Frequency encodings - use real values from frequency maps
    category_freq = category_freq_map.get(category, 0.07)  # Default ~7% if category not found
    state_freq = state_freq_map.get(state, 0.02)  # Default ~2% if state not found
    
    # Create DataFrame with features (15 features total - removed merchant_freq, city_freq, job_freq)
    model_input = pd.DataFrame({
        # Numerical features (will be scaled by RobustScaler)
        'amt_log': [amt_log],
        'user_amt_mean': [user_amt_mean],
        'user_amt_std': [user_amt_std],
        'amt_zscore': [amt_zscore],
        'time_since_last_txn': [time_since_last_txn],
        'geo_distance': [geo_distance],
        'city_pop': [city_population],
        
        # Binary/Ordinal features (passthrough, no scaling)
        'hour': [hour],
        'dayofweek': [day_of_week],
        'is_weekend': [is_weekend],
        'txns_last_1hr': [txns_last_1hr],
        'txns_last_10min': [txns_last_10min],
        
        # Frequency encoded features (ONLY category and state with real frequencies!)
        'state_freq': [state_freq],
        'category_freq': [category_freq],
        
        # Categorical feature (will be one-hot encoded)
        'gender': [gender]
    })
    
    return model_input


def get_shap_explanation(pipeline, model_input, feature_names):
    """Calculate SHAP values and extract top features"""
    try:
        model = pipeline.named_steps['model']
        X_transformed = pipeline.named_steps['prep'].transform(model_input)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_transformed)
        
        # Set feature names for proper display in plots
        shap_values.feature_names = list(feature_names)
        
        # Extract top 5 features by absolute SHAP value
        shap_array = shap_values.values[0]
        top_indices = np.argsort(np.abs(shap_array))[-5:][::-1]
        
        shap_dict = {
            feature_names[i]: float(shap_array[i]) 
            for i in top_indices
        }
        
        return shap_dict, shap_values
        
    except Exception as e:
        st.error(f"❌ SHAP calculation failed: {str(e)}")
        return {}, None

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown("""
    **Step 1:** Enter transaction details
    
    **Step 2:** App engineers model features
    
    **Step 3:** XGBoost predicts fraud probability
    
    **Step 4:** SHAP explains the decision
    
    **Step 5:** LLM Analysis
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Metrics Guide")
    st.markdown("""
    **Fraud Probability**
    - Model's confidence (0-100%)
    - Example: 87% = likely fraud
    
    **Threshold**
    - Decision boundary
    - From business optimization
    
    **Confidence**
    - How certain the model is
    - High = very sure
    - Low = uncertain
    """)
    
    st.markdown("---")
    
    st.info("💡 Using real frequency data for category and state!")

# ============================================================================
# MAIN APP
# ============================================================================
st.title("💳 Credit Card Fraud Detection")
st.markdown("**Streamlined with Essential Features Only**")

# Load model artifacts
pipeline, threshold, feature_names, category_freq_map, state_freq_map = load_model_artifacts()

st.success(f"✅ Model loaded | Threshold: {threshold:.3f} | Features: {len(feature_names)}")

st.markdown("---")

# ============================================================================
# INPUT FORM
# ============================================================================
with st.form("fraud_check_form"):
    st.subheader("📝 Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Transaction details
    with col1:
        st.markdown("**💰 Transaction Info**")
        amount = st.number_input("Amount ($)", min_value=0.01, value=150.0, step=10.0)
        
        category = st.selectbox("Category", [
            "gas_transport", "grocery_pos", "shopping_net", "shopping_pos",
            "food_dining", "entertainment", "personal_care", "health_fitness",
            "travel", "kids_pets", "misc_pos", "misc_net", "home"
        ])
        
        hour = st.slider("Hour (0-23)", 0, 23, 14)
        
        day_of_week = st.selectbox(
            "Day of Week", 
            options=[(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"),
                     (4, "Friday"), (5, "Saturday"), (6, "Sunday")],
            format_func=lambda x: x[1]
        )[0]
        
        minutes_since_last = st.number_input("Minutes Since Last Transaction", min_value=1, max_value=10000, value=120)
    
    # Column 2: Location details
    with col2:
        st.markdown("**📍 Location Info**")
        
        state = st.selectbox("State", [
            'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
            'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
            'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
            'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
            'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
        ], index=34)  # Default to NY (index 34)
        
        city_population = st.number_input("City Population", min_value=1000, max_value=10000000,value=500000, step=10000)
        
        st.markdown("**Your Location**")
        user_lat = st.number_input("Latitude", value=40.7128, format="%.4f")
        user_long = st.number_input("Longitude", value=-74.0060, format="%.4f")
        
        st.markdown("**Merchant Location**")
        merchant_lat = st.number_input("Merchant Lat", value=40.7580, format="%.4f")
        merchant_long = st.number_input("Merchant Long", value=-73.9855, format="%.4f")
    
    # Column 3: User details
    with col3:
        st.markdown("**👤 User Profile**")
        
        gender = st.selectbox("Gender", ["M", "F"])
        
        avg_transaction_amount = st.number_input("Your Avg Transaction ($)", min_value=10.0,max_value=10000.0,value=100.0, step=10.0)
        
        st.markdown("---")
        st.caption("💡 Simplified inputs for faster analysis")
    
    # Submit button
    submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary", use_container_width=True)

# ============================================================================
# PROCESS SUBMISSION
# ============================================================================
if submitted:
    
    with st.spinner("🔄 Analyzing transaction..."):
        
        # Collect inputs
        user_inputs = {
            'amount': amount,
            'category': category,
            'state': state,
            'hour': hour,
            'day_of_week': day_of_week,
            'minutes_since_last': minutes_since_last,
            'user_lat': user_lat,
            'user_long': user_long,
            'merchant_lat': merchant_lat,
            'merchant_long': merchant_long,
            'city_population': city_population,
            'gender': gender,
            'avg_transaction_amount': avg_transaction_amount
        }
        
        # Engineer features
        model_input = engineer_features(user_inputs, category_freq_map, state_freq_map)
        
        # Calculate distance for display
        calculated_distance = haversine_distance(user_lat, user_long, merchant_lat, merchant_long)
        
        st.info(f"✅ Engineered {len(model_input.columns)} features | Distance: {calculated_distance:.1f} km")
        
        # Predict
        fraud_probability = float(pipeline.predict_proba(model_input)[0, 1])
        is_fraud = fraud_probability >= threshold
        confidence = float(abs(fraud_probability - 0.5) * 2)
        
        st.markdown("---")
        
        # ANSWER 1: IS IT FRAUDULENT?
        st.subheader("1️⃣ Is This Transaction Fraudulent?")
        
        if is_fraud:
            st.markdown('<div class="fraud-alert">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-alert">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ANSWER 2: HOW CONFIDENT?
        st.subheader("2️⃣ How Confident Is The Model?")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Fraud Probability", f"{fraud_probability:.1%}")
            st.caption("Model's raw output")
        
        with col_b:
            st.metric("Confidence", f"{confidence:.1%}")
            st.caption("How certain")
        
        with col_c:
            st.metric("Decision Threshold", f"{threshold:.1%}")
            st.caption("Business-optimized")
        
        st.progress(fraud_probability)
        
        decision_symbol = "≥" if is_fraud else "<"
        decision_text = "**FRAUD**" if is_fraud else "**LEGITIMATE**"
        
        st.info(f"""
        **Decision:** Probability ({fraud_probability:.1%}) {decision_symbol} Threshold ({threshold:.1%}) → {decision_text}
        
        **Confidence:** {confidence:.0%} certainty (High >70%, Low <30%)
        """)
        
        st.markdown("---")
        
        # ANSWER 3: WHY?
        st.subheader("3️⃣ Why Did The Model Make This Decision?")
        
        shap_dict, shap_values = get_shap_explanation(pipeline, model_input, feature_names)
        
        if shap_values is not None and shap_dict:
            
            # SHAP Waterfall Plot
            st.markdown("**📊 SHAP Waterfall Plot**")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            plt.close()
            
            st.caption("""
            **Red (positive)** → Increases fraud | **Blue (negative)** → Decreases fraud
            """)
            
            # Top Features
            st.markdown("**🔝 Top 5 Contributing Features**")
            for i, (feature, value) in enumerate(shap_dict.items(), 1):
                emoji = "🔴" if value > 0 else "🔵"
                direction = "increases" if value > 0 else "decreases"
                st.markdown(f"{i}. {emoji} **{feature}**: {direction} fraud probability by **{abs(value):.4f}**")
            
            st.markdown("---")
            
            # LLM Explanation
            st.markdown("**🤖 AI Analysis**")
            
            with st.spinner("🧠 Generating explanation..."):
                try:
                    prediction_label = "Fraudulent" if is_fraud else "Legitimate"
                    llm_explanation = explain_prediction(
                        prediction=prediction_label,
                        probability=fraud_probability,
                        shap_dict=shap_dict
                    )
                    st.info(llm_explanation)
                except Exception as e:
                    st.warning(f"⚠️ LLM explanation unavailable: {str(e)}")
        
        # Show Engineered Features
        st.markdown("---")
        with st.expander("🔬 View All Engineered Features"):
            feature_display = model_input.T.copy()
            feature_display.columns = ["Value"]
            feature_display["Value"] = feature_display["Value"].astype(str)
            st.dataframe(feature_display, width="stretch")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    Built with XGBoost, SHAP, Streamlit, and Hugging Face LLMs<br>
    Simplified for production deployment
</div>
""", unsafe_allow_html=True)
