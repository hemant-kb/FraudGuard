# 💳 Credit Card Fraud Detection System

An intelligent, fraud detection system powered by **XGBoost**, **SHAP explainability**, and **AI-powered natural language explanations** using Hugging Face LLMs.

### Quick Summary

```
Input → Engineer Features → Scale/Encode → XGBoost → 
Compare to Threshold → SHAP Explains → LLM Translates → Display Results
```


### System Flow

```
1. USER ENTERS 13 INPUTS
   ├─ Transaction: Amount, Category, Hour, Day, Minutes since last
   ├─ Location: State, City population, User coords, Merchant coords
   └─ Profile: Gender, Average transaction amount

2. FEATURE ENGINEERING (13 → 15 features)
   ├─ Amount: amt_log, amt_zscore, user_amt_mean, user_amt_std
   ├─ Geographic: geo_distance (Haversine formula)
   ├─ Time: is_weekend, time_since_last_txn
   ├─ Velocity: txns_last_1hr, txns_last_10min
   └─ Frequency: state_freq, category_freq (from training data)

3. PREPROCESSING
   ├─ RobustScaler → 7 numerical features (amt_log, distances, etc.)
   ├─ Passthrough → 7 binary/ordinal features (hour, frequencies, etc.)
   └─ OneHotEncoder → gender (M=1, F=0)

4. XGBOOST PREDICTION
   ├─ Model: Gradient Boosted Trees (200-600 estimators, optimized)
   ├─ Output: fraud_probability (0-1, e.g., 0.87 = 87% fraud)
   ├─ Threshold: 0.23 (business-optimized for max profit)
   ├─ Decision: P ≥ 0.23 → FRAUD, P < 0.23 → LEGITIMATE
   └─ Confidence: |P - 0.5| × 2 (how far from uncertain 50%)

5. SHAP ANALYSIS
   ├─ TreeExplainer calculates each feature's contribution
   ├─ SHAP value: +positive = increases fraud, -negative = decreases
   ├─ Extract top 5 most impactful features
   └─ Generate waterfall plot (visual breakdown)

6. LLM EXPLANATION (SmolLM3-3B)
   ├─ Input: Prediction result + top 5 SHAP features
   ├─ Prompt: "Explain as Senior Fraud Analyst in 4 bullet points"
   └─ Output: Plain English analysis with Key Finding, Evidence, Risk, Next Steps

7. DISPLAY 3 ANSWERS
   ├─ Is it fraud? → Red alert (FRAUD) or Green alert (LEGITIMATE)
   ├─ How confident? → Probability (87%), Confidence (74%), Threshold (23%)
   └─ Why? → Waterfall plot + Top 5 features list + AI plain English explanation
```

<img width="1671" height="773" alt="image" src="https://github.com/user-attachments/assets/59438718-afd8-40da-9aec-753d6d1f52a8" />
