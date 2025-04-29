import streamlit as st
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Set page title
st.set_page_config(page_title="Explain Prediction")
st.title("🔎 Heart Disease Prediction Explanation")

# Step 1: Check if session state contains prediction data
if "patient_features" not in st.session_state:
    st.error("No prediction data found. Please predict a patient's risk first.")
    st.stop()

# Step 2: Load the saved features and prediction
features_df = st.session_state.patient_features
predicted_prob = st.session_state.predicted_probability

# Step 3: Display patient inputs
st.subheader("🗂 Patient Feature Inputs")
st.dataframe(features_df)

# Step 4: Display the predicted probability
st.subheader("📈 Predicted Heart Disease Risk")
st.success(f"Predicted Probability: {predicted_prob:.2%}")

st.markdown("---")

# Step 5: Load model
# Load model
model = joblib.load('model.joblib')

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer(features_df)

# Modify feature names to include actual values
original_feature_names = features_df.columns.tolist()
feature_values = features_df.iloc[0].tolist()

modified_feature_names = [f"{name} = {value}" for name, value in zip(original_feature_names, feature_values)]

# Update Explanation object feature names
shap_values.feature_names = modified_feature_names

# 📊 Bar plot
st.subheader("📊 Feature Impact Summary (Bar Plot)")

fig, ax = plt.subplots()
shap.plots.bar(shap_values)
st.pyplot(fig)
