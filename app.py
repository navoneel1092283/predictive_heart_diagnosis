import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Function to assign color based on risk
def get_prediction_color(prob):
    if prob > 0.8:
        return "#FF0000"
    elif prob > 0.6:
        return "#FF8C00"
    elif prob > 0.4:
        return "#FFD700"
    elif prob > 0.2:
        return "#ADFF2F"
    else:
        return "#00FF7F"

# Load model and encoder ONCE
model = joblib.load('model.joblib')
ohe = joblib.load('ohe.joblib')

# Page Title
st.title("🫀 Heart Disease Risk Prediction")
st.sidebar.header("Instructions")
st.sidebar.info("Fill out all fields carefully and click Predict Risk.")

st.subheader("📝 Patient Information Form")

# --- Helper function ---
def validate_numeric(val, label, is_int=False):
    if val == "":
        st.error(f"{label} is required.")
        st.stop()
    try:
        return int(val) if is_int else float(val)
    except ValueError:
        st.error(f"{label} must be a valid {'integer' if is_int else 'number'}.")
        st.stop()

# Form starts
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input('Age')
        sex = st.selectbox('Sex', options=["-- Select --", "Female", "Male"])
        chest_pain_type = st.selectbox('Chest Pain Type (1–4)', options=["-- Select --", 1, 2, 3, 4])
        resting_bp = st.text_input('Resting Blood Pressure (mm Hg)')
        cholesterol = st.text_input('Cholesterol (mg/dL)')
        fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL?', options=["-- Select --", "No", "Yes"])
        resting_ecg = st.selectbox('Resting ECG Results (0–2)', options=["-- Select --", 0, 1, 2])

    with col2:
        max_heart_rate = st.text_input('Max Heart Rate Achieved')
        exercise_angina = st.selectbox('Exercise Induced Angina?', options=["-- Select --", "No", "Yes"])
        oldpeak = st.text_input('Oldpeak (ST Depression)')
        slope_st_segment = st.selectbox('Slope of ST Segment (1–3)', options=["-- Select --", 1, 2, 3])
        num_major_vessels = st.selectbox('Number of Major Vessels (0–3)', options=["-- Select --", 0, 1, 2, 3])
        thalassemia = st.selectbox('Thalassemia (3=Normal, 6=Fixed, 7=Reversible)', options=["-- Select --", 3, 6, 7])

    st.markdown("---")
    submit_button = st.form_submit_button("🔍 Predict Risk")

# --- After Submit ---
if submit_button:
    try:
        # --- Validate numeric fields ---
        age = validate_numeric(age, "Age", is_int=True)
        resting_bp = validate_numeric(resting_bp, "Resting BP")
        cholesterol = validate_numeric(cholesterol, "Cholesterol")
        max_heart_rate = validate_numeric(max_heart_rate, "Max Heart Rate")
        oldpeak = validate_numeric(oldpeak, "Oldpeak")

        # --- Validate selectboxes ---
        if sex == "-- Select --":
            st.error("Please select Sex."); st.stop()
        if chest_pain_type == "-- Select --":
            st.error("Please select Chest Pain Type."); st.stop()
        if fasting_bs == "-- Select --":
            st.error("Please select Fasting Blood Sugar condition."); st.stop()
        if resting_ecg == "-- Select --":
            st.error("Please select Resting ECG result."); st.stop()
        if exercise_angina == "-- Select --":
            st.error("Please select Exercise Induced Angina."); st.stop()
        if slope_st_segment == "-- Select --":
            st.error("Please select Slope of ST Segment."); st.stop()
        if num_major_vessels == "-- Select --":
            st.error("Please select Number of Major Vessels."); st.stop()
        if thalassemia == "-- Select --":
            st.error("Please select Thalassemia."); st.stop()

        # Convert categories to binary
        sex = 0.0 if sex == "Female" else 1.0
        fasting_bs = 0.0 if fasting_bs == "No" else 1.0
        exercise_angina = 0.0 if exercise_angina == "No" else 1.0

        # --- Prepare continuous features ---
        continuous_features = np.array([[
            age, sex, resting_bp, cholesterol, fasting_bs,
            max_heart_rate, oldpeak, num_major_vessels
        ]])
        continuous_df = pd.DataFrame(continuous_features, columns=[
            'age', 'sex', 'resting_bp', 'cholestoral', 'fasting_blood_sugar',
            'max_heart_rate', 'depression_induced_by_exercise', 'major_blood_vessels_flourosopy'
        ]).astype(float)

        # --- Prepare categorical features ---
        categorical_features = np.array([[
            chest_pain_type, resting_ecg, slope_st_segment,
            thalassemia, exercise_angina
        ]])
        categorical_df = pd.DataFrame(categorical_features, columns=[
            'chest_pain_type', 'restecg', 'slope_of_exercise', 'thal', 'exercise_induced_agina'
        ]).astype(float).astype(str)

        # One-hot encode categorical features
        encoded = ohe.transform(categorical_df).toarray()
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([
            'chest_pain_type', 'restecg', 'slope_of_exercise', 'thal', 'exercise_induced_agina'
        ]))

        # Merge final input
        final_df = pd.concat([continuous_df.reset_index(drop=True),
                              encoded_df.reset_index(drop=True)], axis=1)

        # --- Predict ---
        probability = model.predict_proba(final_df)[:, 1][0]
        color = get_prediction_color(probability)

        final_df.to_csv('final_df.csv', index = False)

        print(probability)

        # 🛡️ Save to session_state
        st.session_state.patient_features = final_df
        st.session_state.predicted_probability = probability

        # --- Display result ---
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 25px; border-radius: 12px;">
                <h2 style="color:black; text-align: center;">🫀 Probability of Heart Disease: {probability:.2%}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Developed for Cardiovascular Health | Powered by Streamlit")
