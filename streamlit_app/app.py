import streamlit as st
import pandas as pd
import joblib 
import os

# 1. ----Page Configuration----
st.set_page_config(page_title="Diabetes AI Diagnostic", layout="wide")

# 2. ----Model & Assets Loading----
@st.cache_resource
def load_models():
    # Get the directory where app.py is located (streamlit_app folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to the main folder (diabetes_indicator) where models are stored
    main_dir = os.path.dirname(current_dir)
    
    # Loading model files using path joining
    bin_mod = joblib.load(os.path.join(main_dir, 'binary_model.joblib'))
    multi_mod = joblib.load(os.path.join(main_dir, 'multiclass_model.joblib'))
    reg_mod = joblib.load(os.path.join(main_dir, 'regression_model.joblib'))
    scl = joblib.load(os.path.join(main_dir, 'scaler.joblib'))
    feats = joblib.load(os.path.join(main_dir, 'feature_names.joblib'))
    
    return bin_mod, multi_mod, reg_mod, scl, feats

# Initialize models and assets
binary_model, multiclass_model, regression_model, scaler, feature_names = load_models()

# 3. ----User Interface (UI)----
st.title("Clinical Diabetes Diagnostic Portal")
st.write("Enter Patient clinical data for three models for analysis.")

with st.form("main_form"):
    # Section 1: Lab Measurements
    st.subheader("Lab & Vital Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5)
        insulin = st.number_input("Insulin(mL)", 0.0, 100.0, 15.0)
        glucose_f = st.number_input("Fasting Glucose", 50, 400, 100)
    with c2:
        glucose_p = st.number_input("Postprandial Glucose(mg/DL)", 50, 500, 140)
        systolic = st.number_input("Systolic BP", 80, 200, 120)
        diastolic = st.number_input("Diastolic BP", 40, 130, 80)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    with c3: 
        chol_total = st.number_input("Total Cholesterol", 100, 400, 190)
        ldl = st.number_input("LDL Cholesterol", 50, 300, 110)
        hdl = st.number_input("HDL Cholesterol", 20, 100, 50)
    with c4:
        trigly = st.number_input("Triglycerides", 50, 500, 150)
        heart_rate = st.number_input("Heart Rate", 40, 150, 75)
        whr = st.number_input("Waist_to_Hip Ratio", 0.5, 1.5, 0.85)

    # Section 2: Demographic & History
    st.subheader("Patient Profile & History")
    h1, h2, h3 = st.columns(3)
    with h1:
        age = st.number_input("Age", 1, 120, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    with h2:
        fam_hist = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        hyp_hist = st.selectbox("Hypertension History", ["No", "Yes"])
        cardio_hist = st.selectbox("Cardiovascular History", ["No", "Yes"])
    with h3:
        edu = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        income = st.selectbox("Income Level", [1, 2, 3, 4, 5, 6, 7, 8])
        employment = st.selectbox("Employment Status", ["Employed", "Retired", "Student", "Unemployed"])

    st.divider()
    e1, e2, e3 = st.columns(3)
    with e1:
        diet = st.number_input("Diet Score (0-10)", 0.0, 10.0, 5.0)
        phys_act = st.number_input("Activity (min/week)", 0, 1000, 150)
    with e2:
        sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        screen = st.number_input("Screen Time", 0.0, 24.0, 4.0)
    with e3:
        alc_cons = st.number_input("Alcohol/Week", 0, 50, 0)
        ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Other"])

    submit = st.form_submit_button("Run Analysis", use_container_width=True)

# 4. ----Logic and Prediction----
if submit:
    try:
        # Initialize input dictionary with zeros based on feature names
        input_dict = {f: 0 for f in feature_names}

        # Mapping numerical inputs
        input_dict.update({
            'age': age, 'bmi': bmi, 'hba1c': hba1c, 'insulin_level': insulin,
            'glucose_fasting': glucose_f, 'systolic_bp': systolic, 'diastolic_bp': diastolic,
            'cholesterol_total': chol_total, 'triglycerides': trigly, 'waist_to_hip_ratio': whr,
            'diet_score': diet, 'physical_activity_minutes_per_week': phys_act, 'sleep_hours_per_day': sleep,
            'screen_time_hours_per_day': screen, 'alcohol_consumption_per_week': alc_cons,
            'education_level': edu, 'income_level': income, 'heart_rate': heart_rate,
            'hdl_cholesterol': hdl, 'ldl_cholesterol': ldl, 'glucose_postprandial': glucose_p
        })

        # Mapping binary categorical inputs
        input_dict['family_history_diabetes'] = 1 if fam_hist == "Yes" else 0
        input_dict['hypertension_history'] = 1 if hyp_hist == "Yes" else 0
        input_dict['cardiovascular_history'] = 1 if cardio_hist == "Yes" else 0

        # One-hot encoding for multi-category inputs
        for val, prefix in [(gender, 'gender'), (ethnicity, 'ethnicity'),
                          (employment, 'employment_status'), (smoking, 'smoking_status')]:
            col = f"{prefix}_{val}"
            if col in input_dict:
                input_dict[col] = 1

        # Prepare DataFrame and Scale Data
        data_frame = pd.DataFrame([input
