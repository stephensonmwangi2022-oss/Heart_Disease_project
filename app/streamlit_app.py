import warnings
warnings.filterwarnings("ignore")
import sys
import os
import pandas as pd
import streamlit as st
import joblib
base_path = os.path.dirname(__file__)

# Set page to wide mode and add a heart icon
icon_path = os.path.join(base_path, "heart_disease_1.jpg")
st.set_page_config(page_title="HeartCare AI", page_icon=icon_path, layout="wide")
# Handle paths so it works on any computer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clean.preprocessing import DataPreprocessor
# --- LOAD MODELS ---
model_path = os.path.join(base_path, "..", "notebooks", "heart_disease_predictor.pkl")
proc_path = os.path.join(base_path, "..", "notebooks", "strategies.pkl")

model = joblib.load(model_path)
preprocessor = joblib.load(proc_path)

# --- SIDEBAR (The Left Hand Corner) ---
with st.sidebar:
    # Use your heart image here
    header_image_path = os.path.join(base_path, "heart_disease.jpg")
    st.image(header_image_path, width='stretch')
    
    st.markdown("## 🏥 Healthcare Awareness")
    st.error("### 📢 CAUTION\n**Heart disease kills.** Early identification is the only way to stay ahead.")
    
    st.info("""
    **Pro Tip:** Visit hospitals regularly. Early screening can prevent 80% of premature heart attacks.
    """)
    st.divider()

# --- MAIN UI ---
st.title("🩺 Heart Disease Diagnostic Assistant")
st.write("Provide patient metrics below to generate a risk profile.")

# Group inputs into cute 'Containers'
with st.container(border=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Basic Info")
        Age = st.slider("Age", 1, 110, 30)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        cp = st.selectbox("Chest Pain Type", ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'])
        fbs = st.toggle("Fasting Blood Sugar > 120 mg/dl")

    with col2:
        st.markdown("#### 📊 Clinical Vitals")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        thalch = st.slider("Max Heart Rate Achieved", 50, 250, 150)
        ca = st.select_slider("Major Vessels Colored (ca)", options=[0, 1, 2, 3, 4])

with st.expander("🔬 Advanced ECG Details"):
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        restecg = st.selectbox("Resting ECG Results", ['normal', 'lv hypertrophy', 'st-t abnormality'])
    with ex_col2:
        slope = st.selectbox("Slope of ST Segment", ['upsloping', 'flat', 'downsloping'])
        thal = st.selectbox("Thallium Stress Test", ['normal', 'fixed defect', 'reversable defect'])
        exang = st.checkbox("Exercise Induced Angina")

# PREDICTION LOGIC 
st.divider()

input_df = pd.DataFrame({
    'age': [Age], 'trestbps': [trestbps], 'chol': [chol], 'thalch': [thalch],
    'oldpeak': [oldpeak], 'ca': [ca], 'sex': [sex], 'cp': [cp],
    'fbs': [fbs], 'restecg': [restecg], 'exang': [exang], 'slope': [slope], 'thal': [thal]
})

if st.button("Analyze Heart Health", type="primary", width='stretch'):
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    # Result Display
    if prediction == 1:
        st.error(f"### ⚠️ High Risk: {prob*100:.1f}% probability")
        st.warning("Immediate consultation with a cardiologist is recommended.")
    else:
        st.success(f"### ✅ Low Risk: {(1-prob)*100:.1f}% confidence")
        st.write("Patient metrics are within a healthy range! Keep up the healthy lifestyle.")