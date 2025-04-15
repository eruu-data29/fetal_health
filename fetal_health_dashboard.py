import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and label encoders
@st.cache_resource
def load_model():
    return joblib.load("fetal_health_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("label_encoders.pkl")

model = load_model()
label_encoders = load_encoders()

# Load dataset for value ranges
df = pd.read_csv("updated_fetal_health_dataset.csv")

# Expected features for input
expected_features = [
    "Maternal_Age", "BMI", "Hypertension", "Diabetes", "Family_History",
    "Family_History_Congenital_Disorder", "Family_History_Diabetes", "Other_Risk_Factors",
    "Gestational_Age", "Fetal_Heart_Rate", "Head_Circumference", 
    "Abdominal_Circumference", "Femur_Length", "Nuchal_Translucency", "AFP",
    "hCG", "Estriol", "Inhibin_A", "Chromosomal_Abnormality", "Brain_MRI_Risk",
    "Heart_Ultrasound_Risk", "fMCG_Risk", "Echocardiography_Risk", "Smoking_Status",
    "Alcohol_Consumption", "Pre_Pregnancy_Weight", "Weight_Gain_During_Pregnancy", 
    "Medications", "Mental_Health_History", "Gravida", "Parity", "Multiple_Pregnancy",
    "History_of_Miscarriage", "Gestational_Diabetes", "Preeclampsia_Risk", 
    "IVF_Conception", "Pregnancy_Complications", "Fetal_Gender", "Placental_Location", 
    "Calcification_Grade", "Amniotic_Fluid_Index", "Umbilical_Artery_Doppler",
    "Biophysical_Profile_Score", "fMCG_STV", "Fetal_Movement_Counts", "PAPP-A", 
    "TSH", "CRP", "Hemoglobin_Level", "Blood_Pressure_Trend",
    "Non_Invasive_Prenatal_Test_Result", "Karyotyping_Result", 
    "Carrier_Status_Parents", "Environmental_Exposure", "Diet_Quality_Index", 
    "Exercise_Frequency"
]

# Get user input function
def get_user_input():
    st.sidebar.header("👩‍⚕️ Enter Patient Data")
    input_data = {}

    for col in expected_features:
        input_data[col] = st.sidebar.text_input(col, "")

    return pd.DataFrame([input_data])

# Get user input
user_input_df = get_user_input()

# Predict
st.subheader("🩺 Prediction Result")
prediction = model.predict(user_input_df)[0]
prediction_proba = model.predict_proba(user_input_df)[0]
health_labels = model.classes_

# Display predicted health status
st.write(f"### 🧾 Predicted Fetal Health Status: `{prediction}`")
st.write("### 📊 Prediction Probabilities:")
prob_df = pd.DataFrame({'Health Status': health_labels, 'Probability': prediction_proba})
st.bar_chart(prob_df.set_index('Health Status'))

# Recommendation based on health status
def get_recommendation(status):
    if status == "Normal":
        return """
        ✅ Fetus appears **Healthy** based on current data.
        - Continue regular prenatal visits and follow a healthy lifestyle.
        - Ensure balanced nutrition and maintain a stress-free environment.
        """
    elif status == "Suspected":
        return """
        ⚠️ Irregular patterns detected. Further medical tests may be required.
        - Regular monitoring of fetal health is recommended.
        - Consult your healthcare provider for additional screening tests.
        - Follow up on blood pressure, glucose, and other prenatal tests.
        """
    elif status == "Pathological":
        return """
        🚨 High risk of fetal complications. Immediate medical attention is recommended.
        - Consult your healthcare provider urgently for further evaluation.
        - Consider additional diagnostic tests (e.g., ultrasound, fetal monitoring).
        - Follow healthcare provider’s instructions closely for immediate intervention.
        """
    else:
        return "No specific recommendation available."

# Display the recommendation for the predicted health status
recommendation = get_recommendation(prediction)
st.write("### 📝 Recommendations:")
st.write(recommendation)

