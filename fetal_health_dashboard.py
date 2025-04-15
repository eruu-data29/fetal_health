import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("fetal_health_model.pkl")

model = load_model()

# Load data and extract columns
df = pd.read_csv("updated_fetal_health_dataset.csv")
column_names = df.drop(columns=["Fetal_Health_Status"]).columns.tolist()

# Binary (Yes/No) features
binary_columns = [
    "Hypertension", "Diabetes", "Family_History", "Family_History_Congenital_Disorder",
    "Family_History_Diabetes", "Other_Risk_Factors", "Chromosomal_Abnormality",
    "Multiple_Pregnancy", "History_of_Miscarriage", "Gestational_Diabetes",
    "Preeclampsia_Risk", "IVF_Conception"
]

# Optional: Any other categorical columns (with multiple options)
categorical_columns = []  # Update if you have more categories

# Sidebar input form
def get_user_input():
    st.sidebar.header("👩‍⚕️ Enter Patient Data")
    input_data = {}

    for col in expected_features:
        if col in binary_columns:
            yes_no = st.sidebar.selectbox(col, ["No", "Yes"])
            input_data[col] = 1 if yes_no == "Yes" else 0
        elif df[col].dtype in [np.float64, np.int64]:
            input_data[col] = st.sidebar.number_input(
                label=col,
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean())
            )
        elif col in categorical_columns:
            options = df[col].dropna().unique().tolist()
            input_data[col] = st.sidebar.selectbox(col, options)

    return pd.DataFrame([input_data])

# Page title and description
st.title("🧬 Fetal Health Risk Prediction Dashboard")
st.markdown("This dashboard uses maternal and fetal parameters to predict fetal health status: **Normal**, **Suspected**, or **Pathological**.")

# Get user input
user_input_df = get_user_input()

# Reorder columns to match training-time feature order
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

# Ensure columns are ordered as per the model input during training
user_input_df = user_input_df[expected_features]


# Predict fetal health
st.subheader("🩺 Prediction Result")
prediction = model.predict(user_input_df)[0]
prediction_proba = model.predict_proba(user_input_df)[0]
health_labels = model.classes_

# Display prediction
st.write(f"### 🧾 Predicted Fetal Health Status: `{prediction}`")
st.write("### 📊 Prediction Probabilities:")
prob_df = pd.DataFrame({'Health Status': health_labels, 'Probability': prediction_proba})
st.bar_chart(prob_df.set_index('Health Status'))

# Interpretation message
if prediction == "Normal":
    st.success("✅ Fetus appears **Healthy** based on current data.")
elif prediction == "Suspected":
    st.warning("⚠️ Irregular patterns detected. Further medical tests may be required.")
elif prediction == "Pathological":
    st.error("🚨 High risk of fetal complications. Immediate medical attention is recommended.")

# SHAP-based explanation
st.subheader("🔍 Feature Contribution (SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(user_input_df)

st.set_option('deprecation.showPyplotGlobalUse', False)
shap.plots.bar(shap_values[0], max_display=10)
st.pyplot(bbox_inches='tight')

# Recommendation section
st.subheader("📌 Key Influencing Features")
top_factors = shap_values[0].values.argsort()[::-1][:3]
for i in top_factors:
    feature_name = user_input_df.columns[i]
    feature_value = user_input_df.iloc[0, i]
    st.write(f"- **{feature_name}**: `{feature_value}` may be influencing the risk.")
