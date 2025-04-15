import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import LabelEncoder

# ------------------- Load Model and Encoders -------------------
@st.cache_resource

def load_model():
    return joblib.load("fetal_health_model.pkl")

@st.cache_resource

def load_encoders():
    return joblib.load("label_encoders.pkl")

model = load_model()
label_encoders = load_encoders()

# ------------------- Load Dataset -------------------
df = pd.read_csv("updated_fetal_health_dataset.csv")

# ------------------- Feature Setup -------------------
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

binary_columns = [
    "Hypertension", "Diabetes", "Family_History", "Family_History_Congenital_Disorder",
    "Family_History_Diabetes", "Other_Risk_Factors", "Chromosomal_Abnormality",
    "Multiple_Pregnancy", "History_of_Miscarriage", "Gestational_Diabetes",
    "Preeclampsia_Risk", "IVF_Conception"
]

categorical_columns = list(label_encoders.keys())

# ------------------- User Input Sidebar -------------------
def get_user_input():
    st.sidebar.header("👩‍⚕️ Enter Patient Data")
    input_data = {}

    for col in expected_features:
        if col in binary_columns:
            input_data[col] = 1 if st.sidebar.selectbox(col, ["No", "Yes"]) == "Yes" else 0

        elif col in categorical_columns:
            options = sorted(df[col].dropna().unique().tolist())
            default_val = str(df[col].mode()[0])
            selection = st.sidebar.selectbox(col, sorted([str(o) for o in options]), index=0)
            try:
                input_data[col] = label_encoders[col].transform([selection])[0]
            except ValueError:
                input_data[col] = label_encoders[col].transform([default_val])[0]

        elif df[col].dtype in [np.float64, np.int64]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_data[col] = st.sidebar.number_input(col, min_val, max_val, mean_val)

        else:
            input_data[col] = st.sidebar.text_input(col, "")

    return pd.DataFrame([input_data])

# ------------------- Prediction -------------------
user_input_df = get_user_input()
user_input_df = user_input_df[expected_features]

st.subheader("🩺 Prediction Result")
prediction = model.predict(user_input_df)[0]
prediction_proba = model.predict_proba(user_input_df)[0]
health_labels = model.classes_

st.write(f"### 📜 Predicted Fetal Health Status: `{prediction}`")
prob_df = pd.DataFrame({'Health Status': health_labels, 'Probability': prediction_proba})
st.bar_chart(prob_df.set_index('Health Status'))

if prediction == "Normal":
    st.success("✅ Fetus appears **Healthy** based on current data.")
elif prediction == "Suspected":
    st.warning("⚠️ Irregular patterns detected. Further medical tests may be required.")
elif prediction == "Pathological":
    st.error("🚨 High risk of fetal complications. Immediate medical attention is recommended.")

# ------------------- SHAP Interpretation -------------------
st.subheader("🔍 Feature Contribution with SHAP")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_input_df)

shap_array = shap_values[np.argmax(prediction_proba)][0]  # Use class with highest prob
shap_df = pd.DataFrame({
    "Feature": user_input_df.columns,
    "SHAP Value": shap_array
})

shap_df["Abs_Val"] = np.abs(shap_df["SHAP Value"])
shap_df["Contribution %"] = (shap_df["Abs_Val"] / shap_df["Abs_Val"].sum()) * 100
shap_df_sorted = shap_df.sort_values("Abs_Val", ascending=False).head(5)

st.write("### 🔹 Top 5 Feature Contributions:")
st.dataframe(shap_df_sorted[["Feature", "Contribution %"]].round(2), use_container_width=True)

# ------------------- Recommendations -------------------
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
        """
    elif status == "Pathological":
        return """
        🚨 High risk of fetal complications. Immediate medical attention is recommended.
        - Consult your healthcare provider urgently for further evaluation.
        - Consider additional diagnostic tests and follow instructions strictly.
        """
    else:
        return "No specific recommendation available."

st.write("### 📝 Recommendations:")
st.write(get_recommendation(prediction))
