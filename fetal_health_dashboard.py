import streamlit as st
import pandas as pd
import numpy as np
import shap
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

# Load the original dataset to get value ranges
df = pd.read_csv("updated_fetal_health_dataset.csv")

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

def get_user_input():
    st.sidebar.header("👩‍⚕️ Enter Patient Data")
    input_data = {}

    for col in expected_features:
        if col in binary_columns:
            # Handle binary (Yes/No) columns
            input_data[col] = 1 if st.sidebar.selectbox(col, ["No", "Yes"]) == "Yes" else 0
        elif col in categorical_columns:
            # Handle categorical columns
            options = df[col].dropna().unique().tolist()
            selection = st.sidebar.selectbox(col, options)
            
            try:
                # Use LabelEncoder to transform categorical values
                input_data[col] = label_encoders[col].transform([selection])[0]
            except ValueError:
                # Handle unseen labels (labels not seen during training)
                st.warning(f"⚠️ '{selection}' not recognized for {col}. Using default.")
                # Default to the most frequent value (mode) in the dataset
                default_val = df[col].mode()[0]
                input_data[col] = label_encoders[col].transform([default_val])[0]
        elif df[col].dtype in [np.float64, np.int64]:
            # Handle numerical columns
            input_data[col] = st.sidebar.number_input(
                label=col,
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean())
            )
        else:
            # Handle text input (for columns that don't fit into the previous categories)
            input_data[col] = st.sidebar.text_input(col, "")

    return pd.DataFrame([input_data])


# Get input
user_input_df = get_user_input()
user_input_df = user_input_df[expected_features]  # Ensure column order

# Predict
st.subheader("🩺 Prediction Result")
prediction = model.predict(user_input_df)[0]
prediction_proba = model.predict_proba(user_input_df)[0]
health_labels = model.classes_

st.write(f"### 🧾 Predicted Fetal Health Status: `{prediction}`")
st.write("### 📊 Prediction Probabilities:")
prob_df = pd.DataFrame({'Health Status': health_labels, 'Probability': prediction_proba})
st.bar_chart(prob_df.set_index('Health Status'))

if prediction == "Normal":
    st.success("✅ Fetus appears **Healthy** based on current data.")
elif prediction == "Suspected":
    st.warning("⚠️ Irregular patterns detected. Further medical tests may be required.")
elif prediction == "Pathological":
    st.error("🚨 High risk of fetal complications. Immediate medical attention is recommended.")

# SHAP Explanation
st.subheader("🔍 Feature Contribution (SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(user_input_df)

st.set_option('deprecation.showPyplotGlobalUse', False)
shap.plots.bar(shap_values[0], max_display=10)
st.pyplot(bbox_inches='tight')

# Top Factors
st.subheader("📌 Key Influencing Features")
top_factors = shap_values[0].values.argsort()[::-1][:3]
for i in top_factors:
    feature_name = user_input_df.columns[i]
    feature_value = user_input_df.iloc[0, i]
    st.write(f"- **{feature_name}**: `{feature_value}` may be influencing the risk.")
