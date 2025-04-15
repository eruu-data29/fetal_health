import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Load model and encoders
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
            input_data[col] = 1 if st.sidebar.selectbox(col, ["No", "Yes"]) == "Yes" else 0
        elif col in categorical_columns:
            options = sorted(df[col].dropna().unique().tolist())
            if all(isinstance(val, (int, float, np.integer, np.floating)) for val in options):
                min_val, max_val = min(options), max(options)
                user_val = st.sidebar.number_input(col, min_value=float(min_val), max_value=float(max_val), value=float(df[col].mode()[0]))
                if user_val not in options:
                    user_val = df[col].mode()[0]
                input_data[col] = label_encoders[col].transform([str(int(user_val))])[0]
            else:
                options = sorted([str(o) for o in options])
                default_val = str(df[col].mode()[0])
                selection = st.sidebar.selectbox(col, options)
                try:
                    input_data[col] = label_encoders[col].transform([selection])[0]
                except ValueError:
                    input_data[col] = label_encoders[col].transform([default_val])[0]
        elif df[col].dtype in [np.float64, np.int64]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_val = st.sidebar.number_input(label=col, min_value=min_val, max_value=max_val, value=mean_val)
            input_data[col] = input_val
        else:
            input_data[col] = st.sidebar.text_input(col, "")

    return pd.DataFrame([input_data])

user_input_df = get_user_input()
user_input_df = user_input_df[expected_features]

# Predict
st.subheader("🩺 Prediction Result")
prediction = model.predict(user_input_df)[0]
prediction_proba = model.predict_proba(user_input_df)[0]
health_labels = model.classes_

st.write(f"### 🧾 Predicted Fetal Health Status: `{prediction}`")
st.write("### 📊 Prediction Probabilities:")
prob_df = pd.DataFrame({'Health Status': health_labels, 'Probability': prediction_proba})
st.bar_chart(prob_df.set_index('Health Status'))

# Explanation
if prediction == "Normal":
    st.success("✅ Fetus appears **Healthy** based on current data.")
elif prediction == "Suspected":
    st.warning("⚠️ Irregular patterns detected. Further medical tests may be required.")
elif prediction == "Pathological":
    st.error("🚨 High risk of fetal complications. Immediate medical attention is recommended.")

# SHAP Explanation - Text only
st.subheader("🔍 Top 5 Feature Contributions")

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_input_df)

# Get the SHAP values for the predicted class
predicted_class_index = list(model.classes_).index(prediction)
shap_vals = shap_values[predicted_class_index][0]  # Single row

# Build a DataFrame of SHAP values
shap_df = pd.DataFrame({
    "Feature": user_input_df.columns,
    "SHAP Value": shap_vals
})

shap_df["Absolute"] = shap_df["SHAP Value"].abs()
shap_df["Percentage Contribution"] = 100 * shap_df["Absolute"] / shap_df["Absolute"].sum()
shap_df_sorted = shap_df.sort_values("Absolute", ascending=False).head(5)

# Display
for i, row in shap_df_sorted.iterrows():
    st.write(f"- **{row['Feature']}**: {row['Percentage Contribution']:.2f}%")

# Recommendations
def get_recommendation(status):
    if status == "Normal":
        return "✅ Fetus appears healthy. Continue regular prenatal care."
    elif status == "Suspected":
        return "⚠️ Irregularities detected. Further medical tests are advised."
    elif status == "Pathological":
        return "🚨 High risk. Immediate medical attention is necessary."
    else:
        return "No recommendation available."

st.write("### 📝 Recommendations:")
st.write(get_recommendation(prediction))
