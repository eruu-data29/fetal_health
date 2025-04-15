import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
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
                user_val = st.sidebar.number_input(
                    f"{col} (allowed: {int(min_val)}–{int(max_val)})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(df[col].mode()[0])
                )

                if user_val not in options:
                    st.warning(f"⚠️ `{int(user_val)}` is out of range for `{col}`. Using default.")
                    user_val = df[col].mode()[0]

                input_data[col] = label_encoders[col].transform([str(int(user_val))])[0]

            else:
                options = sorted([str(o) for o in options])
                default_val = str(df[col].mode()[0])
                selection = st.sidebar.selectbox(col, options)

                try:
                    input_data[col] = label_encoders[col].transform([selection])[0]
                except ValueError:
                    st.warning(f"⚠️ '{selection}' not recognized for {col}. Using default.")
                    input_data[col] = label_encoders[col].transform([default_val])[0]

        elif df[col].dtype in [np.float64, np.int64]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_val = st.sidebar.number_input(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
            input_data[col] = input_val

        else:
            input_data[col] = st.sidebar.text_input(col, "")

    return pd.DataFrame([input_data])

# Get user input
user_input_df = get_user_input()
user_input_df = user_input_df[expected_features]  # Enforce order

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

# SHAP Feature Contribution Table
st.subheader("📌 Top Feature Contributions (SHAP Percentages)")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input_df)

    pred_class_index = list(model.classes_).index(prediction)
    shap_values_for_class = shap_values[pred_class_index]

    abs_shap_vals = np.abs(shap_values_for_class[0])
    total = np.sum(abs_shap_vals)

    if total == 0 or np.isnan(total):
        st.warning("⚠️ SHAP could not compute meaningful feature contributions for this input.")
    else:
        percent_contributions = (abs_shap_vals / total) * 100
        valid_indices = [i for i in np.argsort(percent_contributions)[::-1] if np.isfinite(percent_contributions[i])]
        N = min(5, len(valid_indices))
        top_indices = valid_indices[:N]

        top_features = [user_input_df.columns[i] for i in top_indices]
        top_values = [user_input_df.iloc[0, i] for i in top_indices]
        top_percentages = [percent_contributions[i] for i in top_indices]

        if len(top_features) == len(top_values) == len(top_percentages):
            shap_df = pd.DataFrame({
                "Feature": top_features,
                "Value": top_values,
                "Contribution (%)": top_percentages
            }).round(2)

            st.dataframe(shap_df.style.format({"Contribution (%)": "{:.2f}"}), use_container_width=True)
        else:
            st.error("❌ SHAP mismatch: Unable to align features and values for explanation.")

except Exception as e:
    st.error(f"❌ SHAP Error: {str(e)}")

# Recommendations
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

recommendation = get_recommendation(prediction)
st.write("### 📝 Recommendations:")
st.write(recommendation)
