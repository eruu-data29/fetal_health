import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

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
    st.sidebar.header("\U0001F469‍⚕️ Enter Patient Data")
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

st.write(f"### 💾 Predicted Fetal Health Status: `{prediction}`")
st.write("### 📊 Prediction Probabilities:")
prob_df = pd.DataFrame({'Health Status': health_labels, 'Probability': prediction_proba})
st.bar_chart(prob_df.set_index('Health Status'))

# Interpretation
if prediction == "Normal":
    st.success("✅ Fetus appears **Healthy** based on current data.")
elif prediction == "Suspected":
    st.warning("⚠️ Irregular patterns detected. Further medical tests may be required.")
elif prediction == "Pathological":
    st.error("🚨 High risk of fetal complications. Immediate medical attention is recommended.")

# Feature Importances with permutation
st.subheader("🔍 Top Feature Contributions")
perm_result = permutation_importance(model, user_input_df, model.predict(user_input_df), n_repeats=10, random_state=42)
importances = pd.DataFrame({
    'Feature': expected_features,
    'Importance': perm_result.importances_mean
}).sort_values(by='Importance', ascending=False).head(10)

# Normalize to percentages
importances['Contribution (%)'] = 100 * importances['Importance'] / importances['Importance'].sum()
st.dataframe(importances[['Feature', 'Contribution (%)']])

# Recommendations based on important features
st.subheader("📅 Personalized Recommendations")
for row in importances.itertuples():
    feature = row.Feature
    contrib = row._3
    value = user_input_df.iloc[0][feature]
    if feature == "BMI" and value > 30:
        st.write("- Reduce BMI with healthy diet and regular exercise to improve fetal outcomes.")
    elif feature == "Fetal_Heart_Rate" and (value < 110 or value > 160):
        st.write("- Abnormal Fetal Heart Rate. Consider immediate monitoring or follow-up.")
    elif feature == "Gestational_Age" and value < 24:
        st.write("- Early gestation detected. Ensure close prenatal care and monitoring.")
    elif feature == "Preeclampsia_Risk" and value == 1:
        st.write("- High risk of Preeclampsia. Consult with healthcare provider for risk management.")
    elif contrib > 10:
        st.write(f"- `{feature}` is highly influential. Maintain within normal range if possible.")
