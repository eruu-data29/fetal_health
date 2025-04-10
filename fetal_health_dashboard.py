import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
from io import BytesIO
 
#Set config with custom page title, layout, logo and favicon
st.set_page_config(
    page_title="Fetal Health Risk Predictor",
    layout="wide",
    page_icon="🍼"
)
# Logo (you can upload and use your own logo URL or path)
st.image("https://cdn-icons-png.flaticon.com/512/3943/3943780.png", width=100)
st.title("🤰 Fetal Health Risk Predictor Dashboard")
st.markdown("Enter maternal and fetal health details to assess potential risks:") 
# Load model and label encoder
model = joblib.load("fetal_health_model.pkl")
encoder = joblib.load("label_encoder.pkl")
explainer = shap.Explainer(model)
 
# Form inputs
with st.form("fetal_form"):
    col1, col2 = st.columns(2)
 
    with col1:
        maternal_age = st.slider("Maternal Age", 15, 50, 30)
        bmi = st.slider("BMI", 10.0, 50.0, 22.0)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        family_history = st.selectbox("Family History", ["No", "Yes"])
        family_congenital = st.selectbox("Family Congenital Disorder", ["No", "Yes"])
        gestational_age = st.slider("Gestational Age (weeks)", 10, 42, 20)
        fetal_heart_rate = st.slider("Fetal Heart Rate", 90, 180, 130)
        afp = st.number_input("AFP Level", 0.0, 1000.0, 100.0)
        hcg = st.number_input("hCG Level", 0.0, 1000.0, 150.0)
        estriol = st.number_input("Estriol Level", 0.0, 100.0, 10.0)
        inhibin_a = st.number_input("Inhibin A", 0.0, 1000.0, 200.0)
 
    with col2:
        abdominal_circ = st.slider("Abdominal Circumference (cm)", 10, 40, 30)
        head_circ = st.slider("Head Circumference (cm)", 10, 40, 30)
        femur_len = st.slider("Femur Length (cm)", 1, 10, 4)
        smoking = st.selectbox("Smoking Status", ["No", "Light", "Heavy"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        pre_preg_weight = st.slider("Pre-Pregnancy Weight (kg)", 40, 100, 60)
        weight_gain = st.slider("Weight Gain During Pregnancy (kg)", 0, 30, 10)
        gravida = st.slider("Gravida", 1, 6, 2)
        parity = st.slider("Parity", 0, 5, 1)
        multiple_preg = st.selectbox("Multiple Pregnancy", ["No", "Yes"])
        history_miscarriage = st.selectbox("History of Miscarriage", ["No", "Yes"])
        ivf = st.selectbox("IVF Conception", ["No", "Yes"])
 
    submit = st.form_submit_button("🩺 Predict Fetal Health Status")
 
if submit:
    # Encode inputs
    smoking_map = {"No": 0, "Light": 1, "Heavy": 2}
    binary_map = {"No": 0, "Yes": 1}
 
    input_data = np.array([[
        maternal_age, bmi,
        binary_map[hypertension], binary_map[diabetes], binary_map[family_history],
        binary_map[family_congenital], gestational_age, fetal_heart_rate,
        head_circ, abdominal_circ, femur_len, afp, hcg, estriol, inhibin_a,
        smoking_map[smoking], binary_map[alcohol], pre_preg_weight, weight_gain,
        gravida, parity, binary_map[multiple_preg], binary_map[history_miscarriage],
        binary_map[ivf]
    ]])
 
    df_input = pd.DataFrame(input_data, columns=[
        'Maternal_Age', 'BMI', 'Hypertension', 'Diabetes', 'Family_History',
        'Family_History_Congenital_Disorder', 'Gestational_Age', 'Fetal_Heart_Rate',
        'Head_Circumference', 'Abdominal_Circumference', 'Femur_Length', 'AFP',
        'hCG', 'Estriol', 'Inhibin_A', 'Smoking_Status', 'Alcohol_Consumption',
        'Pre_Pregnancy_Weight', 'Weight_Gain_During_Pregnancy', 'Gravida', 'Parity',
        'Multiple_Pregnancy', 'History_of_Miscarriage', 'IVF_Conception'
    ])
 
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    label = encoder.inverse_transform([prediction])[0]
 
    st.markdown("---")
    st.subheader("🩺 Prediction Result")
    st.success(f"🔍 Predicted Status: **{label}**")
 
    st.write("### 🔢 Prediction Probabilities:")
    proba_dict = {encoder.classes_[i]: f"{proba[i]*100:.2f}%" for i in range(len(proba))}
    st.write(proba_dict)
 
    # SHAP explanation
    shap_values = explainer(df_input)
    shap_df = pd.DataFrame({
        "Feature": df_input.columns,
        "SHAP Impact": shap_values.values[0]
    }).sort_values("SHAP Impact", key=abs, ascending=False)
 
    st.write("### 🔍 Top Risk Factors:")
    st.write(shap_df.head(5))
 
    # Recommendations
    st.markdown("### 💡 Recommendations:")
    if label != "Healthy":
        st.warning("⚠️ Risk detected! Please consult your obstetrician.")
        recommendations = [
            "Schedule a targeted ultrasound.",
            "Monitor hormone levels regularly.",
            "Maintain a healthy diet and exercise.",
            "Limit stress and avoid harmful exposure."
        ]
    else:
        st.success("✅ Everything looks normal. Continue regular prenatal checkups.")
        recommendations = ["Keep following routine prenatal care."]
 
    for r in recommendations:
        st.markdown(f"- {r}")
 
     # 📄 Downloadable Report
    st.markdown("### 📄 Download Report")
    report = f"""Fetal Health Prediction Report
 
Prediction: {label}
 
Prediction Probabilities:
{proba_dict}
 
Top Risk Factors:
{shap_df.head(5).to_string(index=False)}
 
Recommendations:
{chr(10).join(recommendations)}
 
    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)
 
    st.download_button(
        label="📥 Download Report",
        data=buffer,
        file_name="fetal_health_report.txt",
        mime="text/plain"
    )
