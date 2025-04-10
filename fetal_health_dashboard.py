import streamlit as st
import numpy as np
import joblib
import pandas as pd
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Fetal Health Risk Predictor",
    layout="wide",
    page_icon="🍼"
)

# Logo and title
st.image("https://cdn-icons-png.flaticon.com/512/3943/3943780.png", width=100)
st.title("🤰 Fetal Health Risk Predictor Dashboard")
st.markdown("Select maternal and fetal health parameters from dropdown menus:")

# Load model and label encoder
try:
    model = joblib.load("fetal_health_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    # Load median values saved during training
    training_medians = {
        'AFP': 100.0,  # Replace with your actual medians
        'hCG': 150.0,
        'Estriol': 10.0,
        'Inhibin_A': 200.0,
        # Add medians for all other missing features
    }
except Exception as e:
    st.error(f"Failed to load model files: {str(e)}")
    st.stop()

# Define label mapping
LABEL_MAPPING = {0: "Healthy", 1: "Moderate Risk", 2: "High Risk"}

# Define dropdown options
AGE_OPTIONS = [str(x) for x in range(15, 51)]
BMI_OPTIONS = [f"{x:.1f}" for x in np.arange(15.0, 50.1, 0.5)]
YES_NO = ["No", "Yes"]
SMOKING_OPTIONS = ["No", "Light", "Heavy"]
GESTATIONAL_AGE = [str(x) for x in range(10, 43)]
HEART_RATE = [str(x) for x in range(90, 181)]
CIRCUMFERENCE = [str(x) for x in range(10, 41)]
FEMUR_LENGTH = [f"{x:.1f}" for x in np.arange(1.0, 10.1, 0.5)]
WEIGHT_OPTIONS = [str(x) for x in range(40, 101)]
WEIGHT_GAIN = [str(x) for x in range(0, 31)]
GRAVIDA = ["1", "2", "3", "4", "5", "6"]
PARITY = ["0", "1", "2", "3", "4", "5"]

# Form inputs with dropdowns only
with st.form("fetal_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        maternal_age = st.selectbox("Maternal Age (years)", AGE_OPTIONS, index=15)
        bmi = st.selectbox("BMI", BMI_OPTIONS, index=14)
        hypertension = st.selectbox("Hypertension", YES_NO)
        diabetes = st.selectbox("Diabetes", YES_NO)
        family_history = st.selectbox("Family History", YES_NO)
        family_congenital = st.selectbox("Family Congenital Disorder", YES_NO)
        gestational_age = st.selectbox("Gestational Age (weeks)", GESTATIONAL_AGE, index=10)
        fetal_heart_rate = st.selectbox("Fetal Heart Rate (bpm)", HEART_RATE, index=40)
        head_circ = st.selectbox("Head Circumference (cm)", CIRCUMFERENCE, index=20)
        abdominal_circ = st.selectbox("Abdominal Circumference (cm)", CIRCUMFERENCE, index=20)
        
    with col2:
        femur_len = st.selectbox("Femur Length (cm)", FEMUR_LENGTH, index=6)
        smoking = st.selectbox("Smoking Status", SMOKING_OPTIONS)
        alcohol = st.selectbox("Alcohol Consumption", YES_NO)
        pre_preg_weight = st.selectbox("Pre-Pregnancy Weight (kg)", WEIGHT_OPTIONS, index=20)
        weight_gain = st.selectbox("Weight Gain During Pregnancy (kg)", WEIGHT_GAIN, index=10)
        gravida = st.selectbox("Gravida", GRAVIDA, index=1)
        parity = st.selectbox("Parity", PARITY, index=1)
        multiple_preg = st.selectbox("Multiple Pregnancy", YES_NO)
        history_miscarriage = st.selectbox("History of Miscarriage", YES_NO)
        ivf = st.selectbox("IVF Conception", YES_NO)
    
    submit = st.form_submit_button("🩺 Predict Fetal Health Status")

if submit:
    # Convert all inputs to numeric
    input_values = [
        float(maternal_age), float(bmi),
        1 if hypertension == "Yes" else 0,
        1 if diabetes == "Yes" else 0,
        1 if family_history == "Yes" else 0,
        1 if family_congenital == "Yes" else 0,
        float(gestational_age), float(fetal_heart_rate),
        float(head_circ), float(abdominal_circ), float(femur_len),
        0 if smoking == "No" else (1 if smoking == "Light" else 2),
        1 if alcohol == "Yes" else 0,
        float(pre_preg_weight), float(weight_gain),
        float(gravida), float(parity),
        1 if multiple_preg == "Yes" else 0,
        1 if history_miscarriage == "Yes" else 0,
        1 if ivf == "Yes" else 0
    ]
    
    # Create DataFrame with collected features
    collected_features = [
        'Maternal_Age', 'BMI', 'Hypertension', 'Diabetes', 'Family_History',
        'Family_History_Congenital_Disorder', 'Gestational_Age', 'Fetal_Heart_Rate',
        'Head_Circumference', 'Abdominal_Circumference', 'Femur_Length',
        'Smoking_Status', 'Alcohol_Consumption', 'Pre_Pregnancy_Weight',
        'Weight_Gain_During_Pregnancy', 'Gravida', 'Parity', 'Multiple_Pregnancy',
        'History_of_Miscarriage', 'IVF_Conception'
    ]
    df_input = pd.DataFrame([input_values], columns=collected_features)
    
    # Add missing features with median values
    for feature in model.feature_names_in_:
        if feature not in df_input.columns:
            df_input[feature] = training_medians.get(feature, 0)
    
    # Reorder columns to match model
    df_input = df_input[model.feature_names_in_]
    
    # Debug: Show some key inputs
    with st.expander("Debug: Input Values"):
        st.write("Sample inputs being sent to model:")
        st.dataframe(df_input.iloc[0:1, :5])  # Show first 5 features
        
        # Test predictions with extreme values
        test_df = df_input.copy()
        test_df['Fetal_Heart_Rate'] = 120  # Normal
        test_df['BMI'] = 22.0  # Healthy
        st.write("Test prediction (normal inputs):", model.predict(test_df)[0])
        
        test_df['Fetal_Heart_Rate'] = 50  # Abnormal
        test_df['BMI'] = 40.0  # High risk
        st.write("Test prediction (abnormal inputs):", model.predict(test_df)[0])
    
    # Make prediction
    try:
        prediction = model.predict(df_input)[0]
        label = LABEL_MAPPING.get(prediction, f"Unknown ({prediction})")
        
        # Display results
        st.markdown("---")
        
        if prediction == 0:
            st.success(f"## ✅ Healthy")
        elif prediction == 1:
            st.warning(f"## ⚠️ Moderate Risk")
        else:
            st.error(f"## ❗ High Risk")
        
        # Feature importance as percentages
        importance = model.feature_importances_
        total = importance.sum()
        top_features = pd.DataFrame({
            'Factor': model.feature_names_in_,
            'Contribution (%)': (importance / total * 1000).round(1)
        }).sort_values('Contribution (%)', ascending=False).head(5)
        
        st.write("### Top Contributing Factors:")
        st.dataframe(top_features.style.format({'Contribution (%)': '{:.1f}%'}))
        
        # Recommendations
        st.write("### Recommendations:")
        if prediction == 0:
            st.success("""
            - Continue routine prenatal care
            - Maintain balanced nutrition
            - Regular light exercise
            """)
        elif prediction == 1:
            st.warning("""
            - Schedule additional monitoring
            - Consult with specialist
            - Consider diagnostic tests
            """)
        else:
            st.error("""
            - Seek immediate medical attention
            - Hospitalization may be required
            - Intensive monitoring needed
            """)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
