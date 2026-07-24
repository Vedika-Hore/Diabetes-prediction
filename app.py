import streamlit as st
import joblib
import pandas as pd

model=joblib.load("diabetes_model.pkl")
scaler=joblib.load("scaler (3).pkl")
column=joblib.load("columns.pkl")

st.set_page_config(page_title="Diabetes Prediction System"
                   , layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's health details below to predict whether they have diabetes.")


# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)

glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)

blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)

skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)

bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.50,
    format="%.2f"
)

age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict Button
if st.button("Predict"):

    # Create DataFrame
    input_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetespedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    input_df = input_df[column]

    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_scaled)

    st.write("Prediction:", prediction[0])

    # Display result
    if prediction[0] == 0:
        st.success("✅ Prediction: Not Diabetic")
    else:
        st.error("⚠️ Prediction: Diabetic")