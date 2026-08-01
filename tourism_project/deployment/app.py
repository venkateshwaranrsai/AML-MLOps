import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer selecting a tourism package based on customer history.
Enter the required data below to get a prediction.
""")

# Input fields for Tourism Package Prediction
age = st.number_input("Age", min_value=18, max_value=100, value=35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
number_of_person_visiting = st.number_input("How many Persons Visiting?", min_value=1, max_value=10, value=1)
number_of_followups = st.number_input("How many Follow-ups?", min_value=0, max_value=10, value=3)
number_of_trips = st.number_input("How many Trips (annual)?", min_value=0, max_value=50, value=5)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.number_input("How many Children Visiting?", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=50000.0, step=1000.0)

type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Government"])
gender = st.selectbox("Gender", ["Male", "Female", "Unassigned"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "Standard", "Premium"])
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "President"])

input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "Designation": designation
}])

if st.button("Predict Tourism Package"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase Package" if prediction == 1 else "Will Not Purchase Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
