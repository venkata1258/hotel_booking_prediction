import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

# ===== Cache loading =====
@st.cache_resource
def load_artifacts():
    model = load_model("hotel_cancellation_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

st.title("🏨 Hotel Reservation Booking Status Prediction")
#st.write("Deep Learning Model using Artificial Neural Network (ANN)")

st.subheader("Enter Booking Details")

# ===== Inputs =====
no_of_adults = st.number_input("Number of adults", 0, 10, 2)
no_of_children = st.number_input("Number of children", 0, 10, 0)
no_of_weekend_nights = st.number_input("Weekend nights", 0, 20, 1)
no_of_week_nights = st.number_input("Week nights", 0, 30, 2)

type_of_meal_plan = st.selectbox(
    "Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
)

required_car_parking_space = st.number_input(
    "Car parking spaces required", 0, 5, 0
)

room_type_reserved = st.selectbox(
    "Room Type",
    ["Room_Type 1", "Room_Type 2", "Room_Type 3",
     "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]
)

lead_time = st.number_input("Lead time (days)", 0, 365, 30)
arrival_year = st.number_input("Arrival year", 2015, 2030, 2018)
arrival_month = st.number_input("Arrival month", 1, 12, 7)
arrival_date = st.number_input("Arrival date", 1, 31, 1)

market_segment_type = st.selectbox(
    "Market Segment",
    ["Online", "Offline", "Corporate", "Complementary", "Aviation"]
)

repeated_guest = st.selectbox("Repeated Guest?", [0, 1])

no_of_previous_cancellations = st.number_input(
    "Previous cancellations", 0, 50, 0
)
no_of_previous_bookings_not_canceled = st.number_input(
    "Previous non-canceled bookings", 0, 100, 0
)

avg_price_per_room = st.number_input(
    "Average price per room", 0.0, 100000.0, 100.0
)

no_of_special_requests = st.number_input(
    "Special requests", 0, 10, 0
)

# ===== Input DataFrame =====
input_data = pd.DataFrame([{
    "no_of_adults": no_of_adults,
    "no_of_children": no_of_children,
    "no_of_weekend_nights": no_of_weekend_nights,
    "no_of_week_nights": no_of_week_nights,
    "type_of_meal_plan": type_of_meal_plan,
    "required_car_parking_space": required_car_parking_space,
    "room_type_reserved": room_type_reserved,
    "lead_time": lead_time,
    "arrival_year": arrival_year,
    "arrival_month": arrival_month,
    "arrival_date": arrival_date,
    "market_segment_type": market_segment_type,
    "repeated_guest": repeated_guest,
    "no_of_previous_cancellations": no_of_previous_cancellations,
    "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
    "avg_price_per_room": avg_price_per_room,
    "no_of_special_requests": no_of_special_requests
}])

# ===== Prediction =====
if st.button("Predict Booking Status"):
    with st.spinner("Predicting..."):
        X_transformed = preprocessor.transform(input_data)
        prediction = model.predict(X_transformed)[0][0]

    st.write(f"**Cancellation Probability:** {prediction:.4f}")

    if prediction >= 0.5:
        st.error("❌ Booking likely to be CANCELED")
    else:
        st.success("✅ Booking likely to be NOT CANCELED")