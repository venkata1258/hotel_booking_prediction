import streamlit as st
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

# ===== Page Layout =====
st.set_page_config(page_title="Hotel Booking Prediction", page_icon="🛎️", layout="centered")

# ===== Stylish Header =====
st.markdown("""
<div style="background-color:#1E3A8A; padding:20px; border-radius:15px; text-align:center;">
    <h1 style="color:#FBBF24;">🛎️ Hotel Reservation Booking Status Prediction</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Enter Booking Details", unsafe_allow_html=True)

# ===== Styled Input Function =====
def input_card(label, widget):
    st.markdown(f"""
        <div style='background-color:#f3f4f6; padding:10px; border-radius:10px; margin-bottom:10px;'>
            <b>{label}</b>
        </div>
    """, unsafe_allow_html=True)
    return widget

# ===== Inputs with unique keys =====
no_of_adults = input_card("Number of adults", st.number_input("", 0, 10, 2, key="adults"))
no_of_children = input_card("Number of children", st.number_input("", 0, 10, 0, key="children"))
no_of_weekend_nights = input_card("Weekend nights", st.number_input("", 0, 20, 1, key="weekend_nights"))
no_of_week_nights = input_card("Week nights", st.number_input("", 0, 30, 2, key="week_nights"))
type_of_meal_plan = input_card("Meal Plan", st.selectbox("", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"], key="meal_plan"))
required_car_parking_space = input_card("Car parking spaces required", st.number_input("", 0, 5, 0, key="parking"))
room_type_reserved = input_card("Room Type", st.selectbox("", ["Room_Type 1","Room_Type 2","Room_Type 3","Room_Type 4","Room_Type 5","Room_Type 6","Room_Type 7"], key="room_type"))
lead_time = input_card("Lead time (days)", st.number_input("", 0, 365, 30, key="lead_time"))
arrival_year = input_card("Arrival year", st.number_input("", 2015, 2030, 2018, key="arrival_year"))
arrival_month = input_card("Arrival month", st.number_input("", 1, 12, 7, key="arrival_month"))
arrival_date = input_card("Arrival date", st.number_input("", 1, 31, 1, key="arrival_date"))
market_segment_type = input_card("Market Segment", st.selectbox("", ["Online","Offline","Corporate","Complementary","Aviation"], key="market_segment"))
repeated_guest = input_card("Repeated Guest?", st.selectbox("", [0,1], key="repeated_guest"))
no_of_previous_cancellations = input_card("Previous cancellations", st.number_input("", 0, 50, 0, key="prev_cancellations"))
no_of_previous_bookings_not_canceled = input_card("Previous non-canceled bookings", st.number_input("", 0, 100, 0, key="prev_not_canceled"))
avg_price_per_room = input_card("Average price per room", st.number_input("", 0.0, 100000.0, 100.0, key="avg_price"))
no_of_special_requests = input_card("Special requests", st.number_input("", 0, 10, 0, key="special_requests"))

# ===== Build input DataFrame =====
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

# ===== Prediction Button Styling =====
button_html = """
    <style>
    .stButton>button {
        background-color:#FBBF24;
        color:black;
        height:50px;
        width:100%;
        border-radius:10px;
        font-size:18px;
        font-weight:bold;
    }
    .stButton>button:hover {
        background-color:#D97706;
        color:white;
    }
    </style>
"""
st.markdown(button_html, unsafe_allow_html=True)

# ===== Prediction =====
if st.button("Predict Booking Status"):
    with st.spinner("Predicting... 🔄"):
        X_transformed = preprocessor.transform(input_data)
        prediction = model.predict(X_transformed)[0][0]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Probability bar
    st.markdown(f"""
        <div style='background-color:#e5e7eb; border-radius:10px; padding:10px;'>
            <div style='width:{prediction*100}%; background-color:#f59e0b; padding:10px; border-radius:10px; text-align:center; color:white; font-weight:bold;'>
                Cancellation Probability: {prediction:.2f}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Result Box
    if prediction >= 0.5:
        st.markdown('<div style="background-color:#EF4444; padding:15px; border-radius:10px; text-align:center; color:white; font-size:20px;">❌ Booking likely to be CANCELED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color:#10B981; padding:15px; border-radius:10px; text-align:center; color:white; font-size:20px;">✅ Booking likely to be NOT CANCELED</div>', unsafe_allow_html=True)
