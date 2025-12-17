import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime


# LOAD MODEL & SCALER
model = load_model("model_ann.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Rain Prediction ANN",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Rain Prediction App (ANN)")
st.write("Aplikasi prediksi hujan menggunakan Artificial Neural Network")

st.markdown("---")


# USER INPUT
st.subheader("Input Data Cuaca")

date = st.date_input("Tanggal", datetime.today())

precipitation = st.number_input(
    "Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0
)

temp_max = st.number_input(
    "Max Temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0
)

temp_min = st.number_input(
    "Min Temperature (°C)", min_value=-30.0, max_value=40.0, value=10.0
)

wind = st.number_input(
    "Wind Speed", min_value=0.0, max_value=30.0, value=3.0
)

weather = st.selectbox(
    "Weather Type",
    ["drizzle", "fog", "rain", "snow", "sun"]
)

st.markdown("---")

# PREPROCESS INPUT
weather_mapping = {
    "drizzle": 0,
    "fog": 1,
    "rain": 2,
    "snow": 3,
    "sun": 4
}

weather_encoded = weather_mapping[weather]

year = date.year
month = date.month
day = date.day

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
day_sin = np.sin(2 * np.pi * day / 31)
day_cos = np.cos(2 * np.pi * day / 31)


# FEATURE ORDER (HARUS SAMA)
# Urutan ini HARUS sesuai training:
# ['precipitation', 'temp_max', 'temp_min', 'wind',
#  'weather', 'year', 'month_sin', 'month_cos', 'day_sin', 'day_cos']

input_data = np.array([[
    precipitation,
    temp_max,
    temp_min,
    wind,
    weather_encoded,
    year,
    month_sin,
    month_cos,
    day_sin,
    day_cos
]])

# Scaling
input_scaled = scaler.transform(input_data)

# PREDICTION
if st.button("🔍 Predict Rain"):
    prob = model.predict(input_scaled)[0][0]

    if prob >= 0.5:
        st.error(f"🌧️ **Prediksi: HUJAN**\n\nProbabilitas: **{prob:.2f}**")
    else:
        st.success(f"☀️ **Prediksi: TIDAK HUJAN**\n\nProbabilitas: **{1 - prob:.2f}**")

st.markdown("---")
st.caption("Model: Artificial Neural Network | Dataset: Seattle Weather")
