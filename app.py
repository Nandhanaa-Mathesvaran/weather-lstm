import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
import datetime

# -----------------------------
# Load model and scalers
# -----------------------------
model = load_model('lstm_model.h5')
scaler_X = joblib.load('scaler_X.joblib')
scaler_y = joblib.load('scaler_y.joblib')

with open('features.json', 'r') as f:
    features = json.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌦 Weather Temperature Prediction using LSTM")
st.write("Upload your weather data CSV — must include a 'date' column and other features used during training.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'date' not in df.columns:
        st.error("❌ The CSV file must contain a 'date' column.")
    else:
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # 🟢 Keep full original data for displaying after prediction
        df_original = df.copy()

        # Convert 'date' to numeric for model input
        df['date'] = df['date'].map(datetime.datetime.toordinal)

        # Ensure all required features exist
        missing = [f for f in features if f not in df.columns]
        for f in missing:
            df[f] = 0  # fill missing features with 0

        # Reorder columns to match training
        df = df[features]

        # -----------------------------
        # Scale features and predict
        # -----------------------------
        X_scaled = scaler_X.transform(df.values)
        X_scaled = np.expand_dims(X_scaled, axis=1)  # [samples, timesteps, features]

        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # -----------------------------
        # Add predictions to original dataframe
        # -----------------------------
        df_original['Predicted_Temperature'] = y_pred.flatten()

        st.subheader("📊 Prediction Results (All Columns + Predicted Temperature)")
        st.dataframe(df_original)

        # -----------------------------
        # Download full results
        # -----------------------------
        csv = df_original.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Predictions as CSV",
            data=csv,
            file_name="predicted_weather_full.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file containing a 'date' column for prediction.")
