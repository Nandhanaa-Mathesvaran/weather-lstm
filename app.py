import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

# -------------------- LOAD MODEL & SCALERS --------------------
st.title("🌡️ LSTM Temperature Prediction App")

with st.spinner("Loading model and scalers..."):
    try:
        model = load_model("lstm_model.h5", compile=False)
        scaler_X = joblib.load("scaler_X.joblib")
        scaler_y = joblib.load("scaler_y.joblib")

        with open("features.json", "r") as f:
            features = json.load(f)

        st.success("✅ Model, scalers, and feature list loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model or scalers: {e}")
        st.stop()

# -------------------- FILE UPLOAD --------------------
st.write("Upload a CSV file with the same features used during training to predict **Temperature (T)**.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded data preview:")
        st.dataframe(df.head())

        # -------------------- CHECK FEATURES --------------------
        missing = [col for col in features if col not in df.columns]
        extra = [col for col in df.columns if col not in features]

        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            if extra:
                st.info(f"ℹ️ Ignoring extra columns: {extra}")

            # Keep only required features
            X = df[features].values

            # -------------------- PREDICT BUTTON --------------------
            if st.button("🔮 Predict Temperature"):
                with st.spinner("Predicting temperature... please wait ⏳"):
                    # -------------------- STANDARDIZE --------------------
                    X_scaled = scaler_X.transform(X)

                    # -------------------- CREATE SEQUENCES --------------------
                    lookback = 36  # past 6 hours (used in training)
                    X_seq = []
                    for i in range(lookback, len(X_scaled)):
                        X_seq.append(X_scaled[i - lookback:i])
                    X_seq = np.array(X_seq)

                    if len(X_seq) == 0:
                        st.warning("⚠️ Not enough rows in uploaded file. Need at least 36 rows for prediction.")
                        st.stop()

                    # -------------------- MAKE PREDICTION --------------------
                    y_pred_scaled = model.predict(X_seq)
                    y_pred = scaler_y.inverse_transform(y_pred_scaled)

                    # -------------------- DISPLAY RESULTS --------------------
                    st.subheader("📈 Predicted Temperature (°C):")
                    pred_df = pd.DataFrame({
                        "Predicted_Temperature": y_pred.flatten()
                    })
                    st.dataframe(pred_df)

                    # -------------------- DOWNLOAD OPTION --------------------
                    csv = pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Predicted Temperatures",
                        data=csv,
                        file_name="predicted_temperature.csv",
                        mime="text/csv",
                    )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
