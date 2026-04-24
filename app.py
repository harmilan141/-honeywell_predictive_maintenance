import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="SmartPredict Maintenance",
    page_icon="⚙️",
    layout="wide"
)

MODEL_PATH = "models/predictive_maintenance_model.h5"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/processed/train_features.csv"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("⚙️ SmartPredict: Predictive Maintenance System")
st.write("Neural Network-based failure prediction for smart factory equipment.")

df = pd.read_csv(DATA_PATH)

drop_cols = ["unit", "cycle", "RUL", "failure_risk"]
feature_df = df.drop(columns=drop_cols)
feature_df = feature_df.select_dtypes(include=["int64", "float64"])

st.sidebar.header("Input Options")

mode = st.sidebar.radio(
    "Choose prediction mode",
    ["Random equipment sample", "Upload CSV"]
)

if mode == "Random equipment sample":
    sample = feature_df.sample(1, random_state=None)
    st.subheader("Selected Sensor Data")
    st.dataframe(sample)

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        drop_cols = ["unit", "cycle", "RUL", "failure_risk"]

        for col in drop_cols:
            if col in uploaded_df.columns:
                uploaded_df = uploaded_df.drop(columns=[col])

        sample = uploaded_df.select_dtypes(include=["int64", "float64"])

        missing_cols = [col for col in feature_df.columns if col not in sample.columns]

        if missing_cols:
            st.error("Uploaded CSV is missing required feature-engineered columns.")
            st.write("Missing columns:", missing_cols[:20])
            st.stop()

        sample = sample[feature_df.columns]


        st.subheader("Uploaded Sensor Data")
        st.dataframe(sample.head(10))
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

sample_scaled = scaler.transform(sample)
prediction_prob = model.predict(sample_scaled)[0][0]

if prediction_prob < 0.4:
    status = "Normal"
elif prediction_prob < 0.7:
    status = "Warning"
else:
    status = "Critical"

col1, col2, col3 = st.columns(3)

col1.metric("Failure Probability", f"{prediction_prob:.2%}")
col2.metric("Machine Status", status)
col3.metric("Maintenance Priority", "High" if status == "Critical" else "Medium" if status == "Warning" else "Low")

st.subheader("Decision Support")

if status == "Normal":
    st.success("Machine is operating normally. No immediate maintenance required.")
elif status == "Warning":
    st.warning("Early signs of degradation detected. Schedule inspection soon.")
else:
    st.error("High failure risk detected. Immediate maintenance recommended.")

st.subheader("Top Sensor Readings")
st.bar_chart(sample.iloc[0].head(20))