import pandas as pd
import joblib
from tensorflow.keras.models import load_model

MODEL_PATH = "models/predictive_maintenance_model.h5"
SCALER_PATH = "models/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_failure(input_data):
    """
    input_data should be a pandas DataFrame with same feature columns
    used during training.
    """

    input_data = input_data.select_dtypes(include=["int64", "float64"])
    input_scaled = scaler.transform(input_data)

    prediction_prob = model.predict(input_scaled)[0][0]
    prediction_class = 1 if prediction_prob >= 0.5 else 0

    if prediction_prob < 0.4:
        status = "Normal"
    elif prediction_prob < 0.7:
        status = "Warning"
    else:
        status = "Critical"

    return prediction_prob, prediction_class, status