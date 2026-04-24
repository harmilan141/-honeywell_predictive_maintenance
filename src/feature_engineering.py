import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/processed/train_processed.csv"
OUTPUT_PATH = "data/processed/train_features.csv"

df = pd.read_csv(INPUT_PATH)

sensor_cols = [col for col in df.columns if "sensor_" in col]
op_cols = [col for col in df.columns if "op_setting_" in col]

# Rolling features
for sensor in sensor_cols:
    df[f"{sensor}_rolling_mean"] = (
        df.groupby("unit")[sensor]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"{sensor}_rolling_std"] = (
        df.groupby("unit")[sensor]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

# Anomaly detection
feature_cols = sensor_cols + op_cols
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_cols])

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df["anomaly_score"] = iso.fit_predict(scaled_features)

# Convert anomaly output
df["anomaly_flag"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

# Failure risk label
df["failure_risk"] = df["RUL"].apply(lambda x: 1 if x <= 30 else 0)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Feature engineering completed!")
print(f"Saved file: {OUTPUT_PATH}") 