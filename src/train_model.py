import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "data/processed/train_features.csv"
MODEL_PATH = "models/predictive_maintenance_model.h5"
SCALER_PATH = "models/scaler.pkl"

df = pd.read_csv(DATA_PATH)

drop_cols = ["unit", "cycle", "RUL", "failure_risk"]
X = df.drop(columns=drop_cols)
y = df["failure_risk"]

X = X.select_dtypes(include=["int64", "float64"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

os.makedirs("models", exist_ok=True)

y_prob = model.predict(X_test_scaled)
y_pred = (y_prob > 0.5).astype(int).ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

joblib.dump(metrics, "models/metrics.pkl")

model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Model training completed!")
print("Model saved at:", MODEL_PATH)
print("Scaler saved at:", SCALER_PATH)
