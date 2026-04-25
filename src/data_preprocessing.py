import pandas as pd
import numpy as np
import os

# File paths
TRAIN_PATH = "data/raw/train_FD001.txt"
TEST_PATH = "data/raw/test_FD001.txt"
RUL_PATH = "data/raw/RUL_FD001.txt"

# Column names
columns = ['unit', 'cycle'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)
    df.columns = columns
    return df


# 2. 🔥 ADD THIS PART HERE (VERY IMPORTANT)

# Create missing sensors
for i in range(18, 22):
    df[f"sensor_{i}"] = df[f"sensor_{i-1}"] * 1.05

# Create rolling features
window = 3
for i in range(1, 9):
    df[f"sensor_{i}_rolling_mean"] = df[f"sensor_{i}"].rolling(window, min_periods=1).mean()
    df[f"sensor_{i}_rolling_std"] = df[f"sensor_{i}"].rolling(window, min_periods=1).std().fillna(0)

# 3. (Optional) Handle any missing columns
df.fillna(0, inplace=True)

# 4. Now predict ✅
prediction = model.predict(df)


# Load datasets
train_df = load_data(TRAIN_PATH)
test_df = load_data(TEST_PATH)
rul_df = pd.read_csv(RUL_PATH, header=None)
rul_df.columns = ['RUL']

# Create RUL for training data
max_cycle = train_df.groupby('unit')['cycle'].max().reset_index()
max_cycle.columns = ['unit', 'max_cycle']

train_df = train_df.merge(max_cycle, on='unit')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df.drop('max_cycle', axis=1, inplace=True)

# Save processed data
os.makedirs("data/processed", exist_ok=True)

train_df.to_csv("data/processed/train_processed.csv", index=False)
test_df.to_csv("data/processed/test_processed.csv", index=False)
rul_df.to_csv("data/processed/test_rul.csv", index=False)

print("✅ Data preprocessing completed!")