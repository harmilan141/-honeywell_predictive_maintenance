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