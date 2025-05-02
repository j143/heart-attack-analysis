import logging
import pandas as pd
import numpy as np
from systemds.context import SystemDSContext
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict

# Load data with pandas
df = pd.read_csv("Heart_Attack_Analysis_Data.csv")

# Separate features and target
X = df.drop("Target", axis=1).values
y = df["Target"].values.reshape(-1, 1)  # SystemDS expects column vector

# Shuffle and split (80% train, 20% test)
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)

split = int(0.8 * num_samples)
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Optionally, scale features (optional, but recommended)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

with SystemDSContext() as sds:
    # Convert numpy arrays to SystemDS matrices
    X_ds = sds.from_numpy(X_train_scaled)
    y_ds = sds.from_numpy(y_train + 1.0)  # SystemDS expects labels starting from 1

    # Train logistic regression
    bias = multiLogReg(X_ds, y_ds, maxi=100, verbose=False)

    # Prepare test data
    Xt_ds = sds.from_numpy(X_test_scaled)
    yt_ds = sds.from_numpy(y_test + 1.0)

    # Predict and evaluate
    _, y_pred, acc = multiLogRegPredict(Xt_ds, bias, Y=yt_ds, verbose=False).compute()

logging.info(f"SystemDS Logistic Regression Test Accuracy: {acc}")
print(f"SystemDS Logistic Regression Test Accuracy: {acc}")
