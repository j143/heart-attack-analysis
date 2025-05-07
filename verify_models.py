import joblib
import numpy as np
import pandas as pd

# Load the heart attack dataset
df = pd.read_csv('Heart_Attack_Analysis_Data.csv')

# Prepare the test data directly from the dataset
X = df.drop('Target', axis=1).values
y = df['Target'].values.reshape(-1, 1)

# Split the data into training and testing sets
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * num_samples)
train_idx, test_idx = indices[:split], indices[split:]
X_test = X[test_idx]
y_test = y[test_idx]

# Scale the test data using the saved scaler
scaler = joblib.load('scaler.pkl')
mean = scaler['mean']
std = scaler['std']
X_test_scaled = (X_test - mean) / std

# Load and verify Logistic Regression weights
try:
    logistic_regression_weights = joblib.load('logistic_regression_weights.pkl')
    print("Logistic Regression weights loaded successfully:")
    print(logistic_regression_weights)
except Exception as e:
    print(f"Error loading Logistic Regression weights: {e}")

# Load and verify L2SVM weights
try:
    l2svm_weights = joblib.load('l2svm_weights.pkl')
    print("\nL2SVM weights loaded successfully:")
    print(l2svm_weights)
except Exception as e:
    print(f"Error loading L2SVM weights: {e}")

# Adjust L2SVM weights to include the bias term
if len(l2svm_weights) == X_test_scaled.shape[1]:
    l2svm_weights = np.append(l2svm_weights, 0)  # Add a bias term if missing

# Example inference using Logistic Regression weights
def logistic_regression_inference(weights, X):
    bias = weights[0]
    coefficients = weights[1:]
    logits = np.dot(X, coefficients) + bias
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)
    return predictions

# Example inference using L2SVM weights
def l2svm_inference(weights, X):
    bias = weights[-1]
    coefficients = weights[:-1]
    decision_values = np.dot(X, coefficients) + bias
    predictions = (decision_values >= 0).astype(int)
    return predictions

# Logistic Regression inference on real test data
try:
    lr_predictions_real = logistic_regression_inference(logistic_regression_weights, X_test_scaled)
    print("\nLogistic Regression predictions on real test data:")
    print(lr_predictions_real)
    print("Accuracy:", np.mean(lr_predictions_real.flatten() == y_test.flatten()))
except Exception as e:
    print(f"Error during Logistic Regression inference on real test data: {e}")

# L2SVM inference on real test data
try:
    l2svm_predictions_real = l2svm_inference(l2svm_weights, X_test_scaled)
    print("\nL2SVM predictions on real test data:")
    print(l2svm_predictions_real)
    print("Accuracy:", np.mean(l2svm_predictions_real.flatten() == y_test.flatten()))
except Exception as e:
    print(f"Error during L2SVM inference on real test data: {e}")