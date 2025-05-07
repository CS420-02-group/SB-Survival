# linear_regression.py
# Train and visualize OLS linear regression model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_data

# Core functions
def ordinary_least_squares(X, y):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    beta = np.linalg.pinv(X) @ y
    return beta

def predict(X, beta):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X @ beta

# Load preprocessed data
X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
if X_train is None:
    print("Failed to load data")
    exit()

# Convert to float arrays for linear algebra
X_train_np = X_train.values.astype(float)
y_train_np = y_train.values.astype(float)
X_test_np = X_test.values.astype(float)
y_test_np = y_test.values.astype(float)

# Train model
beta = ordinary_least_squares(X_train_np, y_train_np)

# Predict
y_pred = predict(X_test_np, beta)

# Create output directory
os.makedirs("outputs/plots", exist_ok=True)

# Plot predicted vs actual
plt.figure()
plt.scatter(y_test_np, y_pred, alpha=0.5)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predicted vs Actual")
plt.grid(True)
plt.savefig("outputs/plots/linear_regression_pred_vs_actual.png")
plt.close()
print("✅ Saved: outputs/plots/linear_regression_pred_vs_actual.png")

# Plot residuals
residuals = y_test_np - y_pred
plt.figure()
plt.hist(residuals, bins=30, edgecolor="black")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Linear Regression Residuals")
plt.grid(True)
plt.savefig("outputs/plots/linear_regression_residuals.png")
plt.close()
print("✅ Saved: outputs/plots/linear_regression_residuals.png")
