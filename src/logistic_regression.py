# logistic_regression.py
# Author: Adrian Caballero

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../processed/filtered_bds_data.csv")

# Binary label: 1 for births, 0 for deaths
df["label"] = df["dataclass_name"].apply(lambda x: 1 if x == "Establishment Births" else 0)

#Drop rows with missing things
df = df.replace("-", np.nan)
df = df.dropna(subset=["year", "value", "industry_name", "sizeclass_name"])

# Convert numeric fields
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Drop remaining NaNs
df = df.dropna(subset=["year", "value"])

# One-hot encode industry and size class
df = pd.get_dummies(df, columns=["industry_name", "sizeclass_name"], drop_first=True)

# Select features
feature_cols = ["year", "value"] + [col for col in df.columns if col.startswith("industry_name_") or col.startswith("sizeclass_name_")]
X = df[feature_cols].values.astype(float)
y = df["label"].values

# Normalize year and value
year_idx = feature_cols.index("year")
value_idx = feature_cols.index("value")
X[:, [year_idx, value_idx]] = (X[:, [year_idx, value_idx]] - X[:, [year_idx, value_idx]].mean(axis=0)) / X[:, [year_idx, value_idx]].std(axis=0)

# Add bias
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Initialize weights
weights = np.zeros(X.shape[1])

# Sigmoid
def sigmoid(z):
    z = np.array(z, dtype=float)  # ensure z is a NumPy array
    return 1 / (1 + np.exp(-z))

# Train
lr = 0.1
epochs = 1000
for _ in range(epochs):
    z = np.dot(X, weights)
    preds = sigmoid(z)
    error = y - preds
    gradient = np.dot(X.T, error) / len(y)
    weights += lr * gradient

# Results
print("Trained weights:", weights)
pred_probs = sigmoid(np.dot(X, weights))
pred_labels = (pred_probs >= 0.5).astype(int)
accuracy = np.mean(pred_labels == y)
print("Accuracy:", accuracy)

# Visualization (year only)
plt.scatter(X[:, 1], y, label="True", alpha=0.5)
plt.scatter(X[:, 1], pred_probs, label="Predicted", alpha=0.5)
plt.xlabel("Normalized Year")
plt.ylabel("Probability of Survival")
plt.title("Logistic Regression with Feature Engineering")
plt.legend()
plt.grid(True)
plt.show()
