# logistic_regression.py
# Author: Adrian Caballero

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import os


# Load the pre-processed dataset
df = pd.read_csv("../processed/aggregated_bds_data.csv")

# Binary label: survival_rate > 0.5 means more births than deaths → label = 1
df["label"] = (df["survival_rate"] > 0.5).astype(int)

# Drop any missing values
df = df.dropna()

# One-hot encode industry and period
df = pd.get_dummies(df, columns=["industry_name", "period"], drop_first=True)

# Feature columns
feature_cols = ["year", "net_jobs", "survival_rate"] + [col for col in df.columns if col.startswith("industry_name_") or col.startswith("period_")]
X = df[feature_cols].astype(float).values
y = df["label"].values

# Normalize numerical columns
year_idx = feature_cols.index("year")
net_jobs_idx = feature_cols.index("net_jobs")
survival_rate_idx = feature_cols.index("survival_rate")
X[:, [year_idx, net_jobs_idx, survival_rate_idx]] = (X[:, [year_idx, net_jobs_idx, survival_rate_idx]] - X[:, [year_idx, net_jobs_idx, survival_rate_idx]].mean(axis=0)) / X[:, [year_idx, net_jobs_idx, survival_rate_idx]].std(axis=0)

# Add bias column
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights
weights = np.zeros(X_train.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training with gradient descent
lr = 0.1
epochs = 1000
for _ in range(epochs):
    z = np.dot(X_train, weights)
    preds = sigmoid(z)
    error = y_train - preds
    gradient = np.dot(X_train.T, error) / len(y_train)
    weights += lr * gradient

output_dir = os.path.join("..", "outputs", "plots")
os.makedirs(output_dir, exist_ok=True)

# Evaluate on test set
z_test = np.dot(X_test, weights)
test_probs = sigmoid(z_test)
test_preds = (test_probs >= 0.5).astype(int)
accuracy = np.mean(test_preds == y_test)
print("Test Accuracy:", accuracy)

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, test_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plot_path = os.path.join(output_dir, "roc_curve.png")
plt.savefig(plot_path)
print("ROC curve saved to:", os.path.abspath(plot_path))
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
# Print confusion matrix
cm = confusion_matrix(y_test, test_preds)
print("\nConfusion Matrix:")
print(cm)

# Classification report (precision, recall, F1, support)
print("\nClassification Report:")
print(classification_report(y_test, test_preds, digits=3))

# Also print AUC score
print(f"\nAUC Score: {roc_auc:.4f}")