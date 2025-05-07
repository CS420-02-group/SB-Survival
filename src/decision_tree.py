# decision_tree.py
# implementation of decision tree for predicting business survival

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Set output folder and ensure it exists
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed dataset
try:
    df = pd.read_csv("processed/filtered_bds_data.csv")
    print("using filtered bds data")
except FileNotFoundError:
    try:
        df = pd.read_csv("processed/aggregated_bds_data.csv")
        print("using aggregated bds data")
    except FileNotFoundError:
        print("error: no processed data found. run combine_bds_data.py first.")
        exit(1)

# Prepare features and labels
if "dataclass_name" in df.columns:
    df["label"] = df["dataclass_name"].apply(lambda x: 1 if x == "Establishment Births" else 0)
    df = df.replace("-", np.nan)
    df = df.dropna(subset=["year", "value", "industry_name", "sizeclass_name"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"])
    df = pd.get_dummies(df, columns=["industry_name", "sizeclass_name"])
    feature_cols = ["year", "value"] + [col for col in df.columns if col.startswith("industry_name_") or col.startswith("sizeclass_name_")]
    X = df[feature_cols]
    y = df["label"]
else:
    df["label"] = (df["births"] > df["deaths"]).astype(int)
    df = pd.get_dummies(df, columns=["industry_name"])
    feature_cols = ["year"] + [col for col in df.columns if col.startswith("industry_name_")]
    X = df[feature_cols]
    y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluate
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"decision tree accuracy: {accuracy:.4f}\n")

print("classification report:")
print(classification_report(y_test, y_pred))

print("\nconfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nfeature importance:")
print(feature_importance.head(10))

# Save decision tree visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['death', 'birth'],
          filled=True, rounded=True, max_depth=3)
plt.title("Decision Tree for Business Survival Prediction")
tree_path = os.path.join(OUTPUT_DIR, "decision_tree_visualization.png")
plt.savefig(tree_path)
plt.close()
print(f"\ndecision tree visualization saved to: {tree_path}")

# Save ROC curve
y_prob = dt_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "decision_tree_roc.png")
plt.savefig(roc_path)
plt.close()
print(f"decision tree ROC curve saved to: {roc_path}")

# Cross-validation
cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
print(f"\ncross-validation scores: {cv_scores}")
print(f"mean cv accuracy: {cv_scores.mean():.4f}")

# Save predictions
pred_path = os.path.join(OUTPUT_DIR, "decision_tree_predictions.csv")
df_results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})
df_results.to_csv(pred_path, index=False)
print(f"\npredictions saved to: {pred_path}")
