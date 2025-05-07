# evaluation.py
# evaluate and compare model performance for predicting small business survival

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_score, KFold
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import load_and_preprocess_data
from src.linear_regression import ordinary_least_squares, predict
from src.logistic_regression import sigmoid
import warnings

# suppress warnings from sklearn to keep things clean
warnings.filterwarnings('ignore')

def evaluate_linear_regression(X_train, X_test, y_train, y_test):
    """evaluate how well linear regression works on the data"""
    start_time = time.time()
    
    # fit the model using ordinary least squares
    beta = ordinary_least_squares(X_train.values.astype(float), y_train.values.astype(float))

    
    # predict probabilities on the test set
    y_pred_prob = predict(X_test.values, beta)
    y_pred = (y_pred_prob >= 0.5).astype(int)  # convert probabilities to binary predictions
    
    # measure how long training took
    training_time = time.time() - start_time
    
    # calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # cross-validation: check how the model generalizes
    cv_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        beta_cv = ordinary_least_squares(X_cv_train.values.astype(float), y_cv_train.values.astype(float))
        y_cv_pred = (predict(X_cv_val.values, beta_cv) >= 0.5).astype(int)
        cv_scores.append(accuracy_score(y_cv_val, y_cv_pred))
    
    cv_accuracy = np.mean(cv_scores)
    
    return {
        "model": "linear regression",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_accuracy": cv_accuracy,
        "training_time": training_time,
        "predictions": y_pred,
        "probabilities": y_pred_prob
    }

def train_logistic_regression(X_train, y_train, learning_rate=0.1, epochs=1000):
    """train logistic regression model from scratch using gradient descent"""
    # add bias term by appending ones to the feature set
    X = np.hstack([np.ones((X_train.shape[0], 1)), X_train.values])
    
    # initialize weights to zero
    weights = np.zeros(X.shape[1])
    
    # train model with gradient descent over a number of epochs
    for _ in range(epochs):
        z = np.dot(X, weights)
        preds = sigmoid(z)
        error = y_train.values - preds
        gradient = np.dot(X.T, error) / len(y_train)
        weights += learning_rate * gradient  # update weights
    
    return weights

def evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    """evaluate logistic regression model on the test set"""
    start_time = time.time()
    
    # train the model
    weights = train_logistic_regression(X_train, y_train)
    
    # make predictions on the test data
    X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test.values])
    y_pred_prob = sigmoid(np.dot(X_test_bias, weights))
    y_pred = (y_pred_prob >= 0.5).astype(int)  # convert probabilities to binary predictions
    
    # track the time it took to train
    training_time = time.time() - start_time
    
    # evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # cross-validation to test model stability
    cv_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        weights_cv = train_logistic_regression(X_cv_train, y_cv_train)
        X_cv_val_bias = np.hstack([np.ones((X_cv_val.shape[0], 1)), X_cv_val.values])
        y_cv_pred = (sigmoid(np.dot(X_cv_val_bias, weights_cv)) >= 0.5).astype(int)
        cv_scores.append(accuracy_score(y_cv_val, y_cv_pred))
    
    cv_accuracy = np.mean(cv_scores)
    
    return {
        "model": "logistic regression",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_accuracy": cv_accuracy,
        "training_time": training_time,
        "predictions": y_pred,
        "probabilities": y_pred_prob
    }


if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

    # Evaluate both models
    linear_results = evaluate_linear_regression(X_train, X_test, y_train, y_test)
    logistic_results = evaluate_logistic_regression(X_train, X_test, y_train, y_test)

    # Save model performance to CSV
    df_results = pd.DataFrame([
        {k: v for k, v in linear_results.items() if k not in ['predictions', 'probabilities']},
        {k: v for k, v in logistic_results.items() if k not in ['predictions', 'probabilities']}
    ])
    df_results.to_csv("../processed/model_comparison.csv", index=False)

    # Plot ROC Curves
    fpr_lin, tpr_lin, _ = roc_curve(y_test, linear_results["probabilities"])
    fpr_log, tpr_log, _ = roc_curve(y_test, logistic_results["probabilities"])

    plt.figure()
    plt.plot(fpr_lin, tpr_lin, label=f"Linear (AUC={roc_auc_score(y_test, linear_results['probabilities']):.2f})")
    plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC={roc_auc_score(y_test, logistic_results['probabilities']):.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../processed/roc_curves_comparison.png")
    plt.close()

    # Plot Confusion Matrices
    cm_lin = confusion_matrix(y_test, linear_results["predictions"])
    cm_log = confusion_matrix(y_test, logistic_results["predictions"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_lin, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Linear Regression Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_log, annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("Logistic Regression Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("../processed/confusion_matrices.png")
    plt.close()

    print("Evaluation complete. Results saved to ../processed/")
