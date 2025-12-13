"""
Helper functions for benchmark model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, auc


def calc_scores(cm):
    """
    Calculates performance metrics from a confusion matrix.
    cm: 2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    # Balanced accuracy
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc = 0.5 * (specificity + sensitivity)

    # F1 score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = (
        2 * sensitivity * precision / (sensitivity + precision)
        if (sensitivity + precision) > 0
        else 0
    )

    return np.array([balanced_acc, f1, specificity, sensitivity])


def get_model_performance(
    y_prob,
    y_exp,
    calibration_indices,
    eval_indices,
    train_indices=None,
    adjust_threshold=False,
):
    """
    Evaluates model performance using the optimal threshold found from calibration data.

    Parameters:
    -----------
    y_prob : array-like
        Predicted probabilities for all data (for positive class)
    y_exp : array-like
        True labels for all data
    calibration_indices : array-like
        Indices for calibration set (used to find optimal threshold)
    eval_indices : array-like
        Indices for validation set (used to evaluate performance)
    train_indices : array-like, optional
        Indices for training set (to optionally compute in-sample AUC)
        If None, in-sample AUC is not computed.
    adjust_threshold : bool, default=False
        If True, multiply the threshold by 1.2 (only for special chain-models without entropy)

    Returns:
    --------
    dict : Dictionary with performance metrics:
        - auc: AUC score for the validation set (out-of-sample)
        - in_sample_auc: AUC score for the training set (if train_indices is provided)
        - balanced_accuracy: Balanced accuracy
        - f1: F1 score
        - specificity: Specificity
        - sensitivity: Sensitivity (recall)
        - threshold: Threshold used
    """

    # Threshold adjustment
    num_factor = 1.2 if adjust_threshold else 1.0

    # Find optimal threshold on calibration data (closest to top-left)
    fpr_cal, tpr_cal, thresholds_cal = roc_curve(
        y_exp[calibration_indices], y_prob[calibration_indices]
    )

    # Compute distance to top-left corner (0, 1)
    distances = np.sqrt((1 - tpr_cal) ** 2 + fpr_cal**2)
    best_idx = np.argmin(distances)
    optimal_threshold = thresholds_cal[best_idx] * num_factor

    # Predictions on validation data using optimal threshold
    y_pred = (y_prob[eval_indices] >= optimal_threshold).astype(int)

    # Confusion matrix on validation data
    cm = confusion_matrix(y_exp[eval_indices], y_pred)

    # AUC on validation data (out-of-sample)
    auc_score = 100 * roc_auc_score(y_exp[eval_indices], y_prob[eval_indices])

    # In-sample AUC (optional)
    in_sample_auc = None
    if train_indices is not None:
        in_sample_auc = 100 * roc_auc_score(y_exp[train_indices], y_prob[train_indices])

    # Classification metrics
    classification_scores = 100 * calc_scores(cm)

    # Return results as a dictionary
    results = {
        "auc": round(auc_score, 2),
        "in_sample_auc": round(in_sample_auc, 2) if in_sample_auc is not None else None,
        "balanced_accuracy": round(classification_scores[0], 2),
        "f1": round(classification_scores[1], 2),
        "specificity": round(classification_scores[2], 2),
        "sensitivity": round(classification_scores[3], 2),
        "threshold": round(optimal_threshold, 4),
    }

    return results


def plot_roc_curve(y_true, y_prob, save_path):
    """
    Plots an ROC curve for the given data and reports the threshold for the point
    closest to (0, 1). Saves the figure to ASSETS_FOLDER / "roc_plot.png".

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Probabilities or scores for the positive class.
    """
    import os

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Find the point with smallest Euclidean distance to (0, 1)
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
    min_idx = np.argmin(distances)
    closest_fpr = fpr[min_idx]
    closest_tpr = tpr[min_idx]
    closest_threshold = thresholds[min_idx]

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.plot(0, 1, marker="o", color="green", markersize=8, label="(0, 1) Ideal")
    plt.plot(
        closest_fpr,
        closest_tpr,
        marker="o",
        color="red",
        markersize=8,
        label=f"Closest to (0, 1), threshold={closest_threshold*100:.2f}%",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()
