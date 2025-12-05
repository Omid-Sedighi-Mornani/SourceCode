"""
Helper-Funktionen für Benchmark-Model-Evaluation
"""

from typing import List, Dict, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score


def create_model_comparison_df(
    model_configs: List[Dict[str, Any]],
    y_train: npt.NDArray[np.int_],
    y_test: npt.NDArray[np.int_],
) -> pd.DataFrame:
    """
    Erstellt einen DataFrame zum Vergleich mehrerer Modelle.

    Parameters:
    -----------
    model_configs : list of dict
        Liste von Dictionaries mit Keys:
        - 'model_type': str (z.B. "Logistic Regression")
        - 'features': str (z.B. "Mean Rating", "All Features")
        - 'y_proba_train': array
        - 'y_proba_test': array
    y_train : array
        True labels für Training
    y_test : array
        True labels für Test

    Returns:
    --------
    pd.DataFrame
        DataFrame mit Metriken für alle Modelle
    """
    data = {"Model Type": [], "Features": [], "Test ROC-AUC": [], "Train ROC-AUC": []}

    for config in model_configs:
        data["Model Type"].append(config["model_type"])
        data["Features"].append(config["features"])
        data["Test ROC-AUC"].append(roc_auc_score(y_test, config["y_proba_test"]))
        data["Train ROC-AUC"].append(roc_auc_score(y_train, config["y_proba_train"]))

    df = pd.DataFrame(data)

    return df


def print_model_summary(
    model_name: str,
    y_train: npt.NDArray[np.int_],
    y_proba_train: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int_],
    y_proba_test: npt.NDArray[np.float64],
) -> None:
    """
    Druckt eine standardisierte Modell-Zusammenfassung.

    Parameters:
    -----------
    model_name : str
        Name des Modells
    y_train, y_test : array
        True labels
    y_proba_train, y_proba_test : array
        Predicted probabilities
    """
    print("=" * 60)
    print(f"{model_name}")
    print("=" * 60)

    print("\nTRAIN Performance:")
    print(f"  ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}")

    print("\nTEST Performance:")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}")


import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def calc_scores(cm):
    """
    Berechnet Performance-Metriken aus einer Confusion Matrix
    cm: 2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    # Balanced Accuracy
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc = 0.5 * (specificity + sensitivity)

    # F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = (
        2 * sensitivity * precision / (sensitivity + precision)
        if (sensitivity + precision) > 0
        else 0
    )

    return np.array([balanced_acc, f1, specificity, sensitivity])


import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def calc_scores(cm):
    """
    Berechnet Performance-Metriken aus einer Confusion Matrix
    cm: 2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    # Balanced Accuracy
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc = 0.5 * (specificity + sensitivity)

    # F1 Score
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
    adjust_threshold=False,
):
    """
    Evaluiert Model Performance mit optimalem Threshold aus Calibration-Daten

    Parameters:
    -----------
    probability_score : array-like
        Predicted probabilities für alle Daten (für positive Klasse)
    observations : array-like
        True labels für alle Daten
    calibration_indices : array-like
        Indices für Calibration-Set (um optimalen Threshold zu finden)
    validation_indices : array-like
        Indices für Validation-Set (um Performance zu messen)
    adjust_threshold : bool, default=False
        Wenn True, wird Threshold mit Faktor 1.2 multipliziert
        (nur für spezielle Chain-Modelle ohne Entropy)

    Returns:
    --------
    dict : Dictionary mit Performance-Metriken
        - auc: AUC Score auf Validation-Daten
        - balanced_accuracy: Balanced Accuracy
        - f1: F1 Score
        - specificity: Specificity
        - sensitivity: Sensitivity (Recall)
        - threshold: Verwendeter Threshold
    """

    # Threshold-Anpassung
    num_factor = 1.2 if adjust_threshold else 1.0

    # Optimalen Threshold auf Calibration-Daten finden (closest to top-left)
    fpr_cal, tpr_cal, thresholds_cal = roc_curve(
        y_exp[calibration_indices], y_prob[calibration_indices]
    )

    # Berechne Distanz zur oberen linken Ecke (0, 1)
    distances = np.sqrt((1 - tpr_cal) ** 2 + fpr_cal**2)
    best_idx = np.argmin(distances)
    optimal_threshold = thresholds_cal[best_idx] * num_factor

    # Predictions auf Validation-Daten mit optimalem Threshold
    y_pred = (y_prob[eval_indices] >= optimal_threshold).astype(int)

    # Confusion Matrix auf Validation-Daten
    cm = confusion_matrix(y_exp[eval_indices], y_pred)

    # AUC auf Validation-Daten
    auc_score = 100 * roc_auc_score(y_exp[eval_indices], y_prob[eval_indices])

    # Klassifikationsmetriken
    classification_scores = 100 * calc_scores(cm)

    # Ergebnisse als Dictionary zurückgeben
    results = {
        "auc": round(auc_score, 2),
        "balanced_accuracy": round(classification_scores[0], 2),
        "f1": round(classification_scores[1], 2),
        "specificity": round(classification_scores[2], 2),
        "sensitivity": round(classification_scores[3], 2),
        "threshold": round(optimal_threshold, 4),
    }

    return results
