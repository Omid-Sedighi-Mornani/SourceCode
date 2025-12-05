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
