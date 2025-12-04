"""
Helper-Funktionen für Benchmark-Model-Evaluation
"""
from typing import List, Dict, Callable, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def create_model_comparison_df(
    model_configs: List[Dict[str, Any]],
    y_train: npt.NDArray[np.int_],
    y_test: npt.NDArray[np.int_]
) -> pd.DataFrame:
    """
    Erstellt einen DataFrame zum Vergleich mehrerer Modelle.

    Parameters:
    -----------
    model_configs : list of dict
        Liste von Dictionaries mit Keys:
        - 'model_type': str (z.B. "Logistic Regression")
        - 'features': str (z.B. "Mean Rating", "All Features")
        - 'y_pred_train': array
        - 'y_proba_train': array
        - 'y_pred_test': array
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
    data = {
        'Model Type': [],
        'Features': [],
        'Test Accuracy': [],
        'Test ROC-AUC': [],
        'Train ROC-AUC': []
    }

    for config in model_configs:
        data['Model Type'].append(config['model_type'])
        data['Features'].append(config['features'])
        data['Test Accuracy'].append(
            accuracy_score(y_test, config['y_pred_test'])
        )
        data['Test ROC-AUC'].append(
            roc_auc_score(y_test, config['y_proba_test'])
        )
        data['Train ROC-AUC'].append(
            roc_auc_score(y_train, config['y_proba_train'])
        )

    df = pd.DataFrame(data)
    df['Overfitting Gap'] = df['Train ROC-AUC'] - df['Test ROC-AUC']

    return df


def create_single_model_comparison_df(
    model_name: str,
    model_configs: List[Dict[str, Any]],
    y_train: npt.NDArray[np.int_],
    y_test: npt.NDArray[np.int_],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Erstellt einen DataFrame zum Vergleich verschiedener Feature-Sets
    für einen einzelnen Modell-Typ.

    Parameters:
    -----------
    model_name : str
        Name des Modells (z.B. "Random Forest", "XGBoost")
    model_configs : list of dict
        Liste von Dictionaries mit Keys für verschiedene Feature-Sets
    y_train, y_test : array
        True labels
    feature_names : list of str
        Namen der Feature-Sets (z.B. ["Mean Rating", "Business Cov.", "All Features"])

    Returns:
    --------
    pd.DataFrame
        DataFrame mit Vergleich der Feature-Sets
    """
    data = {
        'Model': [],
        'Features': [],
        'Train Accuracy': [],
        'Test Accuracy': [],
        'Train ROC-AUC': [],
        'Test ROC-AUC': []
    }

    for config, feat_name in zip(model_configs, feature_names):
        data['Model'].append(f"{model_name}: {feat_name}")
        data['Features'].append(config['features'])
        data['Train Accuracy'].append(
            accuracy_score(y_train, config['y_pred_train'])
        )
        data['Test Accuracy'].append(
            accuracy_score(y_test, config['y_pred_test'])
        )
        data['Train ROC-AUC'].append(
            roc_auc_score(y_train, config['y_proba_train'])
        )
        data['Test ROC-AUC'].append(
            roc_auc_score(y_test, config['y_proba_test'])
        )

    return pd.DataFrame(data)


def get_model_predictions(
    model: Any,
    X_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int_]
) -> Dict[str, npt.NDArray]:
    """
    Holt Predictions für Train und Test Sets.

    Parameters:
    -----------
    model : sklearn estimator
        Trainiertes Modell
    X_train, X_test : array-like
        Feature matrices
    y_train : array
        Training labels (nur für Kompatibilität)

    Returns:
    --------
    dict
        Dictionary mit predictions und probabilities
    """
    return {
        'y_pred_train': model.predict(X_train),
        'y_proba_train': model.predict_proba(X_train)[:, 1],
        'y_pred_test': model.predict(X_test),
        'y_proba_test': model.predict_proba(X_test)[:, 1]
    }


def print_model_summary(
    model_name: str,
    y_train: npt.NDArray[np.int_],
    y_pred_train: npt.NDArray[np.int_],
    y_proba_train: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int_],
    y_pred_test: npt.NDArray[np.int_],
    y_proba_test: npt.NDArray[np.float64],
    classification_report_fn: Callable
) -> None:
    """
    Druckt eine standardisierte Modell-Zusammenfassung.

    Parameters:
    -----------
    model_name : str
        Name des Modells
    y_train, y_test : array
        True labels
    y_pred_train, y_pred_test : array
        Predicted labels
    y_proba_train, y_proba_test : array
        Predicted probabilities
    classification_report_fn : function
        sklearn classification_report Funktion
    """
    print("=" * 60)
    print(f"{model_name}")
    print("=" * 60)

    print("\nTRAIN Performance:")
    print(f"  Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}")

    print("\nTEST Performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}")

    print("\nTest Classification Report:")
    print(classification_report_fn(y_test, y_pred_test,
                                   target_names=["Open", "Closed"]))
