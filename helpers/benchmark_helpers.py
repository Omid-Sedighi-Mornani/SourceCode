"""
Helper-Funktionen für Benchmark-Model-Evaluation
"""
from typing import List, Dict, Callable, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix


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


def calc_scores(confusion_mat: npt.NDArray[np.int_]) -> tuple[float, float, float, float]:
    """
    Berechnet Classification Scores aus einer Confusion Matrix.

    Übersetzt aus R code (helper_functions.R):
    calc_scores <- function(tbl){
        balanced_acc <- 0.5*sum(diag(tbl)/colSums(tbl))
        spec <- tbl[1,1]/(sum(tbl[,1]))
        sens <- tbl[2,2]/sum(tbl[,2])
        prec <- tbl[2,2]/sum(tbl[2,])
        f1 <- 2*sens*prec/(sens+prec)
        return(c(balanced_acc,f1,spec,sens))
    }

    Parameters
    ----------
    confusion_mat : ndarray
        2x2 Confusion Matrix mit der Struktur:
        [[TN, FP],
         [FN, TP]]

    Returns
    -------
    tuple
        (balanced_accuracy, f1_score, specificity, sensitivity)
    """
    # Specificity (True Negative Rate)
    spec = confusion_mat[0, 0] / np.sum(confusion_mat[:, 0])

    # Sensitivity (True Positive Rate / Recall)
    sens = confusion_mat[1, 1] / np.sum(confusion_mat[:, 1])

    # Precision
    prec = confusion_mat[1, 1] / np.sum(confusion_mat[1, :])

    # F1 Score
    f1 = 2 * sens * prec / (sens + prec)

    # Balanced Accuracy
    balanced_acc = 0.5 * (spec + sens)

    return balanced_acc, f1, spec, sens


def get_model_performance(
    probability_score: npt.NDArray[np.float64],
    observations: npt.NDArray[np.int_],
    calibration_index: npt.NDArray[np.int_],
    validation_index: npt.NDArray[np.int_],
    transpose: bool = False,
    formula_str: str | None = None
) -> npt.NDArray[np.float64]:
    """
    Berechnet Model Performance Metriken inkl. AUC und Classification Scores.

    Übersetzt aus R code (helper_functions.R, Zeilen 91-121):
    get_model_performance <- function(probability_score,
                                      observations,
                                      calibration_index,
                                      validation_index,
                                      transpose = FALSE,
                                      f){
        num_factor = 1.0
        if((!f %>% as.vector() %>% str_detect("ENTR") %>% any()) &
           (f %>% as.vector() %>% str_detect("chain") %>% any())){
            num_factor = 1.2
        }
        thresh <- coords(roc(...), x = "best", best.method = "closest.topleft", ...) * num_factor
        ...
        return(round(c(auc_score,classification_scores_at_thresh), 2))
    }

    Parameters
    ----------
    probability_score : ndarray
        Array von Vorhersagewahrscheinlichkeiten
    observations : ndarray
        Array von True Binary Labels (0 oder 1)
    calibration_index : ndarray
        Indizes für Calibration Set (zum Finden des optimalen Thresholds)
    validation_index : ndarray
        Indizes für Validation/Test Set (zur Evaluation)
    transpose : bool, default=False
        Wenn True, wird die Confusion Matrix anders orientiert
    formula_str : str, optional
        Formula String zur Prüfung spezifischer Anpassungen

    Returns
    -------
    ndarray
        Array mit [AUC, Balanced_Accuracy, F1, Specificity, Sensitivity] (alle in %)

    Examples
    --------
    >>> # Wie in 3_analysis.R Zeilen 500-505 verwendet:
    >>> results = get_model_performance(
    ...     close_prob,
    ...     data_stan['Closed'],
    ...     calibration_idx,
    ...     val_idx,
    ...     f="Closed ~ MEAN + VAR"
    ... )
    """
    # Bestimme num_factor basierend auf Formula (R Code Zeilen 98-100)
    num_factor = 1.0
    if formula_str is not None:
        formula_str_lower = str(formula_str).lower()
        has_chain = 'chain' in formula_str_lower
        has_entr = 'entr' in formula_str_lower

        if has_chain and not has_entr:
            num_factor = 1.2

    # Finde optimalen Threshold mit Calibration Set
    # Verwende Youden's J Statistic (closest to top-left)
    fpr, tpr, thresholds = roc_curve(
        observations[calibration_index],
        probability_score[calibration_index]
    )

    # Berechne J Statistic (Youden's index)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx] * num_factor

    # Erstelle Vorhersagen auf Validation Set
    predictions = (probability_score[validation_index] >= optimal_threshold).astype(int)

    # Erstelle Confusion Matrix
    conf_mat = confusion_matrix(observations[validation_index], predictions)

    # Handle transpose und Column Reordering (R Code Zeilen 107-112)
    if not transpose:
        # Reorder columns [1, 0] -> swap columns (wie in R: [,c(2,1)])
        if conf_mat.shape == (2, 2):
            conf_mat = conf_mat[:, [1, 0]]

    # Berechne AUC auf Validation Set (R Code Zeile 113-114)
    auc_score = 100 * roc_auc_score(
        observations[validation_index],
        probability_score[validation_index]
    )

    # Berechne Classification Scores (R Code Zeile 117)
    classification_scores = calc_scores(conf_mat)
    classification_scores_percent = tuple(100 * np.array(classification_scores))

    # Kombiniere Ergebnisse (R Code Zeile 119)
    results = np.array([auc_score] + list(classification_scores_percent))

    return np.round(results, 2)
