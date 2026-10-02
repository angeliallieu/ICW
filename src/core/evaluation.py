"""
Evaluierung: Berechnung von Evaluierungs-Metriken und Reporting.

Dieses Modul berechnet verschiedene Metriken zur Evaluierung der Modell-Performance.

Funktionen:
-----------
- evaluate()                    : Berechnet alle Metriken
- compute_confusion_matrix()    : Berechnet Confusion Matrix
- save_metrics()                : Speichert Metriken als JSON
- get_classification_report()   : Detaillierter Report mit Precision/Recall/F1
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Any


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Berechnet umfassende Evaluierungs-Metriken.
    
    Diese Funktion berechnet:
    1. Accuracy: Anteil korrekter Vorhersagen
    2. Precision: Von den positiv vorhergesagten, wie viele sind wirklich positiv?
    3. Recall: Von den wirklich positiven, wie viele haben wir erkannt?
    4. F1-Score: Harmonisches Mittel von Precision und Recall
    5. ROC-AUC: Area Under Curve (braucht y_pred_proba)

    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary mit den berechneten Metriken.
        {
            "accuracy": 0.956,
            "precision": 0.970,
            "recall": 0.945,
            "f1": 0.957,
            "roc_auc": 0.987
        }

    """
    # Berechne Metriken
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Berechne ROC-AUC falls Wahrscheinlichkeiten vorhanden
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Berechnet die Confusion Matrix.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Echte Labels.
    
    y_pred : np.ndarray
        Vorhergesagte Labels.
    
    Returns:
    --------
    cm : np.ndarray
        Confusion Matrix mit Shape (2, 2).
        [[TN, FP],
         [FN, TP]]

    """
    cm = sklearn_confusion_matrix(y_true, y_pred)
    return cm


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Gibt einen detaillierten Classification Report aus.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Echte Labels.
    
    y_pred : np.ndarray
        Vorhergesagte Labels.
    
    Returns:
    --------
    report : str
        Detaillierter Report mit Precision, Recall, F1 pro Klasse.
    

    """
    report = classification_report(y_true, y_pred)
    return report


def save_metrics(
    metrics: Dict[str, float],
    cm: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    filepath_metrics: str,
    filepath_cm: str,
    filepath_predictions: str
) -> None:
    """
    Speichert Metriken und Confusion Matrix.
    
    Diese Funktion:
    1. Speichert Metriken als JSON
    2. Speichert Confusion Matrix als CSV
    3. Speichert Predictions mit echten Labels als CSV
    4. Printed Bestätigung
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Dictionary mit Metriken (von evaluate() function).
    
    cm : np.ndarray
        Confusion Matrix (von compute_confusion_matrix() function).
    
    y_pred : np.ndarray
        Vorhergesagte Labels.
    
    y_true : np.ndarray
        Echte Labels.
    
    filepath_metrics : str
        Pfad zum Speichern der Metriken (z.B. "data/05_metrics/metrics.json").
    
    filepath_cm : str
        Pfad zum Speichern der Confusion Matrix (z.B. "data/05_metrics/confusion_matrix.csv").
    
    filepath_predictions : str
        Pfad zum Speichern der Predictions (z.B. "data/05_metrics/predictions.csv").
    
    Beispiel:
    ---------
    >>> save_metrics(
    ...     metrics, cm, y_pred, y_true,
    ...     "data/05_metrics/metrics.json",
    ...     "data/05_metrics/confusion_matrix.csv",
    ...     "data/05_metrics/predictions.csv"
    ... )
    # Output:
    # ✓ Metrics saved to data/05_metrics/metrics.json
    # ✓ Confusion Matrix saved to data/05_metrics/confusion_matrix.csv
    # ✓ Predictions saved to data/05_metrics/predictions.csv

    """
    # Erstelle das Verzeichnis falls nicht vorhanden
    directory = os.path.dirname(filepath_metrics)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Speichere Metriken als JSON
    with open(filepath_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to {filepath_metrics}")
    
    # Speichere Confusion Matrix als CSV
    cm_df = pd.DataFrame(
        cm,
        index=['Predicted 0', 'Predicted 1'],
        columns=['Actual 0', 'Actual 1']
    )
    cm_df.to_csv(filepath_cm)
    print(f"✓ Confusion Matrix saved to {filepath_cm}")
    
    # Speichere Predictions mit echten Labels als CSV
    pred_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'match': (y_true == y_pred).astype(int)
    })
    pred_df.to_csv(filepath_predictions, index=False)
    print(f"✓ Predictions saved to {filepath_predictions}")
