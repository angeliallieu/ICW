"""
Modellierung: Training, Prediction, und Persistierung von ML-Modellen.

Dieses Modul behandelt das Trainieren und Speichern von ML-Modellen.
Aktuell: Random Forest Classifier via scikit-learn.

Funktionen:
-----------
- train_model()    : Trainiert ein Klassifikator-Modell
- predict()        : Macht Predictions mit einem trainierten Modell
- save_model()     : Speichert Modell als joblib
- load_model()     : Lädt Modell von joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Union, Any, Dict


def train_model(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    algorithm: str = "random_forest",
    params: Dict[str, Any] = None
) -> Union[RandomForestClassifier, LogisticRegression]:
    """
    Trainiert ein Klassifikator-Modell auf den Trainings-Daten.

    Diese Funktion:
    1. Erstellt ein Modell basierend auf dem algorithm Parameter
    2. Trainiert es auf X_train und y_train
    3. Returned das trainierte Modell

    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Trainings-Features mit Shape (n_samples, n_features).

    y_train : np.ndarray or pd.Series
        Trainings-Target mit Shape (n_samples,).

    algorithm : str
        "random_forest" oder "logistic_regression".

    params : Dict[str, Any]
        Hyperparameter-Dictionary mit Modell-spezifischen Parametern.
        Falls None: Defaults werden verwendet.

        Für "random_forest":
            {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 42
            }

        Für "logistic_regression":
            {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42
            }

    Returns:
    --------
    model : RandomForestClassifier or LogisticRegression
        Trainiertes Modell, bereit für Predictions.

    Beispiel:
    ---------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = train_model(X_train, y_train, "random_forest")
    >>> model
    RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, ...)

    """
    # Setze defaults falls params nicht übergeben
    if params is None:
        params = {}

    if algorithm == "random_forest":
        # Default Parameter für Random Forest
        rf_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,  # Alle Prozessoren
        }
        # Merge mit übergebenen Parametern (übergeben params überschreiben defaults)
        rf_params.update(params)

        # Erstelle und trainiere das Modell
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

    elif algorithm == "logistic_regression":
        # Default Parameter für Logistic Regression
        lr_params = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        }
        # Merge mit übergebenen Parametern
        lr_params.update(params)

        # Erstelle und trainiere das Modell
        model = LogisticRegression(**lr_params)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'random_forest' or 'logistic_regression'")

    return model


def predict(
    model: Union[RandomForestClassifier, LogisticRegression],
    X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Macht Predictions mit einem trainierten Modell.

    Diese Funktion:
    1. Nimmt ein trainiertes Modell
    2. Macht Predictions auf neuen Daten X
    3. Returned Predictions als numpy array

    Parameters:
    -----------
    model : Trained Classifier
        Ein trainiertes Modell (RandomForest, LogisticRegression, etc.).

    X : np.ndarray or pd.DataFrame
        Input-Daten zum Vorhersagen mit Shape (n_samples, n_features).

    Returns:
    --------
    predictions : np.ndarray
        Vorhergesagte Klassen mit Shape (n_samples,).
        Werte: 0 oder 1 (für Binary Classification).

    Beispiel:
    ---------
    >>> predictions = predict(model, X_test)
    >>> predictions
    array([1, 0, 1, 1, 0, ...])

    """
    predictions = model.predict(X)
    return predictions


def predict_proba(
    model: Union[RandomForestClassifier, LogisticRegression],
    X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Macht probabilistische Predictions .

    Parameters:
    -----------
    model : Trained Classifier
        Ein trainiertes Modell.

    X : np.ndarray or pd.DataFrame
        Input-Daten.

    Returns:
    --------
    probabilities : np.ndarray
        Wahrscheinlichkeiten für jede Klasse mit Shape (n_samples, 2).
        Column 0: P(Klasse 0)
        Column 1: P(Klasse 1)

    Beispiel:
    ---------
    >>> probs = predict_proba(model, X_test)
    >>> probs[0]
    array([0.1, 0.9])  # 90% sicher dass Klasse 1
    """
    probabilities = model.predict_proba(X)
    return probabilities


def save_model(model: Union[RandomForestClassifier, LogisticRegression], filepath: str) -> None:
    """
    Speichert ein trainiertes Modell als joblib File.

    Parameters:
    -----------
    model : Trained Classifier
        Das trainierte Modell.

    filepath : str
        Pfad wo das Modell gespeichert werden soll.
        Beispiel: "data/04_models/random_forest_model.joblib"

    Beispiel:
    ---------
    >>> save_model(model, "data/04_models/model.joblib")
    # Output: "✓ Model saved to data/04_models/model.joblib"

    """
    # Erstelle das Verzeichnis falls nicht vorhanden
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Speichere das Modell
    joblib.dump(model, filepath)

    print(f"✓ Model saved to {filepath}")


def load_model(filepath: str) -> Union[RandomForestClassifier, LogisticRegression]:
    """
    Lädt ein trainiertes Modell von joblib File.

    Diese Funktion:
    1. Lädt das Modell von filepath mit joblib.load()
    2. Returned das Modell
    3. Printed eine Bestätigung

    Parameters:
    -----------
    filepath : str
        Pfad zum gespeicherten Modell.
        Beispiel: "data/04_models/random_forest_model.joblib"

    Returns:
    --------
    model : Trained Classifier
        Das geladene trainierte Modell, bereit für Predictions.

    Beispiel:
    ---------
    >>> model = load_model("data/04_models/model.joblib")
    >>> predictions = model.predict(X_test)

    """
    # Lade das Modell
    model = joblib.load(filepath)
    
    print(f"✓ Model loaded from {filepath}")
    
    return model
