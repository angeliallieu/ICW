"""
Feature Engineering: Feature-Selection, Encoding, und Deterministische Train/Test Split.

Dieses Modul behandelt die Transformation und Preparation von Features für das ML-Modell.

Funktionen:
-----------
- preprocess_features() : Feature-Selection (SelectKBest), Normalisierung
- train_test_split()   : Deterministische Aufteilung in Train/Test Sets
- encode_target()      : Label-Encoding des Targets (falls nötig)

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple


def preprocess_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, list]:
    """
    Feature Preprocessing mit SelectKBest.

    Diese Funktion:
    1. Benutzt SelectKBest mit f_classif um beste Features auszuwählen
    2. Wählt die k besten Features basierend auf F-Statistik
    3. Returned die gefilterte Feature-Matrix und die Namen der selectierten Features

    Parameters:
    -----------
    X : pd.DataFrame
        Feature-Matrix (569, 30).

    y : pd.Series
        Target-Vektor (569,).

    n_features : int
        Anzahl der besten Features zum Auswählen (Default: 20).

    random_state : int
        Für Reproduzierbarkeit (auch wenn SelectKBest selbst nicht zufällig ist,
        verwenden wir ihn für Konsistenz).

    Returns:
    --------
    X_selected : pd.DataFrame
        Features nach Selektion mit Form (569, 20).

    selected_features : list
        Namen der ausgewählten Features.

    Beispiel:
    ---------
    >>> X, y = load_breast_cancer()
    >>> X_selected, feature_names = preprocess_features(X, y, n_features=20)
    >>> X_selected.shape
    (569, 20)
    >>> feature_names[:3]
    ['mean radius', 'mean perimeter', 'mean concavity']

    """
    # Erstelle SelectKBest mit k Features und f_classif scoring
    selector = SelectKBest(
        score_func=f_classif,
        k=n_features
    )

    selector.fit(X, y)

    # Wende Selector an
    X_array = selector.transform(X)

    # Hole die Namen der selektierten Features
    # get_support() gibt boolean array zurück (True für selected Features)
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = X.columns[selected_indices].tolist()

    # Konvertiere zurück zu DataFrame mit nur selectierten Features
    X_selected = pd.DataFrame(
        data=X_array,
        columns=selected_feature_names,
        index=X.index
    )

    return X_selected, selected_feature_names


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Deterministische Aufteilung in Train und Test Sets.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature-Matrix.

    y : pd.Series
        Target-Vektor.

    test_size : float
        Anteil des Test-Sets (0.2 = 20% Test, 80% Train).

    random_state : int
        Seed für Zufallsgenerator (MUSS immer gleich sein!).

    Returns:
    --------
    X_train, X_test : pd.DataFrame
        Train und Test Features.

    y_train, y_test : pd.Series
        Train und Test Targets.


    """
    # Splitte die Daten mit sklearn
    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Stratified Split: respektiert Klassenverteilung
    )

    return X_train, X_test, y_train, y_test


def encode_target(y: pd.Series) -> pd.Series:
    """
    Label-Encoding für Target-Vektor (falls nötig).

    Für Breast Cancer ist das Target bereits 0/1 (numerisch), daher nicht nötig.
    Aber diese Funktion ist bereitgestellt für andere Datasets, wo Target
    kategorisch sein könnte (z.B. ['cat', 'dog', 'bird'] → [0, 1, 2]).

    Parameters:
    -----------
    y : pd.Series
        Target-Vektor (kann numerisch oder kategorisch sein).

    Returns:
    --------
    y_encoded : pd.Series
        Label-encoded Target (immer numerisch 0, 1, 2, ...).

    Beispiel:
    ---------
    >>> y = pd.Series(['malignant', 'benign', 'malignant'])
    >>> y_encoded = encode_target(y)
    >>> y_encoded.values
    array([1, 0, 1])

    """
    # Falls bereits numerisch, return as-is
    if y.dtype in ['int64', 'int32', 'float64']:
        return y

    # Falls kategorisch, encode
    le = LabelEncoder()
    y_encoded = pd.Series(
        le.fit_transform(y),
        index=y.index,
        name=y.name
    )

    return y_encoded
