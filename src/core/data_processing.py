"""
Framework-agnostisches Daten-Modul für Breast Cancer Classification.

Dieses Modul kapselt alle Operationen zum Laden, Bereinigen und Normalisieren
von Daten. Es ist vollständig UNABHÄNGIG von:
- Kedro
- Vanilla Pipeline
- Prefect

Das ermöglicht Wiederverwendbarkeit und Testbarkeit.

Funktionen:
-----------
- load_breast_cancer() : Lädt Sklearn Breast Cancer Dataset
- clean_data()         : Entfernt NaN, Duplikate, Ausreißer
- normalize_features() : Normalisiert Features mit StandardScaler

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def load_breast_cancer(csv_path: str = "data/01_raw/breast-cancer.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Lädt den Breast Cancer Dataset von einer lokalen CSV-Datei.

    Diese Funktion:
    1. Lädt die CSV-Datei von data/01_raw/breast-cancer.csv
    2. Trennt Features (X) und Target (y)
    3. Encodiert Target: M (Malignant) → 1, B (Benign) → 0
    4. Entfernt die id-Spalte (nicht relevant für ML)

    CSV-Struktur erwartet:
    - Spalte 'id': Sample-ID (wird entfernt)
    - Spalte 'diagnosis': M (Malignant) oder B (Benign) → wird zum Target
    - Alle anderen Spalten: Features (radius_mean, texture_mean, etc.)

    Parameters:
    -----------
    csv_path : str
        Pfad zur CSV-Datei (Default: "data/01_raw/breast-cancer.csv")

    Returns:
    --------
    X : pd.DataFrame
        Feature-Matrix mit Form (569, 30).
        Spalten: Tumore-Charakteristiken wie 'radius_mean', 'texture_mean', ...

    y : pd.Series
        Target-Vektor mit Form (569,).
        Werte: 0 (Benign) oder 1 (Malignant).

    Beispiel:
    ---------
    >>> X, y = load_breast_cancer("data/01_raw/breast-cancer.csv")
    >>> X.shape
    (569, 30)
    >>> y.value_counts()
    1    212  # Malignant (M)
    0    357  # Benign (B)

    - Zeigt, wie man lokale CSV-Daten lädt
    - Zeigt Feature-Target Separation
    - Label-Encoding (String → Numeric): M → 1, B → 0
    - Data Loading ist WICHTIG: Garbage In = Garbage Out!
    """
    # Lade die CSV-Datei
    print("DEBUG: Attempting to load CSV from:", csv_path)  # ← DIESE ZEILE
    df = pd.read_csv(csv_path)
    
    # Entferne die id-Spalte (nicht relevant für Modelltraining)
    df = df.drop(columns=['id'])
    
    # Trenne Target und Features
    # diagnosis: M (Malignant) → 1, B (Benign) → 0
    y = (df['diagnosis'] == 'M').astype(int)
    y.name = 'target'
    
    # Features: alle Spalten außer diagnosis
    X = df.drop(columns=['diagnosis'])
    
    return X, y


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt die Feature-Matrix von fehlenden Werten und Duplikaten.
    
    Diese Funktion:
    1. Entfernt Zeilen mit NaN/Null Werten
    2. Entfernt exakte Duplikate
    3. Returned die bereinigte DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input Feature-Matrix, eventuell mit NaN oder Duplikaten.
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Bereinigte Feature-Matrix.
    
    Beispiel:
    ---------
    >>> X, y = load_breast_cancer()
    >>> X_clean = clean_data(X)
    >>> X.shape, X_clean.shape
    ((569, 30), (569, 30))  # Kein Daten-Loss im BC Dataset (ist sehr sauber)

    - Data Cleaning ist ESSENTIELL im ML-Workflow
    - Real-world Daten haben oft NaN, Duplikate, Inkonsistenzen
    - dropna() entfernt Zeilen mit irgendeinem NaN
    - drop_duplicates() entfernt exakte Duplikate
    - In Production würde man detailliertere Quality-Checks durchführen!
    """
    # Entferme Zeilen mit fehlenden Werten
    df = df.dropna()
    
    # Entferne exakte Duplikate (selbe Werte in allen Spalten)
    df = df.drop_duplicates()
    
    # Reset Index, um lückenlose Indizes zu haben (optional aber sauberer)
    df = df.reset_index(drop=True)
    
    return df


def normalize_features(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalisiert Features mit StandardScaler (mean=0, std=1).
    
    Diese Funktion:
    1. Fit: Berechnet mean und std pro Feature (NUR auf Trainings-Daten!)
    2. Transform: Normalisiert Features zu (Feature - mean) / std
    3. Gibt bereinigte Features + Scaler zurück
    
    ✅ RICHTIG:
        # Scaler NUR auf Train fit
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)  # Nicht fit!
        # Der Scaler "sieht" nur Train-Daten, simuliert echte Prediction
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature-Matrix zum Normalisieren.
    
    scaler : Optional[StandardScaler]
        Existierender Scaler (z.B. vom Training).
        Falls None: Neuer Scaler wird erstellt.
    
    fit : bool
        Ob der Scaler fit werden soll (True bei Training, False bei Test).
    
    Returns:
    --------
    X_normalized : pd.DataFrame
        Normalisierte Features.
    
    scaler : StandardScaler
        Der verwendete Scaler (zum Speichern und später auf Test anwenden).
    
    Beispiel:
    ---------
    >>> X, y = load_breast_cancer()
    >>> X_clean = clean_data(X)
    >>> X_train, X_test = train_test_split(X_clean, test_size=0.2, random_state=42)
    
    >>> # Train-Normalisierung: fit=True, fit den Scaler
    >>> X_train_norm, scaler = normalize_features(X_train, fit=True)
    
    >>> # Test-Normalisierung: fit=False, nutze den Scaler vom Training
    >>> X_test_norm, _ = normalize_features(X_test, scaler=scaler, fit=False)
    
    >>> X_train_norm.mean().max()
    0.0001234
    >>> X_train_norm.std().mean()
    1.0000456


    - StandardScaler: Formel = (X - mean) / std
    - ESSENTIELL für Modelle wie SVM, KNN, Linear Regression
    - Random Forest braucht Normalisierung NICHT (tree-based)
    - fit() berechnet mean und std
    - transform() wendet die Normalisierung an
    - fit_transform() = fit + transform in einem
    """
    # Falls kein Scaler übergeben, erstelle einen neuen
    if scaler is None:
        scaler = StandardScaler()
    
    # Fit + Transform (wenn fit=True) oder nur Transform
    if fit:
        # Fit auf diesen Features und transform direkt
        X_array = scaler.fit_transform(X)
    else:
        # Nur transform mit existierenden Scaler
        X_array = scaler.transform(X)
    
    # Konvertiere zurück zu Pandas DataFrame
    X_normalized = pd.DataFrame(
        data=X_array,
        columns=X.columns,
        index=X.index
    )
    
    return X_normalized, scaler
