"""
=============================================================================
KEDRO PIPELINE: Nodes (Wrapper um src/core/ Funktionen)
=============================================================================

"""

import os
import pandas as pd
import numpy as np
import yaml
from typing import Tuple, Dict, Any, List

# Importiere framework-agnostische Core-Module
from src.core.data_processing import load_breast_cancer, clean_data, normalize_features
from src.core.feature_engineering import (
    preprocess_features,
    train_test_split,
    encode_target
)
from src.core.modeling import train_model, predict, save_model, load_model
from src.core.evaluation import (
    evaluate,
    compute_confusion_matrix,
    save_metrics,
    get_classification_report
)


# =============================================================================
# NODE 1: Load Data
# =============================================================================

def node_load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    KEDRO NODE: Lädt Breast Cancer Dataset.
    
    Outputs (an Kedro Catalog):
    --------
    raw_breast_cancer : pd.DataFrame
        Raw Features (569, 30)
    
    y : pd.Series
        Raw Target (569,)
    
    In der pipeline.py:
      pipeline += node(
          func=node_load_data,
          inputs=None,
          outputs=["raw_breast_cancer", "y"]
      )
    """
    print("Loading raw breast cancer data...")
    X, y = load_breast_cancer()
    print(f"✓ Loaded X shape: {X.shape}, y shape: {y.shape}")
    return X, y


# =============================================================================
# NODE 2: Clean Data
# =============================================================================

def node_clean_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    KEDRO NODE: Bereinigt die Daten.
    
    Inputs (vom Catalog):
    -----
    X : pd.DataFrame
        Raw Features
    
    y : pd.Series
        Raw Target
    
    Outputs (an Catalog):
    --------
    cleaned_data : pd.DataFrame
        Bereinigte Features
    
    y : pd.Series
        Target
    """
    print("Cleaning data...")
    X_clean = clean_data(X)
    print(f"✓ Cleaned X shape: {X_clean.shape}")
    return X_clean, y


# =============================================================================
# NODE 3: Normalize Features
# =============================================================================

def node_normalize_features(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, object]:
    """
    KEDRO NODE: Normalisiert Features mit StandardScaler.
    
    Inputs:
    -------
    X : pd.DataFrame
        Bereinigte Features
    
    y : pd.Series
        Target
    
    Outputs:
    --------
    normalized_data : pd.DataFrame
        Normalisierte Features
    
    y : pd.Series
        Target (unverändert, weitergegeben)
    
    feature_scaler : object
        Der fitted Scaler (für später auf Test-Daten)
    """
    print("Normalizing features...")
    X_normalized, scaler = normalize_features(X, fit=True)
    print(f"✓ Normalized X shape: {X_normalized.shape}")
    return X_normalized, y, scaler


# =============================================================================
# NODE 4: Preprocess Features (Feature Selection)
# =============================================================================

def node_preprocess_features(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: Dict
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    KEDRO NODE: Feature Selection mit SelectKBest.
    
    Inputs:
    -------
    X : pd.DataFrame
        Normalisierte Features
    
    y : pd.Series
        Target
    
    parameters : Dict
        Von Kedro Config geladen (aus parameters.yml)
        Enthält: n_features_to_select, test_size, random_state
    
    Outputs:
    --------
    preprocessed_data : pd.DataFrame
        Features nach SelectKBest (569, 20)
    
    y : pd.Series
        Target (unverändert)

    """
    print("Feature selection (SelectKBest)...")
    X_processed, feature_names = preprocess_features(
        X, y,
        n_features=parameters['feature_engineering']['n_features_to_select'],
        random_state=parameters['feature_engineering']['random_state']
    )
    print(f"✓ Selected {X_processed.shape[1]} features")
    return X_processed, y


# =============================================================================
# NODE 5: Train/Test Split
# =============================================================================

def node_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    KEDRO NODE: Deterministische Train/Test Aufteilung.
    
    Inputs:
    -------
    X : pd.DataFrame
        Prozessierte Features
    
    y : pd.Series
        Target
    
    parameters : Dict
        Config mit test_size und random_state
    
    Outputs:
    --------
    X_train, X_test, y_train, y_test : DataFrames/Series
        Die 4 Split-Datasets

    """
    print("Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters['feature_engineering']['test_size'],
        random_state=parameters['feature_engineering']['random_state']
    )
    print(f"✓ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# NODE 6: Train Model
# =============================================================================

def node_train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict
) -> object:
    """
    KEDRO NODE: Trainiert ein Klassifikator-Modell.
    
    Inputs:
    -------
    X_train : pd.DataFrame
        Training Features
    
    y_train : pd.Series
        Training Target
    
    parameters : Dict
        Model Hyperparameter aus Config
    
    Outputs:
    --------
    trained_model : object
        Das trainierte Modell (saved as joblib im Catalog)

    """
    print("Training model...")
    model = train_model(
        X_train, y_train,
        algorithm=parameters['model']['algorithm'],
        params=parameters['model'].get('random_forest', {})
    )
    print(f"✓ Model trained")
    return model


# =============================================================================
# NODE 7: Make Predictions
# =============================================================================

def node_predict(
    model: object,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    KEDRO NODE: Macht Predictions auf Test-Set.
    
    Inputs:
    -------
    model : object
        Das trainierte Modell
    
    X_test : pd.DataFrame
        Test Features
    
    Outputs:
    --------
    predictions : pd.DataFrame
        Predictions mit y_pred und y_pred_proba
    """
    print("Making predictions...")
    y_pred = predict(model, X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    predictions_df = pd.DataFrame({
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    print(f"✓ Predictions made: {predictions_df.shape}")
    return predictions_df


# =============================================================================
# NODE 8: Evaluate Model
# =============================================================================

def node_evaluate(
    y_test: pd.Series,
    predictions: pd.DataFrame,
    model: object
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    KEDRO NODE: Evaluiert das Modell.
    
    Inputs:
    -------
    y_test : pd.Series
        Echte Test Labels
    
    predictions : pd.DataFrame
        Predictions (mit y_pred und y_pred_proba)
    
    model : object
        Das trainierte Modell (für Feature Importance evtl.)
    
    Outputs:
    --------
    metrics : Dict[str, float]
        Evaluierungs-Metriken (saved as JSON im Catalog)
    
    confusion_matrix : pd.DataFrame
        Die Confusion Matrix (saved as CSV im Catalog)

    """
    print("Evaluating model...")
    
    y_pred = predictions['y_pred'].values
    y_pred_proba = predictions['y_pred_proba'].values
    
    metrics = evaluate(y_test.values, y_pred, y_pred_proba)
    cm = compute_confusion_matrix(y_test.values, y_pred)
    
    # Konvertiere CM zu DataFrame für CSV-Speicherung
    cm_df = pd.DataFrame(
        cm,
        index=['Predicted 0', 'Predicted 1'],
        columns=['Actual 0', 'Actual 1']
    )
    
    print(f"✓ Evaluation complete")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    
    return metrics, cm_df
