"""
=============================================================================
KEDRO PIPELINE: Nodes (Wrapper um icw.core/ Funktionen)
=============================================================================
"""

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml

# Importiere framework-agnostische Core-Module
from icw.core.data_processing import clean_data, load_breast_cancer, normalize_features
from icw.core.evaluation import (
    compute_confusion_matrix,
    evaluate,
    get_classification_report,
    save_metrics,
)
from icw.core.feature_engineering import (
    encode_target,
    preprocess_features,
    train_test_split,
)
from icw.core.modeling import load_model, predict, save_model, train_model


# =============================================================================
# NODE 1: Load Data
# =============================================================================

def node_load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """KEDRO NODE: Lädt Breast Cancer Dataset."""
    print("Loading raw breast cancer data...")
    X, y = load_breast_cancer()
    print(f"✓ Loaded X shape: {X.shape}, y shape: {y.shape}")
    return X, y


# =============================================================================
# NODE 2: Clean Data
# =============================================================================

def node_clean_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """KEDRO NODE: Bereinigt die Daten und gibt Features sowie Target zurück."""
    print("Cleaning data...")
    X_clean = clean_data(X)
    print(f"✓ Cleaned X shape: {X_clean.shape}")
    return X_clean, y


# =============================================================================
# NODE 3: Normalize Features (angepasst: arbeitet nur auf X)
# =============================================================================

def node_normalize_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """
    KEDRO NODE: Normalisiert Features mit StandardScaler.

    Inputs:
    -------
    X : pd.DataFrame (cleaned_data)

    Outputs:
    --------
    normalized_data : pd.DataFrame
    feature_scaler : object
    """
    print("Normalizing features...")
    X_normalized, scaler = normalize_features(X, fit=True)
    print(f"✓ Normalized X shape: {X_normalized.shape}")
    return X_normalized, scaler


# =============================================================================
# NODE 4: Preprocess Features (angepasst: gibt nur transformiertes X zurück)
# =============================================================================

def node_preprocess_features(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    KEDRO NODE: Feature Selection mit SelectKBest.

    Inputs:
    -------
    X : pd.DataFrame (normalized_data)
    y : pd.Series (cleaned_y)
    parameters : Dict

    Outputs:
    --------
    preprocessed_data : pd.DataFrame
    """
    print("Feature selection (SelectKBest)...")
    X_processed, feature_names = preprocess_features(
        X, y,
        n_features=parameters['feature_engineering']['n_features_to_select'],
        random_state=parameters['feature_engineering']['random_state']
    )
    print(f"✓ Selected {X_processed.shape[1]} features")
    return X_processed


# =============================================================================
# NODE 5: Train/Test Split
# =============================================================================

def node_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """KEDRO NODE: Deterministische Train/Test Aufteilung."""
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
    parameters: Dict[str, Any]
) -> object:
    """KEDRO NODE: Trainiert ein Klassifikator-Modell."""
    print("Training model...")
    model = train_model(
        X_train, y_train,
        algorithm=parameters['model']['algorithm'],
        params=parameters['model'].get('random_forest', {})
    )
    print("✓ Model trained")
    return model


# =============================================================================
# NODE 7: Make Predictions
# =============================================================================

def node_predict(
    model: object,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """KEDRO NODE: Erstellt Predictions auf dem Test-Set."""
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
    """KEDRO NODE: Evaluiert das Modell."""
    print("Evaluating model...")

    y_pred = predictions['y_pred'].values
    y_pred_proba = predictions['y_pred_proba'].values

    metrics = evaluate(y_test.values, y_pred, y_pred_proba)
    cm = compute_confusion_matrix(y_test.values, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=['Predicted 0', 'Predicted 1'],
        columns=['Actual 0', 'Actual 1']
    )

    print("✓ Evaluation complete")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")

    return metrics, cm_df