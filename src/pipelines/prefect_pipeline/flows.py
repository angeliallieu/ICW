"""
=============================================================================
PREFECT PIPELINE
=============================================================================

"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Prefect Imports
from prefect import flow, task, get_run_logger

# Core ML Imports (IDENTISCH zu Vanilla & Kedro!)
from src.core.data_processing import load_breast_cancer, clean_data, normalize_features
from src.core.feature_engineering import (
    preprocess_features,
    train_test_split,
    encode_target
)
from src.core.modeling import train_model, predict, save_model
from src.core.evaluation import (
    evaluate,
    compute_confusion_matrix,
    save_metrics,
    get_classification_report
)


# =============================================================================
# HELPERS: Konfiguration & Logging
# =============================================================================

def load_config(config_path: str = "config/parameters.yml") -> dict:
    """Lädt YAML-Konfiguration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict) -> None:
    """Erstellt alle Output-Verzeichnisse"""
    paths = config['paths']
    for key, path in paths.items():
        if key.endswith('_dir') and isinstance(path, str):
            os.makedirs(path, exist_ok=True)


# =============================================================================
# PREFECT TASKS (8 Tasks für den ML-Workflow)
# =============================================================================

@task(
    name="Load Data",
    # description="Lädt Breast Cancer Dataset von lokaler CSV",
    # tags=["data-loading", "io"]
)
def task_load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    TASK 1: Lädt die Breast Cancer CSV.
    
    Returns:
        X : Feature-Matrix (569, 30)
        y : Target-Vektor (569,)

        - @task dekoriert eine Funktion als Prefect Task
        - Jede Task wird unabhängig ausgeführt
        - Tasks können parallel laufen
    """
    logger = get_run_logger()
    logger.info("Loading breast cancer data from CSV...")
    
    X, y = load_breast_cancer()
    
    logger.info(f"✓ Data loaded: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


@task(
    name="Clean Data",
    # description="Entfernt NaN und Duplikate",
    # tags=["data-cleaning"]
)
def task_clean_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    TASK 2: Data Cleaning.
    
    Parameters:
        X : Feature-Matrix
        y : Target
    
    Returns:
        X_clean : Bereinigte Features
        y : Target (unverändert)
    """
    logger = get_run_logger()
    logger.info("Cleaning data...")
    
    X_clean = clean_data(X)
    
    logger.info(f"✓ Data cleaned: X.shape={X_clean.shape}")
    return X_clean, y


@task(
    name="Normalize Features",
    # description="Normalisiert Features mit StandardScaler",
    # tags=["preprocessing"]
)
def task_normalize_features(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, object]:
    """
    TASK 3: Feature Normalisierung.
    
    Returns:
        X_normalized : Normalisierte Features
        y : Target
        scaler : Der fitted Scaler
    """
    logger = get_run_logger()
    logger.info("Normalizing features...")
    
    X_normalized, scaler = normalize_features(X, fit=True)
    
    logger.info(f"✓ Features normalized: mean={X_normalized.mean().mean():.6f}, "
                f"std={X_normalized.std().mean():.6f}")
    return X_normalized, y, scaler


@task(
    name="Preprocess Features",
    # description="Feature Selection mit SelectKBest",
    # tags=["feature-engineering"]
)
def task_preprocess_features(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    TASK 4: Feature Engineering (SelectKBest).
    
    Parameters:
        X : Normalized Features
        y : Target
        config : Konfiguration (mit n_features_to_select)
    
    Returns:
        X_processed : Features nach SelectKBest
        y : Target
    """
    logger = get_run_logger()
    logger.info("Preprocessing features (SelectKBest)...")
    
    X_processed, selected_features = preprocess_features(
        X, y,
        n_features=config['feature_engineering']['n_features_to_select'],
        random_state=config['feature_engineering']['random_state']
    )
    
    logger.info(f"✓ Features selected: {X_processed.shape[1]} of {X.shape[1]}")
    return X_processed, y


@task(
    name="Train/Test Split",
    # description="Deterministische Aufteilung mit random_state=42",
    # tags=["data-splitting"]
)
def task_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    TASK 5: Train/Test Split.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger = get_run_logger()
    logger.info("Splitting data (Train/Test)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['feature_engineering']['test_size'],
        random_state=config['feature_engineering']['random_state']
    )
    
    logger.info(f"✓ Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


@task(
    name="Train Model",
    # description="Trainiert Random Forest Klassifikator",
    # tags=["modeling"]
)
def task_train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict
) -> object:
    """
    TASK 6: Model Training.
    
    Returns:
        model : Trainierter Random Forest

        - Modelltraining ist der "Core" des ML
        - random_state für Reproduzierbarkeit ESSENTIELL
    """
    logger = get_run_logger()
    logger.info("Training model (Random Forest)...")
    
    model = train_model(
        X_train, y_train,
        algorithm=config['model']['algorithm'],
        params=config['model'].get('random_forest', {})
    )
    
    # Training Accuracy
    train_pred = model.predict(X_train)
    train_accuracy = (train_pred == y_train).mean()
    logger.info(f"✓ Model trained: Training Accuracy={train_accuracy:.4f}")
    
    return model


@task(
    name="Make Predictions",
    # description="Predictions auf Test-Set",
    # tags=["prediction"]
)
def task_predict(model: object, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 7: Predictions.
    
    Returns:
        predictions_df : DataFrame mit y_pred und y_pred_proba
    """
    logger = get_run_logger()
    logger.info("Making predictions...")
    
    y_pred = predict(model, X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    predictions_df = pd.DataFrame({
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    
    logger.info(f"✓ Predictions made: {predictions_df.shape[0]} samples")
    return predictions_df


@task(
    name="Evaluate Model",
    # description="Berechnet Metriken und speichert Artefakte",
    # tags=["evaluation"]
)
def task_evaluate(
    y_test: pd.Series,
    predictions: pd.DataFrame,
    model: object,
    config: dict
) -> Tuple[dict, pd.DataFrame, str]:
    """
    TASK 8: Model Evaluation.
    
    Returns:
        metrics : Dict mit Accuracy, F1, etc.
        cm_df : Confusion Matrix as DataFrame
        report : Classification Report as String
    """
    logger = get_run_logger()
    logger.info("Evaluating model...")
    
    y_pred = predictions['y_pred'].values
    y_pred_proba = predictions['y_pred_proba'].values
    
    # Metriken berechnen
    metrics = evaluate(y_test.values, y_pred, y_pred_proba)
    cm = compute_confusion_matrix(y_test.values, y_pred)
    report = get_classification_report(y_test.values, y_pred)
    
    # Confusion Matrix als DataFrame
    cm_df = pd.DataFrame(
        cm,
        index=['Predicted 0', 'Predicted 1'],
        columns=['Actual 0', 'Actual 1']
    )
    
    logger.info(f"✓ Evaluation complete:")
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - F1-Score: {metrics['f1']:.4f}")
    logger.info(f"  - Precision: {metrics['precision']:.4f}")
    logger.info(f"  - Recall: {metrics['recall']:.4f}")
    
    return metrics, cm_df, report


@task(
    name="Save Artifacts",
    # description="Speichert Modell und Metriken",
    # tags=["io", "artifacts"]
)
def task_save_artifacts(
    model: object,
    metrics: dict,
    cm_df: pd.DataFrame,
    report: str,
    config: dict
) -> None:
    """
    TASK 9: Speichert alle Artefakte (Optional Prefect-native Alternative).
    """
    logger = get_run_logger()
    logger.info("Saving artifacts...")
    
    # Speichere Modell
    model_path = os.path.join(
        config['paths']['models_dir'],
        config['paths']['model_filename']
    )
    save_model(model, model_path)
    
    # Speichere Metriken
    metrics_path = os.path.join(
        config['paths']['metrics_dir'],
        config['paths']['metrics_filename']
    )
    os.makedirs(config['paths']['metrics_dir'], exist_ok=True)
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"✓ Metrics saved to {metrics_path}")
    
    # Speichere Confusion Matrix
    cm_path = os.path.join(
        config['paths']['metrics_dir'],
        config['paths']['confusion_matrix_filename']
    )
    cm_df.to_csv(cm_path)
    logger.info(f"✓ Confusion Matrix saved to {cm_path}")
    
    # Speichere Report
    report_path = os.path.join(
        config['paths']['metrics_dir'],
        'classification_report.txt'
    )
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"✓ Report saved to {report_path}")


# =============================================================================
# PREFECT FLOW (Orchestriert die Tasks)
# =============================================================================

@flow(
    name="Breast Cancer ML Pipeline (Prefect)"
)
def breast_cancer_pipeline_flow() -> dict:
    """
    MAIN FLOW: Orchestriert alle 9 Tasks.

    """
    logger = get_run_logger()
    
    print("\n" + "="*80)
    print("PREFECT ML-PIPELINE: Breast Cancer Classification")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Lade Konfiguration
    config = load_config("config/parameters.yml")
    ensure_directories(config)
    
    # TASK 1: Load Data
    logger.info("SCHRITT 1: Konfiguration laden")
    logger.info(f"✓ Konfiguration geladen")
    logger.info(f"  - Model: {config['model']['algorithm']}")
    logger.info(f"  - Test Size: {config['feature_engineering']['test_size']}")
    
    logger.info("\nSCHRITT 2: Daten laden")
    X, y = task_load_data()
    
    # TASK 2: Clean Data
    logger.info("\nSCHRITT 3: Daten bereinigen")
    X_clean, y = task_clean_data(X, y)
    
    # TASK 3: Normalize Features
    logger.info("\nSCHRITT 4: Features normalisieren")
    X_norm, y, scaler = task_normalize_features(X_clean, y)
    
    # TASK 4: Preprocess Features
    logger.info("\nSCHRITT 5: Feature Engineering")
    X_proc, y = task_preprocess_features(X_norm, y, config)
    
    # TASK 5: Train/Test Split
    logger.info("\nSCHRITT 6: Train/Test Split")
    X_train, X_test, y_train, y_test = task_train_test_split(X_proc, y, config)
    
    # TASK 6: Train Model
    logger.info("\nSCHRITT 7: Modell trainieren")
    model = task_train_model(X_train, y_train, config)
    
    # TASK 7: Make Predictions
    logger.info("\nSCHRITT 8: Predictions machen")
    predictions = task_predict(model, X_test)
    
    # TASK 8: Evaluate
    logger.info("\nSCHRITT 9: Evaluierung")
    metrics, cm_df, report = task_evaluate(y_test, predictions, model, config)
    
    # TASK 9: Save Artifacts
    logger.info("\nSCHRITT 10: Artefakte speichern")
    task_save_artifacts(model, metrics, cm_df, report, config)
    
    # Abschluss
    logger.info("\n" + "="*80)
    logger.info("✓ Prefect Pipeline erfolgreich abgeschlossen!")
    logger.info(f"✓ Artefakte gespeichert in:")
    logger.info(f"  - Models: {config['paths']['models_dir']}/")
    logger.info(f"  - Metrics: {config['paths']['metrics_dir']}/")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")

    return {
        'model': model,
        'metrics': metrics,
        'cm': cm_df,
        'report': report,
        'config': config,
        'comparison': {
            'y_true': y_test.to_numpy().tolist(),
            'y_pred': predictions['y_pred'].to_numpy().tolist(),
            'y_pred_proba': predictions['y_pred_proba'].to_numpy().tolist(),
    },
}

# =============================================================================
# LOCAL EXECUTION (für Testing & Development)
# =============================================================================

if __name__ == "__main__":
    result = breast_cancer_pipeline_flow()
    print(f"\nPipeline Result: {result}")
