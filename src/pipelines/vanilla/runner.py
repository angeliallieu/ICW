"""
=============================================================================
VANILLA PIPELINE: Minimale, Orchestrierungs-Framework-freie ML-Pipeline
=============================================================================

Ausführung: python -m src.pipelines.vanilla.runner

Ablauf:
1. Konfiguration laden
2. Daten laden
3. Daten bereinigen
4. Features normalisieren
5. Feature Engineering (Selection, Split)
6. Modell trainieren
7. Predictions machen
8. Evaluierung
9. Artefakte speichern

"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Importiere die framework-agnostischen Core-Module
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
# HELPER FUNKTIONEN
# =============================================================================

def load_config(config_path: str = "config/parameters.yml") -> dict:
    """
    Lädt die YAML Konfigurationsdatei.
    
    Parameters:
    -----------
    config_path : str
        Pfad zur parameters.yml
    
    Returns:
    --------
    config : dict
        Geparste YAML als Dictionary

    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_directories(config: dict) -> None:
    """
    Erstellt alle benötigten Output-Verzeichnisse.
    """
    paths = config['paths']
    for key, path in paths.items():
        if key.endswith('_dir') and isinstance(path, str):
            os.makedirs(path, exist_ok=True)


def print_step(step_num: int, step_name: str) -> None:
    """Druckt einen schönen Schritt-Header"""
    print(f"\n{'='*80}")
    print(f"SCHRITT {step_num}: {step_name}")
    print(f"{'='*80}")


def print_data_info(X: pd.DataFrame, y: pd.Series = None, label: str = "") -> None:
    """Druckt Informationen über einen Dataset"""
    print(f"{label}:")
    print(f"  Shape X: {X.shape}")
    if y is not None:
        print(f"  Shape y: {y.shape}")
        print(f"  Class distribution: {dict(y.value_counts())}")
    print()


# =============================================================================
# MAINPIPELINE
# =============================================================================

def main():
    """
    Hauptfunktion: Führt die komplette Pipeline sequenziell aus.
    
    Workflow:
    1. Laden: Daten von sklearn laden
    2. Clean:  NaN und Duplikate entfernen
    3. Normal: Features normalisieren
    4. Feature Eng: Features selektieren, Train/Test splitten
    5. Train: Modell auf Training-Daten trainieren
    6. Predict: Predictions auf Test-Daten machen
    7. Evaluate: Metriken berechnen
    8. Save: Modell und Metriken speichern
    """
    
    print("\n" + "="*80)
    print("VANILLA ML-PIPELINE: Breast Cancer Classification")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # SCHRITT 1: KONFIGURATION LADEN
    # =========================================================================
    print_step(1, "Konfiguration laden")
    config = load_config("config/parameters.yml")
    ensure_directories(config)
    print(f"✓ Konfiguration geladen von config/parameters.yml")
    print(f"  - Model Algorithmus: {config['model']['algorithm']}")
    print(f"  - Test Size: {config['feature_engineering']['test_size']}")
    print(f"  - Random State: {config['feature_engineering']['random_state']}")
    
    # =========================================================================
    # SCHRITT 2: DATEN LADEN
    # =========================================================================
    print_step(2, "Daten laden (Breast Cancer Dataset)")
    X, y = load_breast_cancer()
    print(f"✓ Dataset geladen von sklearn.datasets")
    print_data_info(X, y, "Raw Dataset")
    
    # Speichere Raw Data für Reproduzierbarkeit
    os.makedirs(config['paths']['raw_dir'], exist_ok=True)
    X.to_csv(
        os.path.join(config['paths']['raw_dir'], 'breast_cancer_raw.csv'),
        index=False
    )
    y.to_csv(
        os.path.join(config['paths']['raw_dir'], 'y_raw.csv'),
        index=False
    )
    print(f"✓ Raw Data gespeichert zu {config['paths']['raw_dir']}/")
    
    # =========================================================================
    # SCHRITT 3: DATEN BEREINIGEN
    # =========================================================================
    print_step(3, "Daten bereinigen")
    X_clean = clean_data(X)
    print(f"✓ Data Cleaning abgeschlossen")
    print_data_info(X_clean, label="Nach Cleaning")
    
    # Speichere bereinigte Daten
    X_clean.to_csv(
        os.path.join(config['paths']['intermediate_dir'], 'breast_cancer_cleaned.csv'),
        index=False
    )
    print(f"✓ Bereinigte Data gespeichert")
    
    # =========================================================================
    # SCHRITT 4: FEATURES NORMALISIEREN
    # =========================================================================
    print_step(4, "Features normalisieren (StandardScaler)")
    X_normalized, scaler = normalize_features(
        X_clean,
        fit=True  # Fit den Scaler auf den gesamten Daten
    )
    print(f"✓ Normalisierung abgeschlossen")
    print(f"  - Feature mean: {X_normalized.mean().mean():.6f} (sollte ~0)")
    print(f"  - Feature std: {X_normalized.std().mean():.6f} (sollte ~1)")
    print_data_info(X_normalized, label="Nach Normalisierung")
    
    # Speichere normalisierte Features
    X_normalized.to_csv(
        os.path.join(config['paths']['intermediate_dir'], 'breast_cancer_normalized.csv'),
        index=False
    )
    print(f"✓ Normalisierte Data gespeichert")
    
    # =========================================================================
    # SCHRITT 5: FEATURE ENGINEERING
    # =========================================================================
    print_step(5, "Feature Engineering (SelectKBest)")
    X_processed, selected_features = preprocess_features(
        X_normalized,
        y,
        n_features=config['feature_engineering']['n_features_to_select'],
        random_state=config['feature_engineering']['random_state']
    )
    print(f"✓ Feature-Selection abgeschlossen")
    print(f"  - Features selected: {len(selected_features)} von {X_normalized.shape[1]}")
    print(f"  - Top 5 Features: {selected_features[:5]}")
    print_data_info(X_processed, label="Nach Feature Selection")
    
    # Speichere prozessierte Features
    X_processed.to_csv(
        os.path.join(config['paths']['intermediate_dir'], 'breast_cancer_processed.csv'),
        index=False
    )
    print(f"✓ Prozessierte Data gespeichert")
    
    # =========================================================================
    # SCHRITT 6: DETERMINISTISCHE TRAIN/TEST SPLIT
    # =========================================================================
    print_step(6, "Train/Test Split (Deterministic mit random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=config['feature_engineering']['test_size'],
        random_state=config['feature_engineering']['random_state']
    )
    print(f"✓ Train/Test Split abgeschlossen")
    print_data_info(X_train, y_train, "Training Set")
    print_data_info(X_test, y_test, "Test Set")
    
    # Speichere Train/Test Sets
    X_train.to_csv(
        os.path.join(config['paths']['primary_dir'], 'X_train.csv'),
        index=False
    )
    X_test.to_csv(
        os.path.join(config['paths']['primary_dir'], 'X_test.csv'),
        index=False
    )
    y_train.to_csv(
        os.path.join(config['paths']['primary_dir'], 'y_train.csv'),
        index=False
    )
    y_test.to_csv(
        os.path.join(config['paths']['primary_dir'], 'y_test.csv'),
        index=False
    )
    print(f"✓ Train/Test Sets gespeichert zu {config['paths']['primary_dir']}/")
    
    # =========================================================================
    # SCHRITT 7: MODELLTRAINING
    # =========================================================================
    print_step(7, "Modell trainieren (Random Forest)")
    model = train_model(
        X_train, y_train,
        algorithm=config['model']['algorithm'],
        params=config['model'].get('random_forest', {})
    )
    print(f"✓ Modell trainiert")
    print(f"  - Algorithm: {config['model']['algorithm']}")
    print(f"  - n_estimators: {model.n_estimators}")
    print(f"  - max_depth: {model.max_depth}")
    
    # Train-Accuracy (zur Kontrolle von Overfitting)
    train_pred = predict(model, X_train)
    train_accuracy = (train_pred == y_train).mean()
    print(f"  - Training Accuracy: {train_accuracy:.4f}")
    
    # =========================================================================
    # SCHRITT 8: PREDICTIONS AUF TEST-SET
    # =========================================================================
    print_step(8, "Predictions machen")
    y_pred = predict(model, X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability für positive Klasse
    print(f"✓ Predictions abgeschlossen")
    print(f"  - Predictions shape: {y_pred.shape}")
    print(f"  - Unique predictions: {np.unique(y_pred)}")
    print(f"  - Prediction distribution: {dict(pd.Series(y_pred).value_counts())}")
    
    # =========================================================================
    # SCHRITT 9: EVALUIERUNG
    # =========================================================================
    print_step(9, "Evaluierung")
    metrics = evaluate(y_test, y_pred, y_pred_proba)
    cm = compute_confusion_matrix(y_test, y_pred)
    
    print(f"✓ Metriken berechnet:")
    for metric_name, metric_value in metrics.items():
        print(f"  - {metric_name.upper()}: {metric_value:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    print(f"\nDetailed Classification Report:")
    report = get_classification_report(y_test, y_pred)
    print(report)
    
    # =========================================================================
    # SCHRITT 10: ARTEFAKTE SPEICHERN
    # =========================================================================
    print_step(10, "Artefakte speichern")
    
    # Speichere Modell
    model_path = os.path.join(
        config['paths']['models_dir'],
        config['paths']['model_filename']
    )
    save_model(model, model_path)
    
    # Speichere Metriken, Confusion Matrix, Predictions
    save_metrics(
        metrics=metrics,
        cm=cm,
        y_pred=y_pred,
        y_true=y_test,
        filepath_metrics=os.path.join(
            config['paths']['metrics_dir'],
            config['paths']['metrics_filename']
        ),
        filepath_cm=os.path.join(
            config['paths']['metrics_dir'],
            config['paths']['confusion_matrix_filename']
        ),
        filepath_predictions=os.path.join(
            config['paths']['metrics_dir'],
            config['paths']['predictions_filename']
        )
    )
    
    # =========================================================================
    # ABSCHLUSS
    # =========================================================================
    print_step(11, "Pipeline abgeschlossen")
    print(f"\n✓ Vanilla Pipeline erfolgreich abgeschlossen!")
    print(f"✓ Alle Artefakte gespeichert in:")
    print(f"  - Models: {config['paths']['models_dir']}/")
    print(f"  - Metrics: {config['paths']['metrics_dir']}/")
    print(f"  - Data: {config['paths']['primary_dir']}/")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ FEHLER in Vanilla Pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
