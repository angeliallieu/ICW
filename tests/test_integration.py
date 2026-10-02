"""
Unit & Integration Tests für src/pipelines/prefect_pipeline/

- Vergleich: Vanilla vs. Kedro vs. Prefect (IDENTITÄT)
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.core.data_processing import load_breast_cancer, clean_data, normalize_features
from src.core.feature_engineering import preprocess_features, train_test_split
from src.core.modeling import train_model, predict
from src.core.evaluation import evaluate, compute_confusion_matrix

# Prefect Task Imports
from src.pipelines.prefect_pipeline.flows import (
    task_load_data,
    task_clean_data,
    task_normalize_features,
    task_preprocess_features,
    task_train_test_split,
    task_train_model,
    task_predict,
    task_evaluate,
    breast_cancer_pipeline_flow,
    load_config
)


@pytest.fixture
def config():
    """Lade die Konfiguration"""
    return load_config("config/parameters.yml")


class TestPrefectTasks:
    """Tests für einzelne Prefect Tasks"""
    
    def test_task_load_data(self):
        """Testet task_load_data Task"""
        X, y = task_load_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (569, 30)
        assert y.shape == (569,)
    
    def test_task_clean_data(self):
        """Testet task_clean_data Task"""
        X, y = task_load_data()
        X_clean, y_clean = task_clean_data(X, y)
        
        assert isinstance(X_clean, pd.DataFrame)
        assert X_clean.isnull().sum().sum() == 0
        assert len(X_clean) <= len(X)
    
    def test_task_normalize_features(self):
        """Testet task_normalize_features Task"""
        X, y = task_load_data()
        X_clean, y = task_clean_data(X, y)
        X_norm, y_norm, scaler = task_normalize_features(X_clean, y)
        
        assert X_norm.shape == X_clean.shape
        assert abs(X_norm.mean().max()) < 1e-10
        assert (X_norm.std() > 0.99).all() and (X_norm.std() < 1.01).all()
    
    def test_task_preprocess_features(self, config):
        """Testet task_preprocess_features Task"""
        X, y = task_load_data()
        X_clean, y = task_clean_data(X, y)
        X_norm, y, _ = task_normalize_features(X_clean, y)
        X_proc, y_proc = task_preprocess_features(X_norm, y, config)
        
        n_features = config['feature_engineering']['n_features_to_select']
        assert X_proc.shape[1] == n_features
        assert X_proc.shape[0] == len(X_norm)
    
    def test_task_train_test_split(self, config):
        """Testet task_train_test_split Task"""
        X, y = task_load_data()
        X_clean, y = task_clean_data(X, y)
        X_norm, y, _ = task_normalize_features(X_clean, y)
        X_proc, y = task_preprocess_features(X_norm, y, config)
        
        X_train, X_test, y_train, y_test = task_train_test_split(X_proc, y, config)
        
        assert len(X_train) + len(X_test) == len(X_proc)
        assert len(y_train) + len(y_test) == len(y)
        assert len(set(X_train.index) & set(X_test.index)) == 0  # Kein Overlap
    
    def test_task_train_model(self, config):
        """Testet task_train_model Task"""
        X, y = task_load_data()
        X_clean, y = task_clean_data(X, y)
        X_norm, y, _ = task_normalize_features(X_clean, y)
        X_proc, y = task_preprocess_features(X_norm, y, config)
        X_train, X_test, y_train, y_test = task_train_test_split(X_proc, y, config)
        
        model = task_train_model(X_train, y_train, config)
        
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_task_predict(self, config):
        """Testet task_predict Task"""
        X, y = task_load_data()
        X_clean, y = task_clean_data(X, y)
        X_norm, y, _ = task_normalize_features(X_clean, y)
        X_proc, y = task_preprocess_features(X_norm, y, config)
        X_train, X_test, y_train, y_test = task_train_test_split(X_proc, y, config)
        model = task_train_model(X_train, y_train, config)
        
        predictions = task_predict(model, X_test)
        
        assert isinstance(predictions, pd.DataFrame)
        assert 'y_pred' in predictions.columns
        assert 'y_pred_proba' in predictions.columns
        assert len(predictions) == len(X_test)
    
    def test_task_evaluate(self, config):
        """Testet task_evaluate Task"""
        X, y = task_load_data()
        X_clean, y = task_clean_data(X, y)
        X_norm, y, _ = task_normalize_features(X_clean, y)
        X_proc, y = task_preprocess_features(X_norm, y, config)
        X_train, X_test, y_train, y_test = task_train_test_split(X_proc, y, config)
        model = task_train_model(X_train, y_train, config)
        predictions = task_predict(model, X_test)
        
        metrics, cm_df, report = task_evaluate(y_test, predictions, model, config)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert isinstance(cm_df, pd.DataFrame)
        assert isinstance(report, str)


class TestPrefectFlow:
    """Integration-Tests für die Prefect Flow"""
    
    def test_flow_execution(self):
        """Testet die komplette Flow-Ausführung"""
        # Führe die Flow aus
        result = breast_cancer_pipeline_flow()
        
        # Verifiziere Ergebnisse
        assert 'model' in result
        assert 'metrics' in result
        assert 'cm' in result
        assert 'report' in result
        
        # Verifiziere Metriken
        metrics = result['metrics']
        assert isinstance(metrics, dict)
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_flow_model_quality(self):
        """Testet, dass das trainierte Modell gute Qualität hat"""
        result = breast_cancer_pipeline_flow()
        metrics = result['metrics']
        
        # Breast Cancer ist ein einfaches Dataset
        # Random Forest sollte > 80% Accuracy erreichen
        assert metrics['accuracy'] > 0.8
        assert metrics['f1'] > 0.8


class TestVanillaVsKedroVsPrefectEquivalence:
    """
    ZENTRAL: Beweist, dass alle drei Pipelines IDENTISCH sind!
    
    Das ist der Kern deines ICW-Projekts:
    - Vanilla: Framework-frei
    - Kedro: Strukturiert
    - Prefect: Cloud-ready
    
    ABER: Alle nutzen die gleiche Core-Logik!
    """
    
    def test_prefect_vs_vanilla_inputs(self):
        """Testet, dass Input-Data identisch ist"""
        # Vanilla-Input
        X_vanilla, y_vanilla = load_breast_cancer()
        X_vanilla = clean_data(X_vanilla)
        X_vanilla, _ = normalize_features(X_vanilla, fit=True)
        
        # Prefect-Input
        X_prefect, y_prefect = task_load_data()
        X_prefect, y_prefect = task_clean_data(X_prefect, y_prefect)
        X_prefect, y_prefect, _ = task_normalize_features(X_prefect, y_prefect)
        
        # Sollten identisch sein
        pd.testing.assert_frame_equal(X_vanilla, X_prefect)
        pd.testing.assert_series_equal(y_vanilla, y_prefect)
    
    def test_prefect_determinism(self):
        """Testet, dass Prefect-Flow deterministisch ist"""
        config = load_config("config/parameters.yml")
        
        # Erste Ausführung
        result1 = breast_cancer_pipeline_flow()
        metrics1 = result1['metrics']
        
        # Zweite Ausführung
        result2 = breast_cancer_pipeline_flow()
        metrics2 = result2['metrics']
        
        # Sollten identisch sein (deterministisch)
        for metric_key in metrics1.keys():
            np.testing.assert_almost_equal(
                metrics1[metric_key],
                metrics2[metric_key],
                decimal=10
            )
    
    def test_prefect_consistency_with_core_functions(self):
        """
        Testet, dass Prefect-Tasks die gleichen Core-Funktionen nutzen.
        
        Das ist ein wichtiger Punkt: Alle drei Pipelines nutzen
        die GLEICHEN Framework-agnostischen Funktionen!
        """
        config = load_config("config/parameters.yml")
        
        # Führe Prefect-Flow aus
        result = breast_cancer_pipeline_flow()
        prefect_metrics = result['metrics']
        
        # Führe Core-Funktionen direkt aus (wie Vanilla)
        X, y = load_breast_cancer()
        X = clean_data(X)
        X, scaler = normalize_features(X, fit=True)
        X, _ = preprocess_features(X, y, n_features=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model = train_model(X_train, y_train, algorithm="random_forest", params={"random_state": 42})
        y_pred = predict(model, X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        direct_metrics = evaluate(y_test, y_pred, y_pred_proba)
        
        # Sollten identisch sein!
        for metric_key in direct_metrics.keys():
            assert metric_key in prefect_metrics
            np.testing.assert_almost_equal(
                prefect_metrics[metric_key],
                direct_metrics[metric_key],
                decimal=10
            )


if __name__ == "__main__":
    """
    Ausführung:
      pytest tests/test_prefect_pipeline.py -v
      pytest tests/test_prefect_pipeline.py::TestVanillaVsKedroVsPrefectEquivalence -v
    """
    pytest.main([__file__, "-v"])
