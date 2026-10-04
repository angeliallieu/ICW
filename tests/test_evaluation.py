"""
Unit-Tests für src/core/evaluation.py

Testet:
- evaluate()
- compute_confusion_matrix()
- get_classification_report()
- save_metrics()
"""

import pytest
import os
import json
import tempfile
import pandas as pd
import numpy as np

from src.core.data_processing import load_breast_cancer, clean_data, normalize_features
from src.core.feature_engineering import preprocess_features, train_test_split
from src.core.modeling import train_model, predict
from src.core.evaluation import (
    evaluate,
    compute_confusion_matrix,
    get_classification_report,
    save_metrics
)


@pytest.fixture
def predictions_data():
    """Erstellt Test-Predictions für Tests"""
    # Echte Labels
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    
    # Vorhergesagte Labels
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    
    # Wahrscheinlichkeiten
    y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.7, 0.1, 0.6, 0.95, 0.15])
    
    return y_true, y_pred, y_pred_proba


class TestEvaluate:
    """Tests für evaluate() Funktion"""
    
    def test_evaluate_returns_dict(self, predictions_data):
        """Testet, dass Dictionary zurückgegeben wird"""
        y_true, y_pred, y_pred_proba = predictions_data
        metrics = evaluate(y_true, y_pred, y_pred_proba)
        assert isinstance(metrics, dict)
    
    def test_evaluate_contains_required_metrics(self, predictions_data):
        """Testet, dass alle wichtigen Metriken vorhanden sind"""
        y_true, y_pred, y_pred_proba = predictions_data
        metrics = evaluate(y_true, y_pred, y_pred_proba)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics
    
    def test_evaluate_metric_ranges(self, predictions_data):
        """Testet, dass Metriken im Bereich [0, 1] sind"""
        y_true, y_pred, y_pred_proba = predictions_data
        metrics = evaluate(y_true, y_pred, y_pred_proba)
        
        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1
    
    def test_evaluate_perfect_predictions(self):
        """Testet Metriken für perfekte Predictions"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        
        metrics = evaluate(y_true, y_pred)
        
        # Sollten alle 1.0 sein
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_evaluate_bad_predictions(self):
        """Testet Metriken für schlechte Predictions"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1])  # Alle falsch
        
        metrics = evaluate(y_true, y_pred)
        
        # Accuracy sollte 0.0 sein
        assert metrics['accuracy'] == 0.0


class TestConfusionMatrix:
    """Tests für compute_confusion_matrix() Funktion"""
    
    def test_confusion_matrix_shape(self, predictions_data):
        """Testet, dass Matrix Shape (2, 2) hat"""
        y_true, y_pred, _ = predictions_data
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
    
    def test_confusion_matrix_dtype(self, predictions_data):
        """Testet, dass Matrix Integer-Werte hat"""
        y_true, y_pred, _ = predictions_data
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.dtype in [np.int64, np.int32]
    
    def test_confusion_matrix_sum(self, predictions_data):
        """Testet, dass Summe aller Elemente = n_samples ist"""
        y_true, y_pred, _ = predictions_data
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.sum() == len(y_true)
    
    def test_confusion_matrix_values(self):
        """Testet CM Werte für bekannte Predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        cm = compute_confusion_matrix(y_true, y_pred)
        
        # TN=1, FP=1, FN=0, TP=2
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[1, 1] == 2  # TP


class TestGetClassificationReport:
    """Tests für get_classification_report() Funktion"""
    
    def test_classification_report_returns_string(self, predictions_data):
        """Testet, dass String zurückgegeben wird"""
        y_true, y_pred, _ = predictions_data
        report = get_classification_report(y_true, y_pred)
        assert isinstance(report, str)
    
    def test_classification_report_contains_metrics(self, predictions_data):
        """Testet, dass Report wichtige Metriken enthält"""
        y_true, y_pred, _ = predictions_data
        report = get_classification_report(y_true, y_pred)
        
        # Report sollte diese Wörter enthalten
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
        assert 'support' in report


class TestSaveMetrics:
    """Tests für save_metrics() Funktion"""
    
    def test_save_metrics_creates_files(self, predictions_data):
        """Testet, dass save_metrics Dateien erstellt"""
        y_true, y_pred, y_pred_proba = predictions_data
        
        metrics = evaluate(y_true, y_pred, y_pred_proba)
        cm = compute_confusion_matrix(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = os.path.join(tmpdir, "metrics.json")
            cm_file = os.path.join(tmpdir, "confusion_matrix.csv")
            pred_file = os.path.join(tmpdir, "predictions.csv")
            
            save_metrics(metrics, cm, y_pred, y_true, metrics_file, cm_file, pred_file)
            
            # Alle Dateien sollten existieren
            assert os.path.exists(metrics_file)
            assert os.path.exists(cm_file)
            assert os.path.exists(pred_file)
    
    def test_save_metrics_json_content(self, predictions_data):
        """Testet, dass Metriken-JSON korrekt ist"""
        y_true, y_pred, y_pred_proba = predictions_data
        
        metrics = evaluate(y_true, y_pred, y_pred_proba)
        cm = compute_confusion_matrix(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = os.path.join(tmpdir, "metrics.json")
            cm_file = os.path.join(tmpdir, "confusion_matrix.csv")
            pred_file = os.path.join(tmpdir, "predictions.csv")
            
            save_metrics(metrics, cm, y_pred, y_true, metrics_file, cm_file, pred_file)
            
            # Lade und verifiziere JSON
            with open(metrics_file, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert isinstance(loaded_metrics, dict)
            assert 'accuracy' in loaded_metrics
    
    def test_save_metrics_confusion_matrix_csv(self, predictions_data):
        """Testet, dass CM CSV korrekt ist"""
        y_true, y_pred, _ = predictions_data
        
        cm = compute_confusion_matrix(y_true, y_pred)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = os.path.join(tmpdir, "metrics.json")
            cm_file = os.path.join(tmpdir, "confusion_matrix.csv")
            pred_file = os.path.join(tmpdir, "predictions.csv")
            
            metrics = {'accuracy': 0.8}  # Dummy
            
            save_metrics(metrics, cm, y_pred, y_true, metrics_file, cm_file, pred_file)
            
            # Lade und verifiziere CSV
            cm_df = pd.read_csv(cm_file, index_col=0)
            assert cm_df.shape == (2, 2)


class TestEvaluationPipeline:
    """Integration-Tests für kompletten Evaluierungs-Prozess"""
    
    def test_full_evaluation_pipeline(self):
        """Testet kompletten Evaluierungs-Workflow"""
        # Erstelle echte Predictions
        X, y = load_breast_cancer()
        X = clean_data(X)
        X, _ = normalize_features(X)
        X, _ = preprocess_features(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        # Train Model
        model = train_model(X_train, y_train)
        y_pred = predict(model, X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate(y_test.values, y_pred, y_pred_proba)
        cm = compute_confusion_matrix(y_test.values, y_pred)
        report = get_classification_report(y_test.values, y_pred)
        
        # Assertions
        assert 'accuracy' in metrics
        assert cm.sum() == len(y_test)
        assert 'precision' in report
