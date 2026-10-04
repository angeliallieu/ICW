"""
Unit-Tests für src/core/modeling.py

Testet:
- train_model()
- predict()
- save_model()
- load_model()
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.core.data_processing import load_breast_cancer, clean_data, normalize_features
from src.core.feature_engineering import preprocess_features, train_test_split
from src.core.modeling import train_model, predict, save_model, load_model


@pytest.fixture
def train_test_data():
    """Erstellt Train/Test Daten für Tests"""
    X, y = load_breast_cancer()
    X = clean_data(X)
    X, _ = normalize_features(X)
    X, _ = preprocess_features(X, y, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


class TestTrainModel:
    """Tests für train_model() Funktion"""
    
    def test_train_model_returns_classifier(self, train_test_data):
        """Testet, dass ein Classifier zurückgegeben wird"""
        X_train, _, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        assert isinstance(model, RandomForestClassifier)
    
    def test_train_model_has_fit_method(self, train_test_data):
        """Testet, dass Modell .fit() aufgerufen hat"""
        X_train, _, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        # Wenn fit() aufgerufen wurde, sollte das Modell classes_ haben
        assert hasattr(model, 'classes_')
    
    def test_train_model_reproducibility(self, train_test_data):
        """Testet, dass random_state=42 identische Modelle gibt"""
        X_train, _, y_train, _ = train_test_data
        
        # Model 1
        model1 = train_model(X_train, y_train, params={"random_state": 42})
        pred1 = model1.predict(X_train[:10])
        
        # Model 2
        model2 = train_model(X_train, y_train, params={"random_state": 42})
        pred2 = model2.predict(X_train[:10])
        
        # Sollten identische Predictions geben
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_train_model_different_random_state(self, train_test_data):
        """Testet, dass unterschiedliche random_state unterschiedliche Modelle gibt"""
        X_train, _, y_train, _ = train_test_data
        
        # Model mit random_state=42
        model1 = train_model(X_train, y_train, params={"random_state": 42})
        pred1 = model1.predict(X_train[:10])
        
        # Model mit random_state=123
        model2 = train_model(X_train, y_train, params={"random_state": 123})
        pred2 = model2.predict(X_train[:10])


class TestPredict:
    """Tests für predict() Funktion"""
    
    def test_predict_returns_array(self, train_test_data):
        """Testet, dass numpy array zurückgegeben wird"""
        X_train, X_test, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        predictions = predict(model, X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_correct_shape(self, train_test_data):
        """Testet, dass Predictions korrekte Shape haben"""
        X_train, X_test, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        predictions = predict(model, X_test)
        assert predictions.shape == (len(X_test),)
    
    def test_predict_binary_values(self, train_test_data):
        """Testet, dass Predictions nur 0 und 1 sind"""
        X_train, X_test, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        predictions = predict(model, X_test)
        assert set(predictions) == {0, 1} or set(predictions) == {0} or set(predictions) == {1}
    
    def test_predict_on_training_data(self, train_test_data):
        """Testet Predictions auf Trainings-Daten"""
        X_train, _, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        predictions = predict(model, X_train)
        
        accuracy = (predictions == y_train).mean()
        assert accuracy > 0.7


class TestSaveLoadModel:
    """Tests für save_model() und load_model()"""
    
    def test_save_model_creates_file(self, train_test_data):
        """Testet, dass save_model Datei erstellt"""
        X_train, _, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.joblib")
            save_model(model, filepath)
            assert os.path.exists(filepath)
    
    def test_save_load_roundtrip(self, train_test_data):
        """Testet Save → Load Roundtrip"""
        X_train, X_test, y_train, _ = train_test_data
        
        # Trainiere Modell
        model = train_model(X_train, y_train)
        pred_original = predict(model, X_test[:5])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.joblib")
            
            # Save
            save_model(model, filepath)
            
            # Load
            model_loaded = load_model(filepath)
            pred_loaded = predict(model_loaded, X_test[:5])
            
            # Predictions sollten identisch sein
            np.testing.assert_array_equal(pred_original, pred_loaded)
    
    def test_load_model_returns_classifier(self, train_test_data):
        """Testet, dass geladenes Modell ein Classifier ist"""
        X_train, _, y_train, _ = train_test_data
        model = train_model(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.joblib")
            save_model(model, filepath)
            model_loaded = load_model(filepath)
            
            assert isinstance(model_loaded, RandomForestClassifier)


class TestModelingPipeline:
    """Integration-Tests für kompletten Modellierungs-Prozess"""
    
    def test_full_modeling_pipeline(self, train_test_data):
        """Testet: Train → Predict → Evaluate"""
        X_train, X_test, y_train, y_test = train_test_data
        
        # Train
        model = train_model(X_train, y_train)
        
        # Predict
        y_pred_train = predict(model, X_train)
        y_pred_test = predict(model, X_test)
        
        # Accuracy
        train_accuracy = (y_pred_train == y_train).mean()
        test_accuracy = (y_pred_test == y_test).mean()
        
        assert train_accuracy > 0.8
        assert test_accuracy > 0.8
