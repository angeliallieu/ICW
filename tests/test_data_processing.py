"""
Unit-Tests für src/core/data_processing.py

Testet:
- load_breast_cancer()
- clean_data()
- normalize_features()
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.core.data_processing import (
    load_breast_cancer,
    clean_data,
    normalize_features
)


class TestLoadBreastCancer:
    """Tests für load_breast_cancer() Funktion"""
    
    def test_load_returns_tuple(self):
        """Testet, dass load_breast_cancer ein Tuple zurückgibt"""
        result = load_breast_cancer()
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_load_returns_dataframe_and_series(self):
        """Testet, dass X DataFrame und y Series ist"""
        X, y = load_breast_cancer()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
    
    def test_load_correct_shapes(self):
        """Testet die korrekten Shapes"""
        X, y = load_breast_cancer()
        assert X.shape == (569, 30)  # 569 Samples, 30 Features
        assert y.shape == (569,)      # 569 Targets
    
    def test_load_correct_dtypes(self):
        """Testet die korrekten Datentypen"""
        X, y = load_breast_cancer()
        assert X.dtypes.unique()[0] == np.float64
        assert y.dtype in [np.int64, np.int32]
    
    def test_load_no_missing_values(self):
        """Testet, dass es keine NaN Werte gibt"""
        X, y = load_breast_cancer()
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0
    
    def test_load_binary_classification_target(self):
        """Testet, dass Target nur 0 und 1 hat"""
        _, y = load_breast_cancer()
        unique_values = set(y.unique())
        assert unique_values == {0, 1}
    
    def test_load_target_distribution(self):
        """Testet die Klassenverteilung"""
        _, y = load_breast_cancer()
        value_counts = y.value_counts()
        # Breast Cancer hat ~357 Benign und ~212 Malignant
        assert 350 < value_counts.iloc[0] < 360
        assert 210 < value_counts.iloc[1] < 220


class TestCleanData:
    """Tests für clean_data() Funktion"""
    
    def test_clean_returns_dataframe(self):
        """Testet, dass Output ein DataFrame ist"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        assert isinstance(X_clean, pd.DataFrame)
    
    def test_clean_removes_nan(self):
        """Testet, dass NaN entfernt werden"""
        X, _ = load_breast_cancer()
        X_with_nan = X.copy()
        X_with_nan.iloc[0, 0] = np.nan
        X_with_nan.iloc[5, 5] = np.nan
        
        X_clean = clean_data(X_with_nan)
        assert X_clean.isnull().sum().sum() == 0
    
    def test_clean_removes_duplicates(self):
        """Testet, dass Duplikate entfernt werden"""
        X, _ = load_breast_cancer()
        X_with_dup = pd.concat([X, X.iloc[0:5]], ignore_index=True)
        
        assert len(X_with_dup) == len(X) + 5
        
        X_clean = clean_data(X_with_dup)
        assert len(X_clean) == len(X)
    
    def test_clean_preserves_column_names(self):
        """Testet, dass Spaltennamen erhalten bleiben"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        assert list(X.columns) == list(X_clean.columns)
    
    def test_clean_returns_reset_index(self):
        """Testet, dass Index gelöscht wird (von 0 anfangen)"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        assert list(X_clean.index) == list(range(len(X_clean)))


class TestNormalizeFeatures:
    """Tests für normalize_features() Funktion"""
    
    def test_normalize_returns_tuple(self):
        """Testet, dass Tuple zurückgegeben wird"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        result = normalize_features(X_clean)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_normalize_returns_dataframe_and_scaler(self):
        """Testet korrekten Return-Typen"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, scaler = normalize_features(X_clean)
        assert isinstance(X_norm, pd.DataFrame)
        assert isinstance(scaler, StandardScaler)
    
    def test_normalize_correct_shapes(self):
        """Testet, dass Shapes erhalten bleiben"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        assert X_norm.shape == X_clean.shape
    
    def test_normalize_zero_mean(self):
        """Testet, dass Mittelwert ~0 ist"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        
        # Mittelwert sollte sehr nahe bei 0 sein
        mean = X_norm.mean().abs().max()
        assert mean < 1e-10
    
    def test_normalize_unit_variance(self):
        """Testet, dass Standardabweichung ~1 ist"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        
        # Standardabweichung sollte sehr nahe bei 1 sein
        std = X_norm.std()
        assert (0.99 < std).all() and (std < 1.01).all()
    
    def test_normalize_fit_vs_transform(self):
        """Testet fit=True vs fit=False Verhalten"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        
        # Fit auf erste 100 Zeilen
        X_train = X_clean.iloc[:100]
        X_test = X_clean.iloc[100:150]
        
        # Fit auf Train
        X_train_norm, scaler = normalize_features(X_train, fit=True)
        
        # Transform Test mit Same Scaler
        X_test_norm, _ = normalize_features(X_test, scaler=scaler, fit=False)
        
        # Test-Set sollte nicht perfekt normalisiert sein
        # (weil scaler auf Train gelernt hat)
        test_mean = X_test_norm.mean().abs().max()
        test_std = X_test_norm.std()
        
        # Test-Mean sollte nicht 0 sein (nicht auf Test fit)
        assert test_mean > 0
        # Test-Std sollte nicht genau 1 sein
        assert not (test_std < 1.01).all()
    
    def test_normalize_preserves_column_names(self):
        """Testet, dass Spaltennamen erhalten bleiben"""
        X, _ = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        assert list(X_norm.columns) == list(X_clean.columns)


class TestDataProcessingPipeline:
    """Integration-Tests für den kompletten Daten-Verarbeitungsprozess"""
    
    def test_full_pipeline(self):
        """Testet: Load → Clean → Normalize"""
        # Load
        X, y = load_breast_cancer()
        assert X.shape == (569, 30)
        
        # Clean
        X_clean = clean_data(X)
        assert X_clean.shape[0] <= X.shape[0]
        assert X_clean.shape[1] == X.shape[1]
        
        # Normalize
        X_norm, scaler = normalize_features(X_clean)
        assert X_norm.shape == X_clean.shape
        assert abs(X_norm.mean().max()) < 1e-10
    
    def test_reproducibility(self):
        """Testet, dass mehrfaches Ausführen identisch ist"""
        # Erste Ausführung
        X1, y1 = load_breast_cancer()
        X1_clean = clean_data(X1)
        X1_norm, _ = normalize_features(X1_clean)
        
        # Zweite Ausführung
        X2, y2 = load_breast_cancer()
        X2_clean = clean_data(X2)
        X2_norm, _ = normalize_features(X2_clean)
        
        # Sollten identisch sein
        pd.testing.assert_frame_equal(X1_norm, X2_norm)
        pd.testing.assert_series_equal(y1, y2)
