"""
Unit-Tests für src/core/feature_engineering.py

Testet:
- preprocess_features()
- train_test_split()
- encode_target()
"""

import pytest
import pandas as pd
import numpy as np

from src.core.data_processing import load_breast_cancer, clean_data, normalize_features
from src.core.feature_engineering import (
    preprocess_features,
    train_test_split,
    encode_target
)


class TestPreprocessFeatures:
    """Tests für preprocess_features() Funktion"""
    
    def test_preprocess_returns_tuple(self):
        """Testet, dass Tuple zurückgegeben wird"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        
        result = preprocess_features(X_norm, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_preprocess_returns_dataframe_and_list(self):
        """Testet korrekten Return-Typen"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        
        X_proc, feature_names = preprocess_features(X_norm, y, n_features=20)
        assert isinstance(X_proc, pd.DataFrame)
        assert isinstance(feature_names, list)
    
    def test_preprocess_correct_n_features(self):
        """Testet, dass richtige Anzahl Features selected wird"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        
        X_proc, feature_names = preprocess_features(X_norm, y, n_features=10)
        assert X_proc.shape[1] == 10
        assert len(feature_names) == 10
    
    def test_preprocess_preserves_n_samples(self):
        """Testet, dass Anzahl Samples erhalten bleibt"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        
        original_n = X_norm.shape[0]
        X_proc, _ = preprocess_features(X_norm, y, n_features=20)
        
        assert X_proc.shape[0] == original_n
    
    def test_preprocess_feature_names_valid(self):
        """Testet, dass Feature-Namen von Original sind"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        
        X_proc, feature_names = preprocess_features(X_norm, y, n_features=20)
        
        # Alle Feature-Namen sollten aus der Original-Liste stammen
        for name in feature_names:
            assert name in X_norm.columns


class TestTrainTestSplit:
    """Tests für train_test_split() Funktion"""
    
    def test_train_test_split_returns_tuple(self):
        """Testet, dass 4er-Tuple zurückgegeben wird"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        X_proc, _ = preprocess_features(X_norm, y)
        
        result = train_test_split(X_proc, y, test_size=0.2, random_state=42)
        assert isinstance(result, tuple)
        assert len(result) == 4
    
    def test_train_test_split_correct_sizes(self):
        """Testet, dass Größen korrekt sind (mit Rounding-Toleranz)"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        X_proc, _ = preprocess_features(X_norm, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, y, test_size=0.2, random_state=42
        )
        
        total = len(X_train) + len(X_test)
        expected_test_size = int(len(X_proc) * 0.2)
        expected_train_size = len(X_proc) - expected_test_size
        
        # Allow ±2 samples tolerance due to rounding
        assert abs(len(X_train) - expected_train_size) <= 2
        assert abs(len(X_test) - expected_test_size) <= 2
        assert abs(len(y_train) - expected_train_size) <= 2
        assert abs(len(y_test) - expected_test_size) <= 2
        
        # Total should still be exact
        assert total == len(X_proc)
    
    def test_train_test_split_reproducibility(self):
        """Testet, dass gleicher random_state gleiche Split gibt"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        X_proc, _ = preprocess_features(X_norm, y)
        
        # Split 1
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X_proc, y, test_size=0.2, random_state=42
        )
        
        # Split 2
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X_proc, y, test_size=0.2, random_state=42
        )
        
        # Sollten identisch sein
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
    
    def test_train_test_split_different_random_state(self):
        """Testet, dass unterschiedliche random_state unterschiedliche Splits gibt"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        X_proc, _ = preprocess_features(X_norm, y)
        
        # Split mit random_state=42
        X_train1, _, _, _ = train_test_split(
            X_proc, y, test_size=0.2, random_state=42
        )
        
        # Split mit random_state=123
        X_train2, _, _, _ = train_test_split(
            X_proc, y, test_size=0.2, random_state=123
        )
        
        # Sollten unterschiedlich sein (nicht identische erste Indizes)
        assert not (X_train1.iloc[0] == X_train2.iloc[0]).all()
    
    def test_train_test_split_stratification(self):
        """Testet, dass stratify die Klassenverteilung erhält"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        X_proc, _ = preprocess_features(X_norm, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, y, test_size=0.2, random_state=42
        )
        
        # Klassenverteilung in Original vs Train vs Test
        original_ratio = y.value_counts(normalize=True)
        train_ratio = y_train.value_counts(normalize=True)
        test_ratio = y_test.value_counts(normalize=True)
        
        # Sollten ähnlich sein (stratified)
        # Toleranz: ±5%
        for class_label in [0, 1]:
            assert abs(original_ratio[class_label] - train_ratio[class_label]) < 0.05
            assert abs(original_ratio[class_label] - test_ratio[class_label]) < 0.05
    
    def test_train_test_split_no_overlap(self):
        """Testet, dass Train und Test keinen Overlap haben"""
        X, y = load_breast_cancer()
        X_clean = clean_data(X)
        X_norm, _ = normalize_features(X_clean)
        X_proc, _ = preprocess_features(X_norm, y)
        
        X_train, X_test, _, _ = train_test_split(
            X_proc, y, test_size=0.2, random_state=42
        )
        
        # Keine gemeinsamen Indizes
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0


class TestEncodeTarget:
    """Tests für encode_target() Funktion"""
    
    def test_encode_target_already_numeric(self):
        """Testet, dass numerisch Target unverändert bleibt"""
        _, y = load_breast_cancer()
        y_encoded = encode_target(y)
        pd.testing.assert_series_equal(y, y_encoded)
    
    def test_encode_target_categorical(self):
        """Testet Encoding von kategorischen Labels"""
        # Erstelle kategorisches Target
        y_cat = pd.Series(['benign', 'malignant', 'benign', 'malignant'])
        y_encoded = encode_target(y_cat)
        
        # Sollte numerisch sein
        assert y_encoded.dtype in ['int64', 'int32']
        
        # Sollte 0 und 1 enthalten
        assert set(y_encoded.unique()) == {0, 1}


class TestFeatureEngineeringPipeline:
    """Integration-Tests für kompletten Feature Engineering Prozess"""
    
    def test_full_feature_engineering_pipeline(self):
        """Testet: Load → Clean → Normalize → Preprocess → Split"""
        # Load
        X, y = load_breast_cancer()
        
        # Clean
        X = clean_data(X)
        
        # Normalize
        X, _ = normalize_features(X)
        
        # Preprocess
        X, _ = preprocess_features(X, y, n_features=20)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Assertions
        assert X_train.shape[0] > X_test.shape[0]
        assert X_train.shape[1] == 20
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
