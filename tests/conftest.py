"""
Pytest Configuration und Shared Fixtures für alle Tests.

Fixtures sind wiederverwendbare Test-Daten/-Setups.
Sie werden automatisch in Tests injiziert, z.B:

    def test_something(sample_X_train):
        # sample_X_train wird automatisch vom conftest geladen
        assert sample_X_train.shape == (400, 20)
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from src.core.data_processing import load_breast_cancer


@pytest.fixture
def breast_cancer_data():
    """
    Fixture: Vollständiger Breast Cancer Dataset.
    
    Returns: (X, y) Tuple
    """
    X, y = load_breast_cancer()
    return X, y


@pytest.fixture
def clean_small_dataset():
    """
    Fixture: Kleine, saubere Subset für schnelle Tests.
    
    Returns: (X, y) mit 100 Samples
    """
    X, y = load_breast_cancer()
    X_small = X.iloc[:100]
    y_small = y.iloc[:100]
    return X_small, y_small


@pytest.fixture
def X_train_fixture():
    """Fixture: Training Features (nach Split)"""
    X, y = load_breast_cancer()
    X_train, _, _, _ = sklearn_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train


@pytest.fixture
def y_train_fixture():
    """Fixture: Training Target"""
    X, y = load_breast_cancer()
    _, _, y_train, _ = sklearn_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return y_train


@pytest.fixture
def X_test_fixture():
    """Fixture: Test Features"""
    X, y = load_breast_cancer()
    _, X_test, _, _ = sklearn_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test


@pytest.fixture
def y_test_fixture():
    """Fixture: Test Target"""
    X, y = load_breast_cancer()
    _, _, _, y_test = sklearn_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return y_test


@pytest.fixture
def trained_random_forest_model(X_train_fixture, y_train_fixture):
    """
    Fixture: Ein trainiertes Random Forest Modell.
    
    Nützlich für Tests, die kein Modell trainieren brauchen.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_fixture, y_train_fixture)
    return model


@pytest.fixture
def fitted_scaler():
    """
    Fixture: Ein fitted StandardScaler für Tests.
    """
    X, _ = load_bc()
    X_clean = X.iloc[:100]  # Kleine Subset
    
    scaler = StandardScaler()
    scaler.fit(X_clean)
    return scaler


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config):
    """Wird einmal beim Pytest-Start aufgerufen"""
    print("\n" + "="*80)
    print("Running ML Pipeline Tests")
    print("="*80 + "\n")
