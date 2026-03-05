import pytest
import pandas as pd
from src.data_loader import load_and_preprocess_data
import os

TEST_DATA_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"

@pytest.fixture
def get_data():
    if not os.path.exists(TEST_DATA_PATH):
        pytest.skip(f"Test dataset {TEST_DATA_PATH} not found.")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(TEST_DATA_PATH)
    return X_train, X_test, y_train, y_test

def test_data_loader_columns(get_data):
    X_train, X_test, y_train, y_test = get_data
    # Check that there are exactly 18 specific predictor columns
    assert X_train.shape[1] == 18, "Expected 18 columns in predictor variables DataFrame"
    assert X_test.shape[1] == 18, "Expected 18 columns in predictor variables DataFrame"
    assert type(X_train) == pd.DataFrame, "X_train should be a DataFrame"
    assert type(y_train) == pd.Series, "y_train should be a Series"
    
def test_data_split_proportions(get_data):
    X_train, X_test, y_train, y_test = get_data
    total_samples = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_samples
    assert test_ratio > 0.15 and test_ratio < 0.25, "Test split should be approximately 20%"

def test_y_values(get_data):
    X_train, X_test, y_train, y_test = get_data
    assert set(y_train.unique()) <= {0, 1}, "Outcome variable should be binary (0 or 1)"
    assert set(y_test.unique()) <= {0, 1}, "Outcome variable should be binary (0 or 1)"
