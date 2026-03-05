import pytest
import pandas as pd
from src.data_loader import load_and_preprocess_data
from src.models.logistic_regression import get_base_pipeline, evaluate_base_model
import os

TEST_DATA_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"

@pytest.fixture
def get_data():
    if not os.path.exists(TEST_DATA_PATH):
        pytest.skip(f"Test dataset {TEST_DATA_PATH} not found.")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(TEST_DATA_PATH)
    return X_train, X_test, y_train, y_test

def test_pipeline_creation():
    pipeline = get_base_pipeline()
    assert pipeline is not None
    assert 'scaler' in pipeline.named_steps
    assert 'lr' in pipeline.named_steps

def test_evaluate_base_model(get_data):
    X_train, X_test, y_train, y_test = get_data
    pipeline, train_acc, test_acc = evaluate_base_model(X_train, y_train, X_test, y_test)
    
    assert train_acc > 0, "Train accuracy should be valid"
    assert test_acc > 0, "Test accuracy should be valid"
    assert hasattr(pipeline, "score"), "Model should be a fitted pipeline"
