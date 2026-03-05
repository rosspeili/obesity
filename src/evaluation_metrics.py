import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from rich.console import Console

console = Console()

def evaluate_and_save_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, save_dir: str = "models"):
    """
    Evaluates a model using ML metrics and saves the serialized object to disk for deployment.
    """
    y_pred = model.predict(X_test)
    
    # Generate Advanced Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Ensure Directory Exists and Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.joblib")
    joblib.dump(model, save_path)
    
    return report, cm, save_path
