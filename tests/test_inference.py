import os
import pytest
from src.predict import predict

def test_model_file_exists():
    path = os.path.join(os.getcwd(), "models", "latest", "model.joblib")
    assert os.path.exists(path), "Train a model before running inference tests."

def test_predict_shapes():
    labels, probs = predict(["I love this", "This is bad"])
    assert len(labels) == 2 and len(probs) == 2
