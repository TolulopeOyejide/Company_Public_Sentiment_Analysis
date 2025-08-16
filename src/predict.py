import os
import joblib
from typing import List
from .config import MODELS_DIR
from .preprocess import normalize_batch

_MODEL = None

def load_model():
    global _MODEL
    if _MODEL is None:
        path = os.path.join(MODELS_DIR, "model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError("Trained model not found. Run src/train.py first.")
        _MODEL = joblib.load(path)
    return _MODEL

def predict(texts: List[str]):
    model = load_model()
    texts_norm = normalize_batch(texts)
    labels = model.predict(texts_norm).tolist()
    try:
        proba = model.predict_proba(texts_norm).max(axis=1).tolist()
    except Exception:
        proba = [0.0] * len(texts_norm)
    return labels, proba
