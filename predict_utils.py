import joblib
import numpy as np

def load_model(path):
    return joblib.load(path)

def ensemble_predict(models, X):
    """Average predictions from multiple models"""
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)
