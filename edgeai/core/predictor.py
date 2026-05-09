"""
EdgeAI Predictor
================
Loads pre-trained models and runs inference entirely on the
local device.  This is the heart of the framework — nothing
is ever sent to any server.

Key guarantees:
    • Data transferred  : 0 bytes
    • Internet used     : None
    • Latency measured per prediction
"""

import os
import pickle
import time


# ──────────────────────────────────────────────────────────
#  Path helpers
# ──────────────────────────────────────────────────────────

def _package_dir():
    """Return the absolute path to the edgeai package root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model_path(domain, model_name):
    """
    Return the absolute path to a pretrained .pkl model file.

    Path pattern:
        edgeai/models/pretrained/{domain}_{model_name}.pkl

    Args:
        domain (str):     Domain name (e.g. 'healthcare').
        model_name (str): Model type (e.g. 'logistic').

    Returns:
        str: Absolute file path to the .pkl file.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    filename = f"{domain}_{model_name}.pkl"
    path = os.path.join(_package_dir(), "models", "pretrained", filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pretrained model not found: '{path}'. "
            f"Run the trainer first to generate {filename}."
        )

    return path


# ──────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────

def load_model(domain, model_name):
    """
    Load a pretrained model from its .pkl file.

    Args:
        domain (str):     Domain name.
        model_name (str): Model type.

    Returns:
        object: The deserialised scikit-learn model.

    Raises:
        FileNotFoundError: If the .pkl file is missing.
        RuntimeError:      If the file cannot be unpickled.
    """
    path = get_model_path(domain, model_name)

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from '{path}': {exc}"
        ) from exc

    print(f"Model loaded: {domain} / {model_name}")
    return model


def get_model_size_kb(domain, model_name):
    """
    Get the on-disk size of a pretrained .pkl model in kilobytes.

    Args:
        domain (str):     Domain name.
        model_name (str): Model type.

    Returns:
        float: File size in KB.

    Raises:
        FileNotFoundError: If the .pkl file is missing.
    """
    path = get_model_path(domain, model_name)
    size_bytes = os.path.getsize(path)
    return round(size_bytes / 1024, 2)


# ──────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────

def run_inference(model, input_data):
    """
    Run a single prediction on the local device.

    Measures wall-clock latency using ``time.perf_counter()``.

    Args:
        model:      A fitted scikit-learn estimator.
        input_data: NumPy array of shape (1, n_features).

    Returns:
        dict: Keys —
            prediction              : model output value
            latency_ms              : float
            data_transferred_bytes  : 0
            internet_used           : 'None'
    """
    start = time.perf_counter()
    prediction = model.predict(input_data)
    elapsed = (time.perf_counter() - start) * 1000  # ms

    return {
        "prediction": prediction[0],
        "latency_ms": round(elapsed, 3),
        "data_transferred_bytes": 0,
        "internet_used": "None",
    }


def get_prediction_confidence(model, input_data):
    """
    Get the confidence percentage for a prediction.

    Uses ``predict_proba`` if the model supports it.  Confidence
    is the maximum probability across all classes.

    Args:
        model:      A fitted scikit-learn estimator.
        input_data: NumPy array of shape (1, n_features).

    Returns:
        float: Confidence as a percentage (0.0–100.0).
               Returns 0.0 if the model does not support
               probability estimates.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            return round(float(proba.max()) * 100, 2)
    except Exception:
        pass

    return 0.0


# ──────────────────────────────────────────────────────────
#  Complete local prediction
# ──────────────────────────────────────────────────────────

def predict_local(model, input_data, domain, model_name):
    """
    Full local-only prediction pipeline.

    Combines inference, confidence estimation, and model metadata
    into a single result dictionary.

    Args:
        model:          A fitted scikit-learn estimator.
        input_data:     NumPy array of shape (1, n_features).
        domain (str):   Domain name.
        model_name (str): Model type.

    Returns:
        dict: Keys —
            domain                  : str
            model_name              : str
            model_size_kb           : float
            prediction              : value
            confidence_pct          : float
            latency_ms              : float
            data_transferred_bytes  : 0
            internet_used           : 'None'
    """
    inference = run_inference(model, input_data)
    confidence = get_prediction_confidence(model, input_data)

    try:
        model_size = get_model_size_kb(domain, model_name)
    except FileNotFoundError:
        model_size = 0.0

    return {
        "domain": domain,
        "model_name": model_name,
        "model_size_kb": model_size,
        "prediction": inference["prediction"],
        "confidence_pct": confidence,
        "latency_ms": inference["latency_ms"],
        "data_transferred_bytes": 0,
        "internet_used": "None",
    }


# ──────────────────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Predictor module loaded successfully.")
    print(f"Package directory: {_package_dir()}")
    print("No pretrained models yet — run trainer first.")
