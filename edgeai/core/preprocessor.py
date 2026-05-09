"""
EdgeAI Preprocessor
===================
Handles any incoming raw sensor data and prepares it for model
inference.  Accepts lists, NumPy arrays, and pandas DataFrames.

Domains and their expected features:
    healthcare   – heart_rate, blood_pressure, temperature, spo2, activity_level
    smartcity    – vehicle_count, avg_speed, traffic_density, weather
    environment  – temperature, humidity, pm25, co2
"""

import os

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────
#  Domain feature definitions
# ──────────────────────────────────────────────────────────

_DOMAIN_FEATURES = {
    "healthcare": {
        "features": [
            "heart_rate",
            "blood_pressure",
            "temperature",
            "spo2",
            "activity_level",
        ],
        "expected_count": 5,
    },
    "smartcity": {
        "features": [
            "vehicle_count",
            "avg_speed",
            "traffic_density",
            "weather",
        ],
        "expected_count": 4,
    },
    "environment": {
        "features": [
            "temperature",
            "humidity",
            "pm25",
            "co2",
        ],
        "expected_count": 4,
    },
}


# ──────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────

def get_domain_features(domain):
    """
    Return the list of feature names for a given domain.

    Args:
        domain (str): One of 'healthcare', 'smartcity', 'environment'.

    Returns:
        list[str]: Feature names.

    Raises:
        ValueError: If the domain is not recognised.
    """
    if domain not in _DOMAIN_FEATURES:
        valid = ", ".join(_DOMAIN_FEATURES.keys())
        raise ValueError(
            f"Unknown domain '{domain}'. Valid domains: {valid}"
        )
    return list(_DOMAIN_FEATURES[domain]["features"])


def validate_input(data, domain):
    """
    Validate that the input data has the correct number of features
    for the given domain.

    Accepts lists, tuples, NumPy arrays, or pandas DataFrames.

    Args:
        data: Input data (list, tuple, np.ndarray, or pd.DataFrame).
        domain (str): Target domain name.

    Returns:
        bool: True if the input is valid.

    Raises:
        ValueError: If feature count does not match or the domain
                    is unknown.
        TypeError:  If the input type is not supported.
    """
    if domain not in _DOMAIN_FEATURES:
        valid = ", ".join(_DOMAIN_FEATURES.keys())
        raise ValueError(
            f"Unknown domain '{domain}'. Valid domains: {valid}"
        )

    expected = _DOMAIN_FEATURES[domain]["expected_count"]

    # --- Determine actual feature count ---
    if isinstance(data, pd.DataFrame):
        actual = data.shape[1]
    elif isinstance(data, np.ndarray):
        actual = data.shape[-1] if data.ndim >= 1 else 1
    elif isinstance(data, (list, tuple)):
        # Flat list → single sample
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            actual = len(data[0])
        else:
            actual = len(data)
    else:
        raise TypeError(
            f"Unsupported input type: {type(data).__name__}. "
            "Use list, tuple, numpy array, or pandas DataFrame."
        )

    if actual != expected:
        features = _DOMAIN_FEATURES[domain]["features"]
        raise ValueError(
            f"Domain '{domain}' expects {expected} features "
            f"{features}, but received {actual}."
        )

    return True


def normalize_input(data):
    """
    Convert any accepted input format into a NumPy array shaped
    (1, n_features) suitable for a single-sample prediction.

    Args:
        data: Input data (list, tuple, np.ndarray, or pd.DataFrame).

    Returns:
        np.ndarray: Array with shape (1, n_features).

    Raises:
        TypeError: If the input type is not supported.
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values.astype(float)
    elif isinstance(data, np.ndarray):
        arr = data.astype(float)
    elif isinstance(data, (list, tuple)):
        arr = np.array(data, dtype=float)
    else:
        raise TypeError(
            f"Unsupported input type: {type(data).__name__}. "
            "Use list, tuple, numpy array, or pandas DataFrame."
        )

    # Reshape flat array to (1, n_features)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


def preprocess_input(data, domain, scaler=None):
    """
    Full preprocessing pipeline: validate → normalise → (optional) scale.

    Args:
        data:   Raw input data.
        domain: Target domain name.
        scaler: Optional fitted scaler (e.g. StandardScaler) that
                exposes a ``transform`` method.  If None, raw
                normalised values are returned.

    Returns:
        np.ndarray: Prepared array with shape (1, n_features),
                    ready for model prediction.
    """
    validate_input(data, domain)
    arr = normalize_input(data)

    if scaler is not None:
        try:
            arr = scaler.transform(arr)
        except Exception as exc:
            raise RuntimeError(
                f"Scaler transform failed: {exc}"
            ) from exc

    return arr


def load_training_data(domain, data_dir="data/"):
    """
    Load the training CSV for a domain from the data directory.

    The CSV must contain feature columns matching the domain
    definition plus a column named ``target``.

    Args:
        domain (str):   Domain name.
        data_dir (str): Relative or absolute path to the data
                        directory.  Defaults to ``data/``.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            X — feature matrix of shape (n_samples, n_features).
            y — target vector of shape (n_samples,).

    Raises:
        ValueError:         If domain is unknown.
        FileNotFoundError:  If the CSV file does not exist.
        KeyError:           If required columns are missing.
    """
    if domain not in _DOMAIN_FEATURES:
        valid = ", ".join(_DOMAIN_FEATURES.keys())
        raise ValueError(
            f"Unknown domain '{domain}'. Valid domains: {valid}"
        )

    csv_path = os.path.join(data_dir, f"{domain}.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Training data not found at '{csv_path}'. "
            f"Please ensure {domain}.csv exists in '{data_dir}'."
        )

    df = pd.read_csv(csv_path)

    feature_names = _DOMAIN_FEATURES[domain]["features"]
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise KeyError(
            f"Missing feature columns in {csv_path}: {missing}"
        )
    if "target" not in df.columns:
        raise KeyError(
            f"Missing 'target' column in {csv_path}."
        )

    X = df[feature_names].values.astype(float)
    y = df["target"].values

    return X, y


# ──────────────────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = [72, 120, 36.5, 98, 1]
    domain = "healthcare"

    print(f"Domain features : {get_domain_features(domain)}")
    print(f"Valid input     : {validate_input(sample, domain)}")

    prepared = preprocess_input(sample, domain)
    print(f"Prepared shape  : {prepared.shape}")
    print(f"Prepared values : {prepared}")
