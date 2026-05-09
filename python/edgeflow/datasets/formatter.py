"""
EdgeFlow Dataset Formatter
==========================
Handles label encoding, feature extraction, and data preparation
for EdgeFlow model training.

Converts string labels (SAFE, DANGER) to integers (0, 1) for training,
and back to strings for prediction output.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


class EdgeFlowFormatter:
    """
    Handles all data formatting for EdgeFlow training pipeline.

    Responsibilities:
    - Encode string labels to integers
    - Decode integer predictions back to string labels
    - Fit and apply StandardScaler on training data only
    - Store label map for export to C++ header
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.label_map: Dict[str, int] = {}          # {"SAFE": 0, "DANGER": 1}
        self.inverse_label_map: Dict[int, str] = {}  # {0: "SAFE", 1: "DANGER"}
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, y: np.ndarray, X_train: np.ndarray, feature_names: List[str]):
        """
        Fit the formatter on training data.
        Must be called before transform().

        Parameters
        ----------
        y : np.ndarray of string labels (full dataset labels)
        X_train : np.ndarray of training features (NOT test data)
        feature_names : list of feature column names
        """
        # Fit label encoder on all unique labels
        self.label_encoder.fit(y)

        # Build label maps
        classes = list(self.label_encoder.classes_)
        self.label_map = {label: idx for idx, label in enumerate(classes)}
        self.inverse_label_map = {idx: label for idx, label in enumerate(classes)}

        # Fit scaler on training data only (prevent data leakage)
        self.scaler.fit(X_train)

        # Store feature names
        self.feature_names = list(feature_names)

        self.is_fitted = True

    def encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert string labels to integer array."""
        if not self.is_fitted:
            raise RuntimeError("Formatter not fitted. Call fit() first.")
        return self.label_encoder.transform(y)

    def decode_labels(self, y_int: np.ndarray) -> np.ndarray:
        """Convert integer predictions back to string labels."""
        if not self.is_fitted:
            raise RuntimeError("Formatter not fitted. Call fit() first.")
        return self.label_encoder.inverse_transform(y_int)

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Apply StandardScaler transform to feature array."""
        if not self.is_fitted:
            raise RuntimeError("Formatter not fitted. Call fit() first.")
        return self.scaler.transform(X)

    def get_scaler_params(self) -> Dict:
        """
        Return scaler parameters needed for C++ export.

        Returns
        -------
        dict with keys:
            mean : list of floats (one per feature)
            scale : list of floats (one per feature)
        """
        if not self.is_fitted:
            raise RuntimeError("Formatter not fitted. Call fit() first.")
        return {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
        }

    def get_label_map(self) -> Dict[str, int]:
        """Return the string → integer label mapping."""
        return dict(self.label_map)

    def get_n_classes(self) -> int:
        """Return number of unique classes."""
        return len(self.label_map)

    def decode_single(self, label_int: int) -> str:
        """Decode a single integer prediction to string label."""
        if label_int not in self.inverse_label_map:
            raise ValueError(f"Unknown label index: {label_int}")
        return self.inverse_label_map[label_int]
