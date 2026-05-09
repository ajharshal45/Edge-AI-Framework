"""
EdgeFlow Dataset Loader
=======================
Loads and validates CSV datasets for EdgeFlow training.

Expected CSV format:
    feature_1, feature_2, ..., feature_n, label
    2800, 1, DANGER
    1200, 0, SAFE

Rules:
    - Last column is always the label column
    - Labels are strings (SAFE, DANGER, etc.)
    - No missing values allowed
    - Minimum 50 rows required
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    n_samples: int
    n_features: int
    feature_names: List[str]
    label_column: str
    unique_labels: List[str]
    label_counts: Dict[str, int]
    has_missing: bool
    filepath: str


def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load a CSV dataset and return features, labels, feature names, unique labels.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    Tuple of:
        X : np.ndarray of shape (n_samples, n_features) — float64
        y : np.ndarray of shape (n_samples,) — string labels
        feature_names : list of column names (all except last)
        unique_labels : sorted list of unique label strings

    Raises
    ------
    FileNotFoundError : if file does not exist
    ValueError : if format is invalid
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Dataset not found: '{filepath}'"
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")

    validate_dataframe(df, filepath)

    feature_names = list(df.columns[:-1])
    label_column = df.columns[-1]

    X = df[feature_names].values.astype(np.float64)
    y = df[label_column].astype(str).values
    unique_labels = sorted(list(set(y)))

    return X, y, feature_names, unique_labels


def validate_dataframe(df: pd.DataFrame, filepath: str = "") -> bool:
    """
    Validate a dataframe meets EdgeFlow requirements.

    Raises ValueError with clear message if any check fails.
    """
    # Must have at least 2 columns (1 feature + 1 label)
    if df.shape[1] < 2:
        raise ValueError(
            f"Dataset must have at least 2 columns (features + label). "
            f"Found {df.shape[1]} column."
        )

    # Minimum 50 rows
    if df.shape[0] < 50:
        raise ValueError(
            f"Dataset must have at least 50 rows. "
            f"Found {df.shape[0]} rows."
        )

    # No missing values
    if df.isnull().any().any():
        missing_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"Dataset has missing values in columns: {missing_cols}. "
            f"Please clean your data before training."
        )

    # Feature columns must be numeric
    feature_cols = df.columns[:-1]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Feature column '{col}' must be numeric. "
                f"Found dtype: {df[col].dtype}"
            )

    # Label column must have at least 2 unique values
    label_col = df.columns[-1]
    unique_labels = df[label_col].unique()
    if len(unique_labels) < 2:
        raise ValueError(
            f"Label column '{label_col}' must have at least 2 unique classes. "
            f"Found: {unique_labels}"
        )

    return True


def get_dataset_info(filepath: str) -> DatasetInfo:
    """
    Get information about a dataset without fully loading it for training.
    Useful for quick inspection before training.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found: '{filepath}'")

    df = pd.read_csv(filepath)
    validate_dataframe(df, filepath)

    label_column = df.columns[-1]
    feature_names = list(df.columns[:-1])
    unique_labels = sorted(df[label_column].astype(str).unique().tolist())
    label_counts = df[label_column].astype(str).value_counts().to_dict()

    return DatasetInfo(
        n_samples=len(df),
        n_features=len(feature_names),
        feature_names=feature_names,
        label_column=label_column,
        unique_labels=unique_labels,
        label_counts=label_counts,
        has_missing=df.isnull().any().any(),
        filepath=filepath,
    )
