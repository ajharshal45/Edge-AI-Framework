"""
EdgeFlow Trainer
================
Core training engine for EdgeFlow framework.

Handles the complete training pipeline:
    1. Load dataset via loader.py
    2. Format and encode via formatter.py
    3. Train/test split
    4. Fit model (logistic, decision_tree, or naive_bayes)
    5. Evaluate accuracy
    6. Return TrainingReport

Usage:
    from edgeflow.core.trainer import EdgeFlowTrainer

    trainer = EdgeFlowTrainer(
        model="decision_tree",
        target="esp32",
        domain="safenest"
    )
    report = trainer.train("safenest_data.csv")
"""

import time
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from edgeflow.datasets.loader import load_csv
from edgeflow.datasets.formatter import EdgeFlowFormatter
from edgeflow.devices.device_profiles import get_device_profile, is_model_supported


@dataclass
class TrainingReport:
    """Complete report returned after training."""
    domain: str
    model_type: str
    target_device: str
    accuracy: float
    n_samples: int
    n_features: int
    n_classes: int
    feature_names: List[str]
    label_map: Dict[str, int]
    training_time_ms: float
    model_size_kb: float
    test_size: float
    random_state: int
    compatible_with_target: bool
    warnings: List[str] = field(default_factory=list)


class EdgeFlowTrainer:
    """
    Core training engine for EdgeFlow.

    Parameters
    ----------
    model : str
        One of 'logistic', 'decision_tree', 'naive_bayes', 'random_forest'
    target : str
        Target device: 'esp32', 'raspberry_pi', 'linux'
    domain : str
        User-defined domain name (e.g. 'safenest', 'environment')
    verbose : bool
        Print progress messages
    """

    SUPPORTED_MODELS = ["logistic", "decision_tree", "naive_bayes", "random_forest"]

    def __init__(
        self,
        model: str = "decision_tree",
        target: str = "esp32",
        domain: str = "custom",
        verbose: bool = True,
    ):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Valid models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        self.model_type = model
        self.target = target
        self.domain = domain
        self.verbose = verbose

        # These are set after train() is called
        self.fitted_model = None
        self.formatter: Optional[EdgeFlowFormatter] = None
        self.report: Optional[TrainingReport] = None
        self.is_trained: bool = False

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[EdgeFlow Trainer] {message}")

    def _get_model_instance(self):
        """Return a fresh sklearn model instance."""
        if self.model_type == "logistic":
            return LogisticRegression(
                solver="lbfgs",
                max_iter=200,
                random_state=42,
            )
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            )
        elif self.model_type == "naive_bayes":
            return GaussianNB()
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _estimate_model_size_kb(self) -> float:
        """
        Estimate model size in KB for the target device.
        These are conservative estimates based on model type.
        """
        estimates = {
            "logistic": 2.0,
            "naive_bayes": 1.5,
            "decision_tree": 8.0,
            "random_forest": 45.0,
        }
        return estimates.get(self.model_type, 10.0)

    def _check_device_compatibility(self) -> Tuple[bool, List[str]]:
        """
        Check if selected model is compatible with target device.
        Returns (is_compatible, list_of_warnings).
        """
        warnings = []
        profile = get_device_profile(self.target)
        compatible = is_model_supported(self.target, self.model_type)

        if not compatible:
            warnings.append(
                f"Model '{self.model_type}' is NOT officially supported on {self.target}. "
                f"Supported models: {profile['supported_models']}"
            )

        if self.model_type in profile.get("experimental_models", []):
            warnings.append(
                f"Model '{self.model_type}' is EXPERIMENTAL on {self.target}. "
                f"May exceed memory limits."
            )

        estimated_kb = self._estimate_model_size_kb()
        max_kb = profile.get("max_model_size_kb")
        if max_kb and estimated_kb > max_kb:
            warnings.append(
                f"Estimated model size ({estimated_kb} KB) may exceed "
                f"{self.target} limit ({max_kb} KB)."
            )

        return compatible, warnings

    def train(
        self,
        data: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainingReport:
        """
        Run the full training pipeline.

        Parameters
        ----------
        data : str
            Path to CSV training file.
        test_size : float
            Fraction of data for testing (default 0.2 = 20%)
        random_state : int
            For reproducibility.

        Returns
        -------
        TrainingReport dataclass with all training results.
        """
        self._log("Starting training pipeline...")
        self._log(f"Domain   : {self.domain}")
        self._log(f"Model    : {self.model_type}")
        self._log(f"Target   : {self.target}")

        # Step 1: Load data
        self._log("Loading dataset...")
        X, y, feature_names, unique_labels = load_csv(data)
        self._log(
            f"Loaded {len(X)} samples, "
            f"{len(feature_names)} features, "
            f"{len(unique_labels)} classes"
        )

        # Step 2: Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError:
            # Fallback without stratify if class counts are too small
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
            )

        # Step 3: Fit formatter (scaler + label encoder)
        self._log("Fitting formatter (scaler + label encoder)...")
        self.formatter = EdgeFlowFormatter()
        self.formatter.fit(y, X_train, feature_names)

        # Step 4: Encode labels and scale features
        y_train_enc = self.formatter.encode_labels(y_train)
        y_test_enc = self.formatter.encode_labels(y_test)
        X_train_scaled = self.formatter.scale_features(X_train)
        X_test_scaled = self.formatter.scale_features(X_test)

        # Step 5: Train model
        self._log(f"Training {self.model_type}...")
        model_instance = self._get_model_instance()

        start_time = time.perf_counter()
        model_instance.fit(X_train_scaled, y_train_enc)
        training_time_ms = (time.perf_counter() - start_time) * 1000

        # Step 6: Evaluate
        y_pred = model_instance.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_enc, y_pred)
        self._log(f"Accuracy : {accuracy:.1%}")

        # Step 7: Device compatibility check
        compatible, warnings = self._check_device_compatibility()
        for w in warnings:
            self._log(f"WARNING: {w}")

        # Step 8: Store trained model
        self.fitted_model = model_instance
        self.is_trained = True

        # Step 9: Build and return report
        estimated_size_kb = self._estimate_model_size_kb()
        self.report = TrainingReport(
            domain=self.domain,
            model_type=self.model_type,
            target_device=self.target,
            accuracy=round(accuracy, 4),
            n_samples=len(X),
            n_features=len(feature_names),
            n_classes=self.formatter.get_n_classes(),
            feature_names=feature_names,
            label_map=self.formatter.get_label_map(),
            training_time_ms=round(training_time_ms, 2),
            model_size_kb=estimated_size_kb,
            test_size=test_size,
            random_state=random_state,
            compatible_with_target=compatible,
            warnings=warnings,
        )

        self._log("-" * 40)
        self._log("Training complete!")
        self._log(f"Accuracy         : {accuracy:.1%}")
        self._log(f"Training time    : {training_time_ms:.1f} ms")
        self._log(f"Model size (est) : {estimated_size_kb} KB")
        self._log(f"Compatible       : {'YES' if compatible else 'NO - see warnings'}")
        self._log("-" * 40)

        return self.report

    def get_fitted_model(self):
        """Return the fitted sklearn model. Must call train() first."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self.fitted_model

    def get_formatter(self) -> EdgeFlowFormatter:
        """Return the fitted formatter. Must call train() first."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self.formatter
