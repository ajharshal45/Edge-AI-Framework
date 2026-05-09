"""
EdgeFlow Model Analyzer
=======================
Estimates memory usage and storage requirements for a trained model
before export, so users know whether it fits on the target device.

Used internally by the validator and by the CLI 'validate' command.

Usage:
    from edgeflow.core.analyzer import ModelAnalyzer

    analyzer = ModelAnalyzer()
    result = analyzer.analyze(
        model_type="decision_tree",
        target_device="esp32",
        n_features=2,
        n_classes=2,
    )
    print(result["estimated_ram_kb"])   # e.g. 3.4
    print(result["fits_target"])        # True
"""

from typing import Dict, Optional

from edgeflow.devices.device_profiles import get_device_profile


# ---------------------------------------------------------------------------
# Base size constants (KB) — measured from typical small-dataset exports
# These are the minimum sizes with zero features/classes overhead
# ---------------------------------------------------------------------------
_BASE_RAM_KB = {
    "logistic":            0.5,
    "decision_tree":       1.0,
    "naive_bayes":         0.8,
    "random_forest":      12.0,
    "random_forest_small": 6.0,
    "mlp":                 4.0,
}

_BASE_FLASH_KB = {
    "logistic":            1.0,
    "decision_tree":       2.0,
    "naive_bayes":         1.5,
    "random_forest":      30.0,
    "random_forest_small": 14.0,
    "mlp":                 8.0,
}

# Per-feature overhead (KB added per additional feature)
_RAM_PER_FEATURE = {
    "logistic":       0.05,
    "decision_tree":  0.25,
    "naive_bayes":    0.08,
    "random_forest":  1.50,
    "random_forest_small": 0.60,
    "mlp":            0.30,
}

_FLASH_PER_FEATURE = {
    "logistic":       0.10,
    "decision_tree":  0.50,
    "naive_bayes":    0.15,
    "random_forest":  3.00,
    "random_forest_small": 1.20,
    "mlp":            0.60,
}

# Per-class overhead (KB added per additional class)
_RAM_PER_CLASS = {
    "logistic":       0.05,
    "decision_tree":  0.05,
    "naive_bayes":    0.10,
    "random_forest":  0.20,
    "random_forest_small": 0.10,
    "mlp":            0.15,
}

_FLASH_PER_CLASS = {
    "logistic":       0.10,
    "decision_tree":  0.10,
    "naive_bayes":    0.20,
    "random_forest":  0.40,
    "random_forest_small": 0.20,
    "mlp":            0.30,
}


class ModelAnalyzer:
    """
    Estimates RAM and Flash usage for a model before deployment.

    Size estimates account for:
    - Base model overhead (struct, pointers, runtime state)
    - Per-feature scaling parameters (mean + scale stored in Flash)
    - Per-class parameters (weights, priors, or tree nodes)

    These are conservative upper-bound estimates — actual sizes
    on device may be slightly lower after compiler optimization.
    """

    def analyze(
        self,
        model_type: str,
        target_device: str,
        n_features: int = 2,
        n_classes: int = 2,
    ) -> Dict:
        """
        Estimate model memory usage and check device fit.

        Parameters
        ----------
        model_type : str
            One of 'logistic', 'decision_tree', 'naive_bayes', 'random_forest'
        target_device : str
            One of 'esp32', 'raspberry_pi', 'linux'
        n_features : int
            Number of input features in the model
        n_classes : int
            Number of output classes

        Returns
        -------
        dict with keys:
            estimated_ram_kb   : float  — RAM used during inference
            estimated_flash_kb : float  — Flash/storage for model weights
            fits_target        : bool   — True if within device limits
            device_ram_limit_kb   : float or None
            device_flash_limit_kb : float or None
            recommendation     : str   — human-readable summary
            details            : dict  — breakdown of how estimate was computed
        """
        profile = get_device_profile(target_device)

        # Calculate estimates
        base_ram   = _BASE_RAM_KB.get(model_type, 2.0)
        base_flash = _BASE_FLASH_KB.get(model_type, 4.0)

        feature_ram   = _RAM_PER_FEATURE.get(model_type, 0.1) * n_features
        feature_flash = _FLASH_PER_FEATURE.get(model_type, 0.2) * n_features

        class_ram   = _RAM_PER_CLASS.get(model_type, 0.1) * n_classes
        class_flash = _FLASH_PER_CLASS.get(model_type, 0.2) * n_classes

        # Scaler parameters always stored in Flash: 2 floats (mean + scale) per feature
        scaler_flash = (n_features * 2 * 4) / 1024  # bytes → KB

        total_ram_kb   = round(base_ram   + feature_ram   + class_ram,   2)
        total_flash_kb = round(base_flash + feature_flash + class_flash + scaler_flash, 2)

        # Device limits
        max_model_kb  = profile.get("max_model_size_kb")   # practical RAM limit for model
        flash_limit   = profile.get("flash_kb")
        ram_limit     = profile.get("ram_kb")

        # Fit check: use max_model_size_kb as primary RAM gating for ESP32
        fits_ram   = True
        fits_flash = True

        if max_model_kb is not None:
            fits_ram = total_ram_kb <= max_model_kb
        if flash_limit is not None:
            fits_flash = total_flash_kb <= flash_limit

        fits_target = fits_ram and fits_flash

        # Build recommendation string
        if fits_target:
            recommendation = (
                f"✅ {model_type} fits on {profile['name']}. "
                f"RAM: {total_ram_kb} KB | Flash: {total_flash_kb} KB"
            )
        else:
            parts = []
            if not fits_ram:
                parts.append(
                    f"RAM too large ({total_ram_kb} KB > limit {max_model_kb} KB)"
                )
            if not fits_flash:
                parts.append(
                    f"Flash too large ({total_flash_kb} KB > limit {flash_limit} KB)"
                )
            recommendation = (
                f"⚠️  {model_type} may NOT fit on {profile['name']}: "
                + " | ".join(parts)
            )

        return {
            "estimated_ram_kb":      total_ram_kb,
            "estimated_flash_kb":    total_flash_kb,
            "fits_target":           fits_target,
            "device_ram_limit_kb":   ram_limit,
            "device_flash_limit_kb": flash_limit,
            "recommendation":        recommendation,
            "details": {
                "base_ram_kb":       base_ram,
                "feature_ram_kb":    round(feature_ram, 3),
                "class_ram_kb":      round(class_ram, 3),
                "base_flash_kb":     base_flash,
                "feature_flash_kb":  round(feature_flash, 3),
                "class_flash_kb":    round(class_flash, 3),
                "scaler_flash_kb":   round(scaler_flash, 3),
                "n_features":        n_features,
                "n_classes":         n_classes,
            },
        }

    def compare_models(
        self,
        target_device: str,
        n_features: int = 2,
        n_classes: int = 2,
    ) -> Dict[str, Dict]:
        """
        Analyze all supported models for a device and return comparison dict.

        Returns
        -------
        dict mapping model_name → analysis result dict
        """
        from edgeflow.devices.device_profiles import get_device_profile
        profile = get_device_profile(target_device)
        all_models = profile.get("supported_models", []) + profile.get("experimental_models", [])

        results = {}
        for model in all_models:
            results[model] = self.analyze(model, target_device, n_features, n_classes)
        return results
