"""
EdgeFlow Device Validator
=========================
Checks whether a trained model is compatible with a target device.

Produces the formatted compatibility report shown in the EdgeFlow CLI:

    [EdgeFlow Validator]
    ═══════════════════════════════════════
    Target Device   : ESP32
    Model           : decision_tree
    Estimated RAM   : 3.2 KB
    Estimated Flash : 8.1 KB
    ESP32 RAM Limit : 320 KB
    ═══════════════════════════════════════
    ✅ Compatible — Safe to deploy

Usage:
    from edgeflow.core.validator import DeviceValidator

    validator = DeviceValidator()
    result = validator.validate("decision_tree", "esp32")
    print(result.compatible)       # True
    print(result.estimated_ram_kb) # 3.2
"""

from dataclasses import dataclass, field
from typing import List, Optional

from edgeflow.devices.device_profiles import (
    DEVICE_PROFILES,
    get_device_profile,
    is_model_supported,
)


# ---------------------------------------------------------------------------
# RAM and Flash size estimates per model type (in KB)
# Based on typical export sizes for small IoT datasets (2–10 features)
# ---------------------------------------------------------------------------
_RAM_ESTIMATES_KB = {
    "logistic":          1.1,
    "decision_tree":     3.2,
    "naive_bayes":       1.8,
    "random_forest":    42.0,
    "random_forest_small": 18.0,
    "mlp":              12.0,
}

_FLASH_ESTIMATES_KB = {
    "logistic":          2.0,
    "decision_tree":     8.1,
    "naive_bayes":       3.5,
    "random_forest":    85.0,
    "random_forest_small": 35.0,
    "mlp":              24.0,
}

# Alternative model recommendations when the selected one doesn't fit
_ALTERNATIVES = ["logistic", "naive_bayes", "decision_tree"]


@dataclass
class ValidationResult:
    """Result returned by DeviceValidator.validate()."""
    compatible: bool
    warnings: List[str]
    estimated_ram_kb: float
    estimated_size_kb: float        # flash / model storage size
    recommendation: str             # suggested alternative if incompatible
    target_device: str
    model_type: str
    device_name: str
    device_ram_limit_kb: Optional[float]
    device_flash_limit_kb: Optional[float]


class DeviceValidator:
    """
    Validates whether a trained model can be safely deployed on a target device.

    Checks:
    - Is the model in the device's supported_models list?
    - Is the model in experimental_models (warns but allows)?
    - Does estimated RAM usage fit within device RAM limit?
    - Does estimated Flash usage fit within device Flash limit?

    Prints a formatted compatibility report to stdout.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _estimate_ram_kb(self, model_type: str) -> float:
        return _RAM_ESTIMATES_KB.get(model_type, 10.0)

    def _estimate_flash_kb(self, model_type: str) -> float:
        return _FLASH_ESTIMATES_KB.get(model_type, 20.0)

    def _build_recommendation(
        self,
        model_type: str,
        target_device: str,
        profile: dict,
    ) -> str:
        """Build a recommendation string listing alternative models that do fit."""
        supported = profile.get("supported_models", [])
        alternatives = [m for m in _ALTERNATIVES if m in supported and m != model_type]
        if not alternatives:
            return "No supported alternatives found for this device."

        lines = ["Suggested alternatives:"]
        for alt in alternatives:
            ram = self._estimate_ram_kb(alt)
            lines.append(f"  -> {alt:<20} (estimated {ram} KB RAM)")
        return "\n".join(lines)

    def validate(self, model_type: str, target_device: str) -> ValidationResult:
        """
        Validate model compatibility with target device.

        Parameters
        ----------
        model_type : str
            One of 'logistic', 'decision_tree', 'naive_bayes', 'random_forest'
        target_device : str
            One of 'esp32', 'raspberry_pi', 'linux'

        Returns
        -------
        ValidationResult dataclass
        """
        profile = get_device_profile(target_device)
        device_name = profile["name"]

        estimated_ram_kb   = self._estimate_ram_kb(model_type)
        estimated_flash_kb = self._estimate_flash_kb(model_type)
        ram_limit   = profile.get("ram_kb")
        flash_limit = profile.get("flash_kb")
        max_model_kb = profile.get("max_model_size_kb")

        warnings: List[str] = []
        compatible = True

        # Check 1: Is model officially supported?
        officially_supported = model_type in profile.get("supported_models", [])
        is_experimental      = model_type in profile.get("experimental_models", [])

        if not officially_supported and not is_experimental:
            compatible = False
            warnings.append(
                f"Model '{model_type}' is not supported on {device_name}. "
                f"Supported: {profile['supported_models']}"
            )

        if is_experimental:
            warnings.append(
                f"Model '{model_type}' is EXPERIMENTAL on {device_name}. "
                f"May exceed memory limits in practice."
            )

        # Check 2: RAM limit (use max_model_size_kb as the practical limit for model RAM)
        if max_model_kb is not None and estimated_ram_kb > max_model_kb:
            compatible = False
            warnings.append(
                f"Estimated RAM usage ({estimated_ram_kb} KB) exceeds "
                f"{device_name} model RAM limit ({max_model_kb} KB)."
            )

        # Check 3: Flash limit (only if device has a defined flash limit)
        if flash_limit is not None and estimated_flash_kb > flash_limit:
            compatible = False
            warnings.append(
                f"Estimated Flash usage ({estimated_flash_kb} KB) exceeds "
                f"{device_name} Flash ({flash_limit} KB)."
            )

        # Build recommendation
        if compatible:
            recommendation = f"[OK] Safe to deploy '{model_type}' on {device_name}."
        else:
            recommendation = self._build_recommendation(model_type, target_device, profile)

        result = ValidationResult(
            compatible=compatible,
            warnings=warnings,
            estimated_ram_kb=estimated_ram_kb,
            estimated_size_kb=estimated_flash_kb,
            recommendation=recommendation,
            target_device=target_device,
            model_type=model_type,
            device_name=device_name,
            device_ram_limit_kb=ram_limit,
            device_flash_limit_kb=flash_limit,
        )

        if self.verbose:
            self._print_report(result)

        return result

    def _print_report(self, result: ValidationResult):
        """Print the formatted validator report to stdout."""
        sep = "=" * 39

        # RAM limit display
        if result.device_ram_limit_kb is not None:
            ram_limit_str = f"{result.device_ram_limit_kb} KB"
        else:
            ram_limit_str = "Unlimited"

        device_label = result.device_name.upper()

        print()
        print("[EdgeFlow Validator]")
        print(sep)
        print(f"Target Device   : {result.device_name}")
        print(f"Model           : {result.model_type}")
        print(f"Estimated RAM   : {result.estimated_ram_kb} KB")
        print(f"Estimated Flash : {result.estimated_size_kb} KB")
        print(f"{device_label} RAM Limit : {ram_limit_str}")
        print(sep)

        if result.compatible:
            print("[OK] Compatible -- Safe to deploy")
        else:
            print("[WARNING] Model may not fit on target device")
            for w in result.warnings:
                print(f"   * {w}")
            print()
            print(result.recommendation)

        print()
