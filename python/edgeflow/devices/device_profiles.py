# ============================================================
# EdgeFlow Framework v1.0
# python/edgeflow/devices/device_profiles.py
#
# Hardware profiles for all supported target devices.
# These limits are used by the validator to determine whether
# a trained model is safe to deploy on the chosen device.
# ============================================================

DEVICE_PROFILES = {
    "esp32": {
        "name": "ESP32 DevKit",
        "ram_kb": 320,           # usable SRAM
        "flash_kb": 4096,        # total flash
        "cpu_mhz": 240,
        "has_fpu": False,        # no floating point unit
        "supported_models": ["logistic", "decision_tree", "naive_bayes"],
        "experimental_models": ["random_forest_small"],
        "max_model_size_kb": 50, # safe limit for model in RAM
        "architecture": "xtensa_lx6",
    },
    "raspberry_pi": {
        "name": "Raspberry Pi 3/4",
        "ram_kb": 1024 * 1024,   # 1GB+
        "flash_kb": None,        # SD card, unlimited
        "cpu_mhz": 1400,
        "has_fpu": True,
        "supported_models": ["logistic", "decision_tree", "naive_bayes",
                             "random_forest", "mlp"],
        "experimental_models": [],
        "max_model_size_kb": None,
        "architecture": "arm_cortex_a",
    },
    "linux": {
        "name": "Generic Linux",
        "ram_kb": None,
        "flash_kb": None,
        "cpu_mhz": None,
        "has_fpu": True,
        "supported_models": ["logistic", "decision_tree", "naive_bayes",
                             "random_forest", "mlp"],
        "experimental_models": [],
        "max_model_size_kb": None,
        "architecture": "x86_64",
    },
}


def get_device_profile(device_name: str) -> dict:
    """
    Get hardware profile for a specific device.
    Raises ValueError if device not found.
    """
    if device_name not in DEVICE_PROFILES:
        valid = ", ".join(DEVICE_PROFILES.keys())
        raise ValueError(
            f"Unknown device '{device_name}'. Valid devices: {valid}"
        )
    return dict(DEVICE_PROFILES[device_name])


def list_devices() -> list:
    """Return list of all supported device names."""
    return list(DEVICE_PROFILES.keys())


def is_model_supported(device_name: str, model_name: str) -> bool:
    """Check if a model is supported on a device."""
    profile = get_device_profile(device_name)
    return (
        model_name in profile["supported_models"] or
        model_name in profile["experimental_models"]
    )
