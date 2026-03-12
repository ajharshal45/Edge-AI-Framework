"""
EdgeAI Model Selector
=====================
Automatically selects the best lightweight model based on the
detected device class.  No manual model choice — the framework
decides the optimal model for every device.

Selection Map:
    ultra_edge  → Logistic Regression
    edge        → Logistic Regression
    mid_edge    → Random Forest
    high_edge   → Neural Network (MLP)
"""


# ──────────────────────────────────────────────────────────
#  Selection map
# ──────────────────────────────────────────────────────────

_MODEL_MAP = {
    "ultra_edge": {
        "model_name": "logistic",
        "reason": "Ultra low RAM detected",
    },
    "edge": {
        "model_name": "logistic",
        "reason": "Low RAM detected",
    },
    "mid_edge": {
        "model_name": "random_forest",
        "reason": "Sufficient RAM for ensemble",
    },
    "high_edge": {
        "model_name": "neural_network",
        "reason": "High performance device",
    },
}


# ──────────────────────────────────────────────────────────
#  Model descriptions
# ──────────────────────────────────────────────────────────

_MODEL_DESCRIPTIONS = {
    "logistic": {
        "name": "Logistic Regression",
        "parameters": "~1k-5k",
        "size_estimate": "< 50 KB",
        "suitable_for": "Ultra constrained devices",
    },
    "random_forest": {
        "name": "Random Forest",
        "parameters": "~10k-50k",
        "size_estimate": "< 500 KB",
        "suitable_for": "Mid range edge devices",
    },
    "neural_network": {
        "name": "Neural Network",
        "parameters": "~50k-200k",
        "size_estimate": "< 2 MB",
        "suitable_for": "High performance edge devices",
    },
}


# ──────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────

def select_model(device_class):
    """
    Select the best model for the given device class.

    The framework never asks the user to choose a model —
    it picks the optimal one based on detected hardware.

    Args:
        device_class (str): One of 'ultra_edge', 'edge',
                            'mid_edge', 'high_edge'.

    Returns:
        dict: Keys — model_name, reason, device_class.

    Raises:
        ValueError: If device_class is not recognised.
    """
    if device_class not in _MODEL_MAP:
        valid = ", ".join(_MODEL_MAP.keys())
        raise ValueError(
            f"Unknown device class '{device_class}'. "
            f"Valid classes: {valid}"
        )

    entry = _MODEL_MAP[device_class]
    return {
        "model_name": entry["model_name"],
        "reason": entry["reason"],
        "device_class": device_class,
    }


def get_model_description(model_name):
    """
    Return a human-readable description of a model type.

    Args:
        model_name (str): One of 'logistic', 'random_forest',
                          'neural_network'.

    Returns:
        dict: Keys — name, parameters, size_estimate, suitable_for.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name not in _MODEL_DESCRIPTIONS:
        valid = ", ".join(_MODEL_DESCRIPTIONS.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid models: {valid}"
        )

    return dict(_MODEL_DESCRIPTIONS[model_name])


# ──────────────────────────────────────────────────────────
#  Pretty-printed output
# ──────────────────────────────────────────────────────────

def print_model_selection(selection):
    """
    Print the model selection result in a box-drawn format.

    Args:
        selection (dict): Dictionary returned by select_model().

    Example output::

        ╔════════════════════════════════════════╗
        ║         Model Selection                ║
        ╠════════════════════════════════════════╣
        ║  Device Class     : mid_edge           ║
        ║  Model Selected   : Random Forest      ║
        ║  Reason           : Sufficient RAM     ║
        ║  Est. Model Size  : < 500 KB           ║
        ║  Parameters       : ~10k-50k           ║
        ╚════════════════════════════════════════╝
    """
    w = 40  # inner width

    desc = get_model_description(selection["model_name"])

    lines = [
        f"  Device Class     : {selection['device_class']}",
        f"  Model Selected   : {desc['name']}",
        f"  Reason           : {selection['reason']}",
        f"  Est. Model Size  : {desc['size_estimate']}",
        f"  Parameters       : {desc['parameters']}",
    ]

    print(f"╔{'═' * w}╗")
    print(f"║{'Model Selection':^{w}}║")
    print(f"╠{'═' * w}╣")
    for line in lines:
        print(f"║{line:<{w}}║")
    print(f"╚{'═' * w}╝")


# ──────────────────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    for dc in ["ultra_edge", "edge", "mid_edge", "high_edge"]:
        sel = select_model(dc)
        print_model_selection(sel)
        print()
