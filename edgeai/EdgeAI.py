"""
EdgeAI — Main Public API
=========================
The single entry point for the entire framework.  Any developer
on any device uses these two lines::

    from edgeai import EdgeAI

    ai = EdgeAI(domain="healthcare")
    result = ai.predict([72, 120, 36.5, 98, 1])
    print(result)

Everything runs locally.  No cloud.  No internet.
"""

import os
import pickle
import time

from edgeai.core.device_profiler import get_device_profile
from edgeai.core.model_selector import select_model, get_model_description
from edgeai.core.preprocessor import preprocess_input
from edgeai.core.predictor import predict_local


# ──────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────

_VERSION = "1.0.0"
_VALID_DOMAINS = ["healthcare", "smartcity", "environment"]

_PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pretrained"
)


# ──────────────────────────────────────────────────────────
#  Helper — load pkl payload
# ──────────────────────────────────────────────────────────

def _load_pkl_payload(domain, model_name):
    """Load the full {model, scaler, …} dict from a .pkl file."""
    filename = f"{domain}_{model_name}.pkl"
    path = os.path.join(_PRETRAINED_DIR, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pretrained model not found: '{path}'. "
            f"Run the trainer first:  "
            f"python -m edgeai.models.trainer"
        )

    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────
#  Box-drawing helpers
# ──────────────────────────────────────────────────────────

def _box_top(w=42):
    return f"╔{'═' * w}╗"

def _box_mid(w=42):
    return f"╠{'═' * w}╣"

def _box_bot(w=42):
    return f"╚{'═' * w}╝"

def _box_title(text, w=42):
    return f"║{text:^{w}}║"

def _box_line(text, w=42):
    return f"║{text:<{w}}║"


# ══════════════════════════════════════════════════════════
#  EdgeAI class
# ══════════════════════════════════════════════════════════

class EdgeAI:
    """
    Lightweight Edge AI framework for any device.

    Parameters
    ----------
    domain : str
        One of ``'healthcare'``, ``'smartcity'``, ``'environment'``.
    verbose : bool, optional
        If True (default), print startup and prediction banners.

    Attributes
    ----------
    domain : str
    device_profile : dict
    model_selection : dict
    model : sklearn estimator
    scaler : sklearn StandardScaler
    """

    def __init__(self, domain, verbose=True):
        # --- Validate domain ---
        if domain not in _VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Valid domains: {', '.join(_VALID_DOMAINS)}"
            )

        self.domain = domain
        self.verbose = verbose

        # --- Device profiling ---
        self.device_profile = get_device_profile()

        # --- Model selection ---
        self.model_selection = select_model(
            self.device_profile["device_class"]
        )
        self._model_name = self.model_selection["model_name"]

        # --- Load pretrained model + scaler ---
        payload = _load_pkl_payload(domain, self._model_name)
        self.model = payload["model"]
        self.scaler = payload.get("scaler", None)
        self._accuracy = payload.get("accuracy", 0.0)

        # --- Model size on disk ---
        pkl_path = os.path.join(
            _PRETRAINED_DIR, f"{domain}_{self._model_name}.pkl"
        )
        self._model_size_kb = round(
            os.path.getsize(pkl_path) / 1024, 1
        )

        # --- Startup banner ---
        if self.verbose:
            self._print_startup_banner()

    # ──────────────────────────────────────────────────────
    #  predict
    # ──────────────────────────────────────────────────────

    def predict(self, input_data):
        """
        Run a local prediction on raw sensor data.

        Parameters
        ----------
        input_data : list, tuple, np.ndarray, or pd.DataFrame
            A single sample with the correct number of features
            for the chosen domain.

        Returns
        -------
        dict
            Prediction result with keys: domain, model_name,
            model_size_kb, prediction, confidence_pct,
            latency_ms, data_transferred_bytes, internet_used.
        """
        processed = preprocess_input(input_data, self.domain, self.scaler)

        result = predict_local(
            self.model, processed, self.domain, self._model_name
        )

        # Override model_size_kb with our cached value
        result["model_size_kb"] = self._model_size_kb

        if self.verbose:
            self._print_prediction_banner(result)

        return result

    # ──────────────────────────────────────────────────────
    #  get_device_info / get_model_info
    # ──────────────────────────────────────────────────────

    def get_device_info(self):
        """Return the device hardware profile dictionary."""
        return dict(self.device_profile)

    def get_model_info(self):
        """
        Return a summary of the loaded model.

        Returns
        -------
        dict
            Keys: domain, model_name, model_size_kb, device_class,
            accuracy.
        """
        return {
            "domain": self.domain,
            "model_name": self._model_name,
            "model_size_kb": self._model_size_kb,
            "device_class": self.device_profile["device_class"],
            "accuracy": self._accuracy,
        }

    # ──────────────────────────────────────────────────────
    #  benchmark
    # ──────────────────────────────────────────────────────

    def benchmark(self, test_data_list):
        """
        Benchmark prediction latency over a list of inputs.

        Parameters
        ----------
        test_data_list : list
            Each element is a single input sample (list / array).

        Returns
        -------
        dict
            Keys: total_predictions, avg_latency_ms,
            min_latency_ms, max_latency_ms,
            data_transferred_bytes.
        """
        old_verbose = self.verbose
        self.verbose = False

        latencies = []
        for sample in test_data_list:
            result = self.predict(sample)
            latencies.append(result["latency_ms"])

        self.verbose = old_verbose

        if not latencies:
            return {
                "total_predictions": 0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "data_transferred_bytes": 0,
            }

        return {
            "total_predictions": len(latencies),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            "min_latency_ms": round(min(latencies), 3),
            "max_latency_ms": round(max(latencies), 3),
            "data_transferred_bytes": 0,
        }

    # ──────────────────────────────────────────────────────
    #  Banners (private)
    # ──────────────────────────────────────────────────────

    def _print_startup_banner(self):
        w = 42
        dp = self.device_profile
        desc = get_model_description(self._model_name)

        print(_box_top(w))
        print(_box_title(f"EdgeAI Framework v{_VERSION}", w))
        print(_box_title("Lightweight Edge AI for Any Device", w))
        print(_box_mid(w))
        print(_box_line("  Device Profile", w))
        print(_box_line(f"  RAM            : {dp['ram_mb']:.0f} MB", w))
        print(_box_line(f"  CPU Cores      : {dp['cpu_cores']}", w))
        print(_box_line(f"  Platform       : {dp['platform']}", w))
        print(_box_line(f"  Device Class   : {dp['device_class']}", w))
        print(_box_mid(w))
        print(_box_line("  Model Ready", w))
        print(_box_line(f"  Domain         : {self.domain}", w))
        print(_box_line(f"  Model          : {desc['name']}", w))
        print(_box_line(f"  Reason         : {self.model_selection['reason']}", w))
        print(_box_line(f"  Model Size     : {self._model_size_kb} KB", w))
        print(_box_mid(w))
        print(_box_line("  Status         : READY", w))
        print(_box_line("  Internet Needed: NO", w))
        print(_box_line("  Cloud Needed   : NO", w))
        print(_box_bot(w))

    def _print_prediction_banner(self, result):
        w = 42

        print(_box_top(w))
        print(_box_title("Prediction Result", w))
        print(_box_mid(w))
        print(_box_line(f"  Domain         : {result['domain']}", w))
        print(_box_line(f"  Prediction     : {result['prediction']}", w))
        print(_box_line(f"  Confidence     : {result['confidence_pct']}%", w))
        print(_box_line(f"  Latency        : {result['latency_ms']} ms", w))
        print(_box_line(f"  Data Sent      : 0 bytes", w))
        print(_box_line(f"  Internet Used  : None", w))
        print(_box_bot(w))


# ──────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────

def main():
    """Console-script entry point for quick testing."""
    ai = EdgeAI(domain="healthcare")
    result = ai.predict([72, 120, 36.5, 98, 1])
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
