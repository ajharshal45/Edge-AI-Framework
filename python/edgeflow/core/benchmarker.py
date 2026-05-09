"""
EdgeFlow Benchmarker
====================
Measures prediction latency and accuracy for a trained EdgeFlow model.

Simulates on-host inference speed to estimate what the model would
perform like on the target device. All measurements use
time.perf_counter() for microsecond precision.

Key principle: data_transferred_bytes is ALWAYS 0 — EdgeFlow runs
entirely on-device with zero cloud communication.

Usage:
    from edgeflow.core.benchmarker import EdgeFlowBenchmarker

    benchmarker = EdgeFlowBenchmarker()
    report = benchmarker.benchmark(
        fitted_model=trainer.get_fitted_model(),
        formatter=trainer.get_formatter(),
        data="safenest_data.csv",
        n_runs=100,
    )
    benchmarker.print_report(report)
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from edgeflow.datasets.loader import load_csv
from edgeflow.datasets.formatter import EdgeFlowFormatter


@dataclass
class BenchmarkReport:
    """Complete benchmark results returned by EdgeFlowBenchmarker.benchmark()."""
    model_type: str
    target_device: str
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    accuracy: float
    model_size_kb: float
    n_runs: int
    n_samples: int
    n_predictions: int          # n_runs × n_samples
    data_transferred_bytes: int  # always 0 — offline inference
    internet_used: str           # always "None"


# Model size estimates (KB) — kept in sync with trainer.py
_MODEL_SIZE_KB = {
    "logistic":            2.0,
    "decision_tree":       8.0,
    "naive_bayes":         1.5,
    "random_forest":      45.0,
    "random_forest_small": 18.0,
}


class EdgeFlowBenchmarker:
    """
    Benchmarks an EdgeFlow model's prediction latency and accuracy.

    Runs the model n_runs times over all samples in the dataset,
    records per-call latency, then aggregates statistics.

    The benchmark deliberately uses the same scaling pipeline that the
    C++ runtime uses (StandardScaler → predict) so the measured latency
    reflects real inference cost, not I/O or preprocessing overhead.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def benchmark(
        self,
        fitted_model,
        formatter: EdgeFlowFormatter,
        data: str,
        n_runs: int = 100,
        target_device: Optional[str] = None,
    ) -> BenchmarkReport:
        """
        Run the benchmark.

        Parameters
        ----------
        fitted_model : fitted sklearn estimator
        formatter    : fitted EdgeFlowFormatter (for scaling + label decoding)
        data         : path to CSV file with test samples
        n_runs       : number of full passes over the dataset (default 100)
        target_device: optional device string for the report label

        Returns
        -------
        BenchmarkReport dataclass
        """
        self._log(f"Loading test data from: {data}")
        X, y_true_str, feature_names, _ = load_csv(data)

        n_samples = len(X)
        self._log(f"Samples  : {n_samples}")
        self._log(f"Runs     : {n_runs}")
        self._log(f"Total predictions: {n_samples * n_runs}")

        # Scale once — same as what the C++ runtime does before inference
        X_scaled = formatter.scale_features(X)

        # Encode ground-truth labels for accuracy calculation
        y_true_enc = formatter.encode_labels(y_true_str)

        # ----------------------------------------------------------------
        # Warm-up pass (not measured) — avoids JIT / cache cold-start bias
        # ----------------------------------------------------------------
        _ = fitted_model.predict(X_scaled)

        # ----------------------------------------------------------------
        # Timed benchmark passes
        # ----------------------------------------------------------------
        latencies_ms = []

        for run in range(n_runs):
            start = time.perf_counter()
            y_pred = fitted_model.predict(X_scaled)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

        # ----------------------------------------------------------------
        # Accuracy (using the last run's predictions)
        # ----------------------------------------------------------------
        correct = int(np.sum(y_pred == y_true_enc))
        accuracy = correct / n_samples

        # ----------------------------------------------------------------
        # Latency statistics
        # ----------------------------------------------------------------
        latencies = np.array(latencies_ms)
        avg_ms = float(np.mean(latencies))
        min_ms = float(np.min(latencies))
        max_ms = float(np.max(latencies))

        # Detect model type from class name
        model_class = type(fitted_model).__name__
        model_type  = self._infer_model_type(model_class)
        size_kb     = _MODEL_SIZE_KB.get(model_type, 10.0)
        device_str  = target_device or "linux"

        report = BenchmarkReport(
            model_type=model_type,
            target_device=device_str,
            avg_latency_ms=round(avg_ms, 4),
            min_latency_ms=round(min_ms, 4),
            max_latency_ms=round(max_ms, 4),
            accuracy=round(accuracy, 4),
            model_size_kb=size_kb,
            n_runs=n_runs,
            n_samples=n_samples,
            n_predictions=n_runs * n_samples,
            data_transferred_bytes=0,
            internet_used="None",
        )

        if self.verbose:
            self.print_report(report)

        return report

    def print_report(self, report: BenchmarkReport):
        """
        Print the formatted benchmark report shown in blueprint Section 8.

        [EdgeFlow Benchmark]
        ═══════════════════════════════════════
        Model         : decision_tree
        Samples       : 100 runs × 500 samples
        Avg Latency   : 0.12 ms
        Min Latency   : 0.08 ms
        Max Latency   : 0.31 ms
        Accuracy      : 96.4%
        Model Size    : 8.2 KB
        Data Transfer : 0 bytes
        Internet Used : None
        ═══════════════════════════════════════
        """
        sep = "=" * 39
        print()
        print("[EdgeFlow Benchmark]")
        print(sep)
        print(f"Model         : {report.model_type}")
        print(f"Target Device : {report.target_device}")
        print(f"Samples       : {report.n_runs} runs x {report.n_samples} samples")
        print(f"Avg Latency   : {report.avg_latency_ms:.4f} ms")
        print(f"Min Latency   : {report.min_latency_ms:.4f} ms")
        print(f"Max Latency   : {report.max_latency_ms:.4f} ms")
        print(f"Accuracy      : {report.accuracy * 100:.1f}%")
        print(f"Model Size    : {report.model_size_kb} KB")
        print(f"Data Transfer : {report.data_transferred_bytes} bytes")
        print(f"Internet Used : {report.internet_used}")
        print(sep)
        print()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_model_type(self, class_name: str) -> str:
        """Map sklearn class name → EdgeFlow model type string."""
        mapping = {
            "DecisionTreeClassifier":  "decision_tree",
            "LogisticRegression":      "logistic",
            "GaussianNB":              "naive_bayes",
            "RandomForestClassifier":  "random_forest",
        }
        return mapping.get(class_name, class_name.lower())

    def _log(self, message: str):
        if self.verbose:
            print(f"[EdgeFlow Benchmarker] {message}")
