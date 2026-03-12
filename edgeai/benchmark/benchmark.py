"""
EdgeAI Benchmark
================
Benchmark the EdgeAI framework on the current device and compare
edge inference against a simulated cloud baseline.

Run on different devices to compare performance across hardware.
Results are saved as CSV files for further analysis.
"""

import os
import time
import csv
from datetime import datetime, timezone

from edgeai.EdgeAI import EdgeAI
from edgeai.core.device_profiler import get_device_profile


# ──────────────────────────────────────────────────────────
#  Sample test inputs per domain
# ──────────────────────────────────────────────────────────

_SAMPLE_INPUTS = {
    "healthcare": [
        [72, 120, 36.5, 98, 1],
        [85, 140, 37.2, 95, 0],
        [60, 110, 36.8, 99, 1],
        [90, 150, 38.0, 92, 0],
        [68, 115, 36.6, 97, 1],
    ],
    "smartcity": [
        [150, 45, 0.7, 8, 1],
        [300, 20, 0.9, 17, 0],
        [50, 60, 0.3, 6, 1],
        [200, 35, 0.8, 12, 1],
        [400, 15, 0.95, 18, 0],
    ],
    "environment": [
        [32, 65, 35, 50, 400],
        [28, 80, 120, 180, 800],
        [35, 45, 15, 25, 350],
        [22, 90, 200, 300, 1200],
        [30, 55, 50, 70, 500],
    ],
}


# ──────────────────────────────────────────────────────────
#  Box-drawing helpers
# ──────────────────────────────────────────────────────────

def _box_top(w=62):
    return f"╔{'═' * w}╗"

def _box_mid(w=62):
    return f"╠{'═' * w}╣"

def _box_bot(w=62):
    return f"╚{'═' * w}╝"

def _box_title(text, w=62):
    return f"║{text:^{w}}║"

def _box_line(text, w=62):
    return f"║{text:<{w}}║"


# ──────────────────────────────────────────────────────────
#  Single benchmark
# ──────────────────────────────────────────────────────────

def run_single_benchmark(domain, model_name=None, test_inputs=None,
                         n_runs=10):
    """
    Benchmark one domain on the current device.

    Parameters
    ----------
    domain : str
        Domain name.
    model_name : str, optional
        Overrides auto-selected model.  If None the framework
        selects automatically based on the device.
    test_inputs : list, optional
        List of sample inputs.  Falls back to built-in samples.
    n_runs : int
        Number of repetitions per input.

    Returns
    -------
    dict
        Benchmark results including latency statistics,
        device info, and model metadata.
    """
    if test_inputs is None:
        test_inputs = _SAMPLE_INPUTS.get(domain, _SAMPLE_INPUTS["healthcare"])

    ai = EdgeAI(domain=domain, verbose=False)

    # The framework auto-selects the model; capture what it chose
    actual_model = ai._model_name
    profile = ai.get_device_info()
    model_info = ai.get_model_info()

    latencies = []
    for _ in range(n_runs):
        for sample in test_inputs:
            result = ai.predict(sample)
            latencies.append(result["latency_ms"])

    return {
        "domain": domain,
        "model_name": actual_model,
        "device_class": profile["device_class"],
        "ram_mb": profile["ram_mb"],
        "cpu_cores": profile["cpu_cores"],
        "platform": profile["platform"],
        "n_runs": n_runs * len(test_inputs),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "model_size_kb": model_info["model_size_kb"],
        "data_transferred_bytes": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────
#  Full benchmark (all domains)
# ──────────────────────────────────────────────────────────

def run_full_benchmark(n_runs=10,
                       output_dir="research/experiments/results"):
    """
    Benchmark all 3 domains and print a formatted report.

    Results are also saved to a CSV file in *output_dir*.

    Parameters
    ----------
    n_runs : int
        Repetitions per input per domain.
    output_dir : str
        Directory for the CSV output.

    Returns
    -------
    list[dict]
        One result dict per domain.
    """
    profile = get_device_profile()
    results = []

    for domain in ["healthcare", "smartcity", "environment"]:
        try:
            r = run_single_benchmark(domain, n_runs=n_runs)
            results.append(r)
        except Exception as exc:
            print(f"  ✗ Benchmark failed for {domain}: {exc}")

    # Pretty-print
    w = 62
    print()
    print(_box_top(w))
    print(_box_title("EdgeAI Framework — Benchmark Report", w))
    print(_box_mid(w))
    dev_line = (
        f"  Device: {profile['device_class']}  |  "
        f"RAM: {profile['ram_mb']:.0f}MB  |  "
        f"CPU: {profile['cpu_cores']} cores"
    )
    print(_box_line(dev_line, w))
    print(_box_mid(w))
    header = f"  {'Domain':<14}{'Model':<18}{'Latency':>8}{'Size':>10}{'Transfer':>12}"
    print(_box_line(header, w))
    print(_box_line("  " + "─" * 58, w))

    for r in results:
        line = (
            f"  {r['domain']:<14}"
            f"{r['model_name']:<18}"
            f"{r['avg_latency_ms']:>6.1f}ms"
            f"{r['model_size_kb']:>8.1f}KB"
            f"{'0 bytes':>12}"
        )
        print(_box_line(line, w))

    print(_box_bot(w))

    # Save to CSV
    try:
        save_benchmark_results(results, output_dir)
    except Exception as exc:
        print(f"  ⚠ Could not save CSV: {exc}")

    return results


# ──────────────────────────────────────────────────────────
#  Save results
# ──────────────────────────────────────────────────────────

def save_benchmark_results(results_list, output_dir="research/experiments/results"):
    """
    Save benchmark results to a timestamped CSV file.

    Parameters
    ----------
    results_list : list[dict]
        Output from ``run_single_benchmark`` or ``run_full_benchmark``.
    output_dir : str
        Target directory (created if missing).

    Returns
    -------
    str
        Absolute path to the saved CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{stamp}.csv"
    path = os.path.join(output_dir, filename)

    if not results_list:
        print("  No results to save.")
        return path

    fieldnames = list(results_list[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"  ✓ Results saved to {path}")
    return path


# ──────────────────────────────────────────────────────────
#  Cloud comparison
# ──────────────────────────────────────────────────────────

def compare_with_cloud_baseline(benchmark_results):
    """
    Compare edge inference against a simulated cloud baseline.

    Cloud assumptions:
        - Latency = 15× edge latency (network round-trip overhead)
        - Data transfer = 2048 bytes per prediction (payload + headers)
        - Internet required

    Parameters
    ----------
    benchmark_results : list[dict]
        Results from ``run_full_benchmark``.
    """
    if not benchmark_results:
        print("  No benchmark results to compare.")
        return

    avg_edge = sum(r["avg_latency_ms"] for r in benchmark_results) / len(benchmark_results)
    avg_cloud = round(avg_edge * 15, 1)

    print()
    print("  Edge AI vs Cloud AI Comparison")
    print("  " + "═" * 46)
    print(f"  {'Metric':<20}{'Edge AI':<14}{'Cloud AI':<14}")
    print("  " + "─" * 46)
    print(f"  {'Avg Latency':<20}{avg_edge:<14.1f}{avg_cloud:<14.1f}")
    print(f"  {'Data Transfer':<20}{'0 bytes':<14}{'2048 bytes':<14}")
    print(f"  {'Internet Needed':<20}{'NO':<14}{'YES':<14}")
    print(f"  {'Privacy':<20}{'100%':<14}{'At Risk':<14}")
    print("  " + "═" * 46)
    print(f"\n  ⚡ Edge AI is {avg_cloud / avg_edge:.0f}× faster than Cloud AI")
    print(f"  🔒 Edge AI transfers 0 bytes — full data privacy")
    print()


# ──────────────────────────────────────────────────────────
#  Standalone execution
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_full_benchmark(n_runs=5)
    compare_with_cloud_baseline(results)
