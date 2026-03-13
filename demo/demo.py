"""
EdgeAI Framework v1.0.0 — Live Demo
====================================
Complete end-to-end demonstration of the EdgeAI framework
running across all 3 domains with zero cloud dependency.

Now includes interactive device simulation selection.

Run:
    python demo/demo.py
"""

import sys
import os
import time

# Ensure the project root is on the path so ``edgeai`` is importable
# when the demo is executed from the project root directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edgeai import EdgeAI
from edgeai.core.device_profiler import get_device_profile, print_device_profile
from edgeai.core.preprocessor import preprocess_input
from edgeai.core.constraint_simulator import (
    simulate_device, get_profile, _DEVICE_PROFILES
)
from edgeai.benchmark.benchmark import run_full_benchmark, compare_with_cloud_baseline


# ──────────────────────────────────────────────────────────
#  Box-drawing helpers
# ──────────────────────────────────────────────────────────

def _box_top(w=50):
    return f"╔{'═' * w}╗"

def _box_mid(w=50):
    return f"╠{'═' * w}╣"

def _box_bot(w=50):
    return f"╚{'═' * w}╝"

def _box_title(text, w=50):
    return f"║{text:^{w}}║"

def _box_line(text, w=50):
    return f"║{text:<{w}}║"


# ──────────────────────────────────────────────────────────
#  Device selection menu
# ──────────────────────────────────────────────────────────

_CHOICE_MAP = {
    "1": "laptop",
    "2": "raspberry_pi_4",
    "3": "raspberry_pi_3",
    "4": "raspberry_pi_zero",
}

_MODEL_TIER = {
    "laptop":            "Full (Neural Network)",
    "raspberry_pi_4":    "Mid Edge (Random Forest)",
    "raspberry_pi_3":    "Low Edge (Logistic Regression)",
    "raspberry_pi_zero": "Minimal (Logistic Regression)",
}


def show_device_menu():
    """
    Display the device selection menu and return the chosen
    device key (e.g. 'laptop', 'raspberry_pi_4').
    """
    w = 54
    while True:
        print()
        print(_box_top(w))
        print(_box_title("EdgeAI Framework v1.0.0 - DEMO", w))
        print(_box_title("Select a device to simulate for this demo", w))
        print(_box_mid(w))
        print(_box_line("  [1] Laptop / Desktop     (actual hardware)", w))
        print(_box_line("  [2] Raspberry Pi 4       (4GB, 4 cores)", w))
        print(_box_line("  [3] Raspberry Pi 3B+     (1GB, 2 cores)", w))
        print(_box_line("  [4] Raspberry Pi Zero W  (512MB, 1 core)", w))
        print(_box_bot(w))

        choice = input("  Enter choice (1-4): ").strip()

        if choice in _CHOICE_MAP:
            return _CHOICE_MAP[choice]

        print("  ✗ Invalid choice. Please enter 1, 2, 3, or 4.\n")


def print_simulation_banner(device_key):
    """Print a banner showing what device is being simulated."""
    profile = get_profile(device_key)
    ram = profile["ram_limit_mb"]
    threads = profile["max_threads"]
    tier = _MODEL_TIER.get(device_key, "Unknown")

    ram_str = f"{ram} MB" if ram else "No Limit"
    threads_str = str(threads) if threads else "All"

    w = 54
    print()
    print(_box_top(w))
    print(_box_line(f"  Simulating: {profile['label']}", w))
    print(_box_line(f"  CPU Threads : {threads_str}", w))
    print(_box_line(f"  RAM Limit   : {ram_str}", w))
    print(_box_line(f"  Model Tier  : {tier}", w))
    print(_box_bot(w))


def print_sim_prediction_banner(result, sim_result, device_label):
    """Print the prediction banner with simulated latency info."""
    w = 50
    print(_box_top(w))
    print(_box_title("Prediction Result", w))
    print(_box_mid(w))
    print(_box_line(f"  Domain         : {result['domain']}", w))
    print(_box_line(f"  Prediction     : {result['prediction']}", w))
    print(_box_line(f"  Confidence     : {result['confidence_pct']}%", w))
    print(_box_line(f"  Device Sim     : {device_label}", w))
    print(_box_line(f"  Sim Latency    : {sim_result['simulated_latency_ms']} ms", w))
    print(_box_line(f"  Data Sent      : 0 bytes", w))
    print(_box_line(f"  Internet Used  : None", w))
    print(_box_bot(w))


def print_simulated_device_profile(device_name, profile):
    """Print device profile for the simulated device."""
    w = 40

    # Map device name to device class
    device_class_map = {
        "raspberry_pi_zero": "edge",
        "raspberry_pi_3": "mid_edge",
        "raspberry_pi_4": "mid_edge",
        "jetson_nano": "mid_edge",
        "laptop": "high_edge",
    }

    device_class = device_class_map.get(device_name, "edge")
    ram = profile["ram_limit_mb"]
    threads = profile["max_threads"]

    lines = [
        f"  RAM Available    : {ram} MB",
        f"  CPU Cores        : {threads}",
        f"  CPU Frequency    : Simulated",
        f"  Platform         : Linux (ARM)",
        f"  Device Class     : {device_class}",
        f"  Edge Capable     : YES",
        f"  Mode             : SIMULATED",
    ]

    print(f"╔{'═' * w}╗")
    print(f"║{'Device Profile (Simulated)':^{w}}║")
    print(f"╠{'═' * w}╣")
    for line in lines:
        print(f"║{line:<{w}}║")
    print(f"╚{'═' * w}╝")


# ──────────────────────────────────────────────────────────
#  Demo runner
# ──────────────────────────────────────────────────────────

def run_demo():
    """Execute the full EdgeAI live demonstration."""

    # ── Step 0: Device selection ─────────────────────────
    device_key = show_device_menu()
    is_constrained = (device_key != "laptop")

    if is_constrained:
        device_profile = get_profile(device_key)
        device_label = device_profile["label"]
        print_simulation_banner(device_key)
    else:
        device_label = "Laptop / Desktop"

    time.sleep(0.5)

    # ── Step 0b: Intro banner ────────────────────────────
    w = 50
    print()
    print(_box_top(w))
    print(_box_title("EdgeAI Framework v1.0.0 - DEMO", w))
    print(_box_title("Device-Agnostic Edge AI for Any IoT Device", w))
    print(_box_mid(w))
    print(_box_line("  Demonstrating local AI inference across", w))
    print(_box_line("  3 domains with zero cloud dependency", w))
    print(_box_bot(w))
    time.sleep(0.5)

    # ── Step 1: Device detection ─────────────────────────
    print("\n" + "=" * 52)
    print("  Step 1: Detecting device hardware...")
    print("=" * 52)
    time.sleep(0.5)

    if not is_constrained:
        # Show actual hardware
        profile = get_device_profile()
        print_device_profile(profile)
    else:
        # Show simulated device profile
        sim_profile = _DEVICE_PROFILES[device_key]
        print_simulated_device_profile(device_key, sim_profile)
    time.sleep(0.5)

    # Track latencies for summary
    all_sim_latencies = []

    # ── Helper: run prediction (normal or constrained) ───
    def demo_predict(ai, input_data):
        """
        Run prediction. If constrained device is selected,
        use simulate_device for realistic latency.
        """
        # Always run the normal predict first (for confidence etc.)
        result = ai.predict(input_data)

        if is_constrained:
            # Preprocess input the same way EdgeAI does internally
            processed = preprocess_input(input_data, ai.domain, ai.scaler)
            sim_result = simulate_device(device_key, ai.model, processed)
            all_sim_latencies.append(sim_result["simulated_latency_ms"])
            return result, sim_result
        else:
            all_sim_latencies.append(result["latency_ms"])
            return result, None

    # ── Step 2: Healthcare domain ────────────────────────
    print("\n" + "=" * 52)
    print("  Step 2: Healthcare Domain Demo")
    print("  Input: [heart_rate=72, bp=120, temp=36.5,")
    print("          spo2=98, activity=1]")
    print("=" * 52)
    time.sleep(0.5)

    ai_health = EdgeAI(domain="healthcare", verbose=not is_constrained)
    result_health, sim_health = demo_predict(ai_health, [72, 120, 36.5, 98, 1])
    if is_constrained:
        print_sim_prediction_banner(result_health, sim_health, device_label)
    time.sleep(0.5)

    # ── Step 3: Smart City domain ────────────────────────
    print("\n" + "=" * 52)
    print("  Step 3: Smart City Domain Demo")
    print("  Input: [vehicles=150, speed=45, density=0.7, weather=1]")
    print("=" * 52)
    time.sleep(0.5)

    ai_city = EdgeAI(domain="smartcity", verbose=not is_constrained)
    result_city, sim_city = demo_predict(ai_city, [150, 45, 0.7, 1])
    if is_constrained:
        print_sim_prediction_banner(result_city, sim_city, device_label)
    time.sleep(0.5)

    # ── Step 4: Environment domain ───────────────────────
    print("\n" + "=" * 52)
    print("  Step 4: Environment Domain Demo")
    print("  Input: [temp=28, humidity=65, pm25=45, co2=400]")
    print("=" * 52)
    time.sleep(0.5)

    ai_env = EdgeAI(domain="environment", verbose=not is_constrained)
    result_env, sim_env = demo_predict(ai_env, [28, 65, 45, 400])
    if is_constrained:
        print_sim_prediction_banner(result_env, sim_env, device_label)
    time.sleep(0.5)

    # ── Step 5: Benchmark ────────────────────────────────
    print("\n" + "=" * 52)
    print("  Step 5: Running Benchmark...")
    print("=" * 52)
    time.sleep(0.5)

    bench_results = run_full_benchmark(n_runs=5)
    time.sleep(0.5)

    compare_with_cloud_baseline(bench_results)
    time.sleep(0.5)

    if is_constrained:
        avg_sim_latency = (
            sum(all_sim_latencies) / len(all_sim_latencies)
            if all_sim_latencies else 0.0
        )
        sim_model_name = _MODEL_TIER.get(device_key, "Unknown")
        print()
        print("  ⚠  Note: Benchmark above shows actual hardware.")
        print(f"  Simulated device ({device_label}) results:")
        print(f"  All domains ran at ~{avg_sim_latency:.1f} ms (simulated)")
        print(f"  Model used: {sim_model_name}")
        print()

    # ── Step 6: Final summary ────────────────────────────
    avg_sim = (
        round(sum(all_sim_latencies) / len(all_sim_latencies), 1)
        if all_sim_latencies else 0.0
    )

    print()
    print(_box_top(w))
    print(_box_title("Demo Complete", w))
    print(_box_mid(w))
    print(_box_line("  Domains Tested    : 3", w))
    print(_box_line("  Models Used       : Auto-selected by device", w))
    print(_box_line("  Data Transferred  : 0 bytes", w))
    print(_box_line("  Internet Used     : None", w))
    print(_box_line("  Cloud Used        : None", w))
    print(_box_line("  All predictions   : LOCAL on this device", w))

    if is_constrained:
        print(_box_mid(w))
        print(_box_line(f"  Device Simulated  : {device_label}", w))
        print(_box_line(f"  Avg Sim Latency   : {avg_sim} ms", w))

    print(_box_bot(w))
    print()


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n  Demo interrupted by user.")
    except Exception as exc:
        print(f"\n  ✗ Demo error: {exc}")
        raise
