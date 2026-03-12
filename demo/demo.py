"""
EdgeAI Framework v1.0.0 — Live Demo
====================================
Complete end-to-end demonstration of the EdgeAI framework
running across all 3 domains with zero cloud dependency.

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
#  Demo runner
# ──────────────────────────────────────────────────────────

def run_demo():
    """Execute the full EdgeAI live demonstration."""

    # ── Step 0: Intro banner ─────────────────────────────
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

    profile = get_device_profile()
    print_device_profile(profile)
    time.sleep(0.5)

    # ── Step 2: Healthcare domain ────────────────────────
    print("\n" + "=" * 52)
    print("  Step 2: Healthcare Domain Demo")
    print("  Input: [heart_rate=72, bp=120, temp=36.5,")
    print("          spo2=98, activity=1]")
    print("=" * 52)
    time.sleep(0.5)

    ai_health = EdgeAI(domain="healthcare", verbose=True)
    result_health = ai_health.predict([72, 120, 36.5, 98, 1])
    time.sleep(0.5)

    # ── Step 3: Smart City domain ────────────────────────
    print("\n" + "=" * 52)
    print("  Step 3: Smart City Domain Demo")
    print("  Input: [vehicles=150, speed=45, density=0.7, weather=1]")
    print("=" * 52)
    time.sleep(0.5)

    ai_city = EdgeAI(domain="smartcity", verbose=True)
    result_city = ai_city.predict([150, 45, 0.7, 1])
    time.sleep(0.5)

    # ── Step 4: Environment domain ───────────────────────
    print("\n" + "=" * 52)
    print("  Step 4: Environment Domain Demo")
    print("  Input: [temp=28, humidity=65, pm25=45, co2=400]")
    print("=" * 52)
    time.sleep(0.5)

    ai_env = EdgeAI(domain="environment", verbose=True)
    result_env = ai_env.predict([28, 65, 45, 400])
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

    # ── Step 6: Final summary ────────────────────────────
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
