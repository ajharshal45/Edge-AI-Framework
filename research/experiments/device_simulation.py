"""
EdgeAI Framework — Multi-Device Simulation Report
=================================================
Runs the full benchmark simulating ALL device profiles
across all 3 domains.
"""

import os
import sys
import csv
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from edgeai.core.constraint_simulator import get_device_profiles, get_profile, simulate_device
from edgeai.core.device_profiler import classify_device, get_device_profile
from edgeai.core.model_selector import select_model
from edgeai.core.predictor import load_model
from edgeai.core.preprocessor import preprocess_input


_DOMAINS = ["healthcare", "smartcity", "environment"]

_SAMPLE_INPUTS = {
    "healthcare": [72, 120, 36.5, 98, 1],
    "smartcity": [150, 45, 0.7, 1],
    "environment": [32, 65, 35, 400],
}

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def get_short_label(device_name):
    if device_name == "raspberry_pi_zero": return "RPi Zero"
    if device_name == "raspberry_pi_3": return "RPi 3B+"
    if device_name == "raspberry_pi_4": return "RPi 4 (4GB)"
    if device_name == "jetson_nano": return "Jetson Nano"
    return "Laptop"


def run_simulation():
    print("======================================================================")
    print("             EdgeAI Framework — Multi-Device Simulation               ")
    print("======================================================================\n")

    results = []

    devices = [
        "raspberry_pi_zero", 
        "raspberry_pi_3", 
        "raspberry_pi_4", 
        "laptop"
    ]

    for device_name in devices:
        for domain in _DOMAINS:
            # 1. Select model based on simulated device's ram_limit_mb
            profile = get_profile(device_name)
            ram = profile["ram_limit_mb"]
            
            if ram is None:
                # laptop - use actual device profiler
                model_name = select_model(get_device_profile()["device_class"])["model_name"]
            elif ram < 512:
                model_name = "logistic"
            elif ram < 2048:
                model_name = "logistic"
            elif ram < 8192:
                model_name = "random_forest"
            else:
                model_name = "neural_network"
            
            # Load the pretrained model payload
            payload = load_model(domain, model_name)
            model = payload["model"]
            scaler = payload.get("scaler")
            
            # Preprocess the input
            input_data = preprocess_input(_SAMPLE_INPUTS[domain], domain, scaler)
            
            # Run simulation
            # We run it 5 times for a more stable average
            sim_latencies = []
            mem_deltas = []
            for _ in range(5):
                res = simulate_device(device_name, model, input_data)
                sim_latencies.append(res["simulated_latency_ms"])
                mem_deltas.append(abs(res["memory_delta_mb"]))  # absolute in case it's negative
                
            avg_sim_latency = np.mean(sim_latencies)
            avg_mem = np.mean(mem_deltas)
            
            results.append({
                "Device": device_name,
                "DeviceLabel": get_short_label(device_name),
                "Domain": domain,
                "Model": model_name,
                "Latency_ms": round(avg_sim_latency, 2),
                "Memory_MB": round(avg_mem, 2)
            })

    # Print Comparison Table
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║           EdgeAI Framework — Multi-Device Simulation Report          ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Device':<20}{'Domain':<14}{'Model':<13}{'Latency':<11}{'Memory':<10}║")
    print("║  ──────────────────────────────────────────────────────────────────  ║")
    
    for r in results:
        line = f"  {r['DeviceLabel']:<20}{r['Domain']:<14}{r['Model']:<13}{r['Latency_ms']:>5.1f}ms{r['Memory_MB']:>8.1f}MB  "
        print(f"║{line:<70}║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    # Feasibility Summary
    latencies = [r["Latency_ms"] for r in results]
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    fastest = min(results, key=lambda x: x["Latency_ms"])
    
    models_used = set(r["Model"] for r in results)
    ubiquitous_model = "logistic" if "logistic" in models_used else list(models_used)[0]
    
    print("Feasibility Summary:")
    print(f"  - Most feasible combination: {fastest['DeviceLabel']} + {fastest['Domain']} ({fastest['Latency_ms']}ms)")
    print(f"  - Applicable across all constrained devices: {ubiquitous_model}")
    print(f"  - Latency range: {min_lat}ms to {max_lat}ms\n")

    # Save to CSV
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(_RESULTS_DIR, f"device_simulation_{stamp}.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"  ✓ Saved CSV: {csv_path}")

    # Generate Chart
    chart_path = os.path.join(_RESULTS_DIR, f"device_simulation_chart_{stamp}.png")
    
    # We want grouped bars: X axis=domains, Y axis=latency, Group=device
    devices_labels = [get_short_label(d) for d in devices]
    
    x = np.arange(len(_DOMAINS))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, dev in enumerate(devices):
        dev_lats = [r["Latency_ms"] for r in results if r["Device"] == dev]
        ax.bar(x + (i - 1.5) * width, dev_lats, width, label=get_short_label(dev))
        
    ax.set_ylabel('Simulated Latency (ms)')
    ax.set_title('EdgeAI Multi-Device Simulation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(_DOMAINS)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    
    print(f"  ✓ Saved Chart: {chart_path}\n")

if __name__ == "__main__":
    run_simulation()
