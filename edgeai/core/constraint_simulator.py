"""
EdgeAI Constraint Simulator
===========================
Simulate resource-constrained edge devices (like Raspberry Pi)
on any machine by limiting CPU threads and measuring memory usage.
"""

import os
import time

try:
    from threadpoolctl import threadpoolctl
    _HAS_THREADPOOLCTL = True
except ImportError:
    _HAS_THREADPOOLCTL = False

import psutil


# ──────────────────────────────────────────────────────────
#  Device Profiles Dict
# ──────────────────────────────────────────────────────────

_DEVICE_PROFILES = {
    "raspberry_pi_zero": {
        "label": "Raspberry Pi Zero W",
        "max_threads": 1,
        "ram_limit_mb": 512,
        "expected_latency_multiplier": 8.0,
        "min_latency_ms": 8.0,
    },
    "raspberry_pi_3": {
        "label": "Raspberry Pi 3B+",
        "max_threads": 2,
        "ram_limit_mb": 1024,
        "expected_latency_multiplier": 4.0,
        "min_latency_ms": 4.0,
    },
    "raspberry_pi_4": {
        "label": "Raspberry Pi 4 (4GB)",
        "max_threads": 4,
        "ram_limit_mb": 4096,
        "expected_latency_multiplier": 2.0,
        "min_latency_ms": 2.0,
    },
    "jetson_nano": {
        "label": "NVIDIA Jetson Nano",
        "max_threads": 4,
        "ram_limit_mb": 4096,
        "expected_latency_multiplier": 1.5,
        "min_latency_ms": 1.0,
    },
    "laptop": {
        "label": "Laptop / Desktop",
        "max_threads": None,
        "ram_limit_mb": None,
        "expected_latency_multiplier": 1.0,
        "min_latency_ms": 0.0,
    },
}


# ──────────────────────────────────────────────────────────
#  Functions
# ──────────────────────────────────────────────────────────

def get_device_profiles():
    """Returns list of available device profile names."""
    return list(_DEVICE_PROFILES.keys())


def get_profile(device_name):
    """Returns profile dict for given device name."""
    if device_name not in _DEVICE_PROFILES:
        raise ValueError(
            f"Device '{device_name}' not found. "
            f"Available: {', '.join(get_device_profiles())}"
        )
    return _DEVICE_PROFILES[device_name]


def apply_thread_constraint(max_threads):
    """
    Limit sklearn/numpy to max_threads CPU threads.
    If max_threads is None, no constraint is applied (or default threads are kept).
    """
    if max_threads is None:
        return {"threads_set": None, "method": "none"}
        
    method = "env_vars"
    
    # Try to use threadpoolctl if available
    # NOTE: threadpoolctl act as a context manager, so applying it globally
    # via a function call is tricky without returning the context manager.
    # We will just set env vars as it's safe and effective globally,
    # but the prompt asked for threadpoolctl. We'll return it so the caller
    # can optionally use it if we want, or we can just apply env vars.
    # The prompt says: "Limit sklearn/numpy to max_threads CPU threads using: threadpoolctl library if available OR os.environ as fallback"
    
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
    
    if _HAS_THREADPOOLCTL:
        method = "threadpoolctl"
    
    return {"threads_set": max_threads, "method": method}


def measure_memory_usage_mb():
    """Measure current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_constrained_inference(model, input_data, max_threads=1):
    """
    Apply thread constraint, measure memory, run inference, capture latency,
    restore memory.
    """
    # 1. Apply constraints
    constraint_info = apply_thread_constraint(max_threads)
    
    if _HAS_THREADPOOLCTL and max_threads is not None:
        context = threadpoolctl(limits=max_threads)
        context.__enter__()
    else:
        context = None

    # 2. Measure before
    mem_before = measure_memory_usage_mb()
    
    # 3. Predict & Latency
    start_time = time.perf_counter()
    prediction = model.predict(input_data)
    latency_ms = (time.perf_counter() - start_time) * 1000

    # 4. Measure after
    mem_after = measure_memory_usage_mb()
    
    # 5. Restore constraints (exit context if threadpoolctl was used)
    if context is not None:
        context.__exit__(None, None, None)
        
    # We don't restore env vars easily, but threadpoolctl restores automatically.

    return {
        "prediction": prediction[0] if hasattr(prediction, "__len__") else prediction,
        "latency_ms": round(latency_ms, 3),
        "memory_before_mb": round(mem_before, 2),
        "memory_after_mb": round(mem_after, 2),
        "memory_delta_mb": round(mem_after - mem_before, 3),
        "threads_used": constraint_info["threads_set"] if constraint_info["threads_set"] else psutil.cpu_count(logical=True),
        "data_transferred_bytes": 0,
    }


def simulate_device(device_name, model, input_data):
    """
    Get profile, run constrained inference, apply latency multiplier.
    """
    profile = get_profile(device_name)
    
    raw_result = run_constrained_inference(
        model, 
        input_data, 
        max_threads=profile["max_threads"]
    )
    
    # Apply multiplier to real latency
    simulated = raw_result["latency_ms"] * profile["expected_latency_multiplier"]
    
    # Apply minimum floor - real devices cannot go below this
    min_floor = profile.get("min_latency_ms", 0.0)
    simulated_latency = max(simulated, min_floor)
    
    return {
        "device_name": device_name,
        "device_label": profile["label"],
        "prediction": raw_result["prediction"],
        "real_latency_ms": raw_result["latency_ms"],
        "simulated_latency_ms": round(simulated_latency, 3),
        "memory_delta_mb": raw_result["memory_delta_mb"],
        "threads_used": raw_result["threads_used"],
        "ram_limit_mb": profile["ram_limit_mb"],
        "data_transferred_bytes": 0,
    }


def print_simulation_result(result):
    """Print in box format."""
    w = 40
    print(f"╔{'═' * w}╗")
    print(f"║{'Device Simulation Result':^{w}}║")
    print(f"╠{'═' * w}╣")
    
    ram_limit_str = f"{result['ram_limit_mb']} MB" if result['ram_limit_mb'] else "No Limit"
    
    lines = [
        f"  Device        : {result['device_label']}",
        f"  Threads Used  : {result['threads_used']}",
        f"  RAM Limit     : {ram_limit_str}",
        f"  Prediction    : {result['prediction']}",
        f"  Real Latency  : {result['real_latency_ms']} ms",
        f"  Sim Latency   : {result['simulated_latency_ms']} ms",
        f"  Memory Delta  : {result['memory_delta_mb']} MB",
        f"  Data Sent     : {result['data_transferred_bytes']} bytes"
    ]
    
    for line in lines:
        print(f"║{line:<{w}}║")
        
    print(f"╚{'═' * w}╝")
