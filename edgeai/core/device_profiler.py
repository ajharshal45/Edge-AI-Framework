"""
EdgeAI Device Profiler
======================
Detects the current device's hardware capabilities and classifies
it into an edge device category. Works on any platform — laptop,
Raspberry Pi, or any Python-capable device.

Uses psutil for cross-platform hardware detection.

Device Classes:
    ultra_edge  → RAM < 256 MB
    edge        → RAM < 512 MB
    mid_edge    → RAM < 2048 MB
    high_edge   → RAM >= 2048 MB
"""

import platform as _platform

import psutil


# ──────────────────────────────────────────────────────────
#  Device class thresholds (in MB)
# ──────────────────────────────────────────────────────────
_ULTRA_EDGE_MAX_RAM = 256
_EDGE_MAX_RAM = 512
_MID_EDGE_MAX_RAM = 2048


# ──────────────────────────────────────────────────────────
#  Hardware detection functions
# ──────────────────────────────────────────────────────────

def get_ram_mb():
    """
    Get total available RAM in megabytes.

    Uses psutil.virtual_memory() to read the total physical
    memory installed on the device.

    Returns:
        float: Total RAM in MB. Returns 0.0 if detection fails.
    """
    try:
        mem = psutil.virtual_memory()
        return round(mem.total / (1024 * 1024), 2)
    except Exception:
        return 0.0


def get_cpu_cores():
    """
    Get the number of logical CPU cores.

    Uses psutil.cpu_count() which works across all platforms
    including constrained devices like Raspberry Pi.

    Returns:
        int: Number of logical CPU cores. Returns 1 if detection fails.
    """
    try:
        cores = psutil.cpu_count(logical=True)
        return cores if cores is not None else 1
    except Exception:
        return 1


def get_platform():
    """
    Get the operating system name.

    Returns:
        str: Platform name — 'Linux', 'Windows', 'Darwin' (macOS),
             or the raw system string if unrecognised.
             Returns 'Unknown' if detection fails.
    """
    try:
        return _platform.system() or "Unknown"
    except Exception:
        return "Unknown"


def get_cpu_frequency():
    """
    Get current CPU frequency in MHz.

    Some devices (especially constrained ARM boards) do not
    expose frequency information, so this gracefully returns
    0.0 when the value is unavailable.

    Returns:
        float: CPU frequency in MHz. Returns 0.0 if not available.
    """
    try:
        freq = psutil.cpu_freq()
        if freq is not None and freq.current > 0:
            return round(freq.current, 2)
        return 0.0
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────
#  Device classification
# ──────────────────────────────────────────────────────────

def classify_device(ram_mb):
    """
    Classify a device into an edge category based on RAM.

    Args:
        ram_mb (float): Total RAM in megabytes.

    Returns:
        str: One of 'ultra_edge', 'edge', 'mid_edge', 'high_edge'.

    Raises:
        ValueError: If ram_mb is negative.
    """
    if ram_mb < 0:
        raise ValueError(f"RAM cannot be negative: {ram_mb}")

    if ram_mb < _ULTRA_EDGE_MAX_RAM:
        return "ultra_edge"
    elif ram_mb < _EDGE_MAX_RAM:
        return "edge"
    elif ram_mb < _MID_EDGE_MAX_RAM:
        return "mid_edge"
    else:
        return "high_edge"


# ──────────────────────────────────────────────────────────
#  Full device profile
# ──────────────────────────────────────────────────────────

def get_device_profile():
    """
    Build a complete hardware profile for the current device.

    Aggregates RAM, CPU cores, CPU frequency, platform info,
    and the computed device class into a single dictionary.

    Returns:
        dict: Device profile with keys:
            - ram_mb        (float)  : Total RAM in MB
            - cpu_cores     (int)    : Number of logical CPU cores
            - cpu_freq_mhz  (float)  : CPU frequency in MHz
            - platform      (str)    : OS name
            - device_class  (str)    : Edge category
            - edge_capable  (bool)   : Always True
    """
    ram = get_ram_mb()
    cores = get_cpu_cores()
    freq = get_cpu_frequency()
    plat = get_platform()
    device_class = classify_device(ram)

    return {
        "ram_mb": ram,
        "cpu_cores": cores,
        "cpu_freq_mhz": freq,
        "platform": plat,
        "device_class": device_class,
        "edge_capable": True,
    }


# ──────────────────────────────────────────────────────────
#  Pretty-printed output
# ──────────────────────────────────────────────────────────

def print_device_profile(profile):
    """
    Print the device profile in a professional box-drawn format.

    Args:
        profile (dict): Dictionary returned by get_device_profile().

    Example output::

        ╔════════════════════════════════════════╗
        ║         Device Profile                 ║
        ╠════════════════════════════════════════╣
        ║  RAM Available    : 8192 MB            ║
        ║  CPU Cores        : 8                  ║
        ║  CPU Frequency    : 2400 MHz           ║
        ║  Platform         : Linux              ║
        ║  Device Class     : high_edge          ║
        ║  Edge Capable     : YES                ║
        ╚════════════════════════════════════════╝
    """
    w = 40  # inner width between the box borders

    edge_str = "YES" if profile.get("edge_capable", True) else "NO"

    lines = [
        f"  RAM Available    : {profile['ram_mb']:.0f} MB",
        f"  CPU Cores        : {profile['cpu_cores']}",
        f"  CPU Frequency    : {profile['cpu_freq_mhz']:.0f} MHz",
        f"  Platform         : {profile['platform']}",
        f"  Device Class     : {profile['device_class']}",
        f"  Edge Capable     : {edge_str}",
    ]

    print(f"╔{'═' * w}╗")
    print(f"║{'Device Profile':^{w}}║")
    print(f"╠{'═' * w}╣")
    for line in lines:
        print(f"║{line:<{w}}║")
    print(f"╚{'═' * w}╝")


# ──────────────────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    profile = get_device_profile()
    print_device_profile(profile)
