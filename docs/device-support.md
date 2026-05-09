# EdgeFlow — Device Support

Complete reference for supported devices, hardware limits, and model compatibility.

---

## Supported Devices

| Device | RAM | Flash | CPU | FPU | Status |
|--------|-----|-------|-----|-----|--------|
| ESP32 DevKit | 320 KB SRAM | 4 MB | 240 MHz dual-core Xtensa LX6 | No | **Fully Supported** |
| Raspberry Pi 3/4 | 1 GB+ | SD card | 1.4 GHz ARM Cortex-A | Yes | **Fully Supported** |
| Generic Linux (x86_64) | Unlimited | Unlimited | Any | Yes | **Fully Supported** |

---

## Model Compatibility Matrix

| Model | ESP32 | Raspberry Pi | Linux | Notes |
|-------|-------|--------------|-------|-------|
| `decision_tree` | **Yes** | Yes | Yes | Flagship for ESP32 — fastest, lowest memory |
| `logistic` | **Yes** | Yes | Yes | Smallest memory footprint |
| `naive_bayes` | **Yes** | Yes | Yes | Good for sensor classification |
| `random_forest` | Experimental | Yes | Yes | May exceed ESP32 RAM on large forests |
| `mlp` | No | Yes | Yes | Neural net — requires too much ESP32 RAM |

---

## ESP32 Memory Estimates

Estimates for typical small IoT datasets (2–5 features, 2–4 classes):

| Model | Est. RAM | Est. Flash | Fits ESP32? |
|-------|----------|------------|-------------|
| `logistic` | 1.1 KB | 2.0 KB | Yes (320 KB limit) |
| `decision_tree` | 3.2 KB | 8.1 KB | Yes |
| `naive_bayes` | 1.8 KB | 3.5 KB | Yes |
| `random_forest` (10 trees) | ~42 KB | ~85 KB | Yes (with caution) |
| `random_forest` (100 trees) | ~420 KB | ~850 KB | **No — exceeds limit** |

> Memory grows with number of features, classes, and tree depth. Always run  
> `edgeflow validate --target esp32` before exporting.

---

## ESP32 Technical Notes

- **No FPU:** ESP32 Xtensa LX6 has no hardware floating point unit. All float arithmetic is software-emulated. Decision Trees avoid most float math (integer comparisons after scaling). Logistic Regression and Naive Bayes use `expf()` which is slower.
- **PROGMEM storage:** All model weights are stored in Flash (PROGMEM) and read via `pgm_read_float_near()`. This saves precious SRAM.
- **Latency target:** Decision Tree should complete inference in **< 0.1 ms** at 240 MHz. Logistic Regression typically under 0.5 ms.
- **Safe model RAM limit:** EdgeFlow uses 50 KB as the practical model RAM limit for ESP32, leaving the remaining ~270 KB for application stack, WiFi buffers, and OS overhead.

---

## Raspberry Pi Notes

- All model types are supported with no memory constraints.
- The Python benchmarker is the primary profiling tool — no C++ runtime needed.
- Use `target="raspberry_pi"` in the EdgeFlow constructor to get accurate compatibility reports.
- The `.edgeai` binary format can be loaded from the filesystem at runtime.

---

## Linux Notes

- All model types supported.
- No RAM or Flash limits enforced.
- Primary use: development, testing, and benchmarking before deploying to embedded targets.
- Set `target="linux"` for local development.

---

## Adding New Devices

To add a custom device profile, edit `python/edgeflow/devices/device_profiles.py`:

```python
DEVICE_PROFILES["my_device"] = {
    "name":              "My Custom Board",
    "ram_kb":            128,
    "flash_kb":          2048,
    "cpu_mhz":           120,
    "has_fpu":           False,
    "supported_models":  ["logistic", "decision_tree", "naive_bayes"],
    "experimental_models": [],
    "max_model_size_kb": 20,
    "architecture":      "arm_cortex_m4",
}
```

Then use:
```bash
edgeflow validate --target my_device --model decision_tree
```
