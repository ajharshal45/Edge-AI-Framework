# EdgeFlow

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-Arduino-green)
![ESP32](https://img.shields.io/badge/ESP32-Ready-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Train on your laptop. Deploy to any IoT device. Run forever offline.**

EdgeFlow is an open-source Edge AI framework that enables any IoT developer to add real Machine Learning inference to ESP32, Raspberry Pi, or any Linux device — with **zero cloud dependency**, **zero internet requirement**, and **zero complex configuration**.

---

## The Problem

Every IoT system today sends sensor data to the cloud for AI decisions:

```
Sensor --> Device --> Internet --> Cloud AI --> Back to Device
   Latency: 200ms+        Cost: $$        Failure: if no internet
```

EdgeFlow eliminates the cloud step:

```
Sensor --> Device --> EdgeFlow Runtime --> Decision (local, instant)
   Latency: <1ms           Cost: $0        Failure: never
```

---

## Quick Start

### Python (5 lines)

```python
from edgeflow import EdgeFlow

ef = EdgeFlow(domain="safenest", target="esp32", model="decision_tree")
ef.train("safenest_data.csv")
ef.export(output="safenest_model", format="both")
```

### C++ / Arduino (5 lines + result)

```cpp
#include "safenest_model.h"
#include <EdgeFlow.h>

float input[] = { (float)analogRead(34), (float)digitalRead(27) };
EdgeFlowResult r = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);
if (r.label == 0) { /* DANGER */ }
// r.latency_ms < 0.1   r.memory_used_kb ~0   r.confidence ~50-99
```

---

## Installation

### Python Toolkit

```bash
git clone https://github.com/ajharshal45/Edge-AI-Framework.git edgeflow
cd edgeflow/python
pip install -e .
```

### C++ Runtime (Arduino)

Copy the `cpp/` folder into your Arduino libraries directory:
```
Documents/Arduino/libraries/EdgeFlow/
```

---

## Supported Devices

| Device | RAM | Supported Models | Status |
|--------|-----|-----------------|--------|
| **ESP32 DevKit** | 320 KB | decision_tree, logistic, naive_bayes | Fully Supported |
| **Raspberry Pi 3/4** | 1 GB+ | All | Fully Supported |
| **Generic Linux** | Unlimited | All | Fully Supported |

---

## Supported Models

| Model | ESP32 Latency | ESP32 RAM | Best For |
|-------|--------------|-----------|---------|
| `decision_tree` | **< 0.1 ms** | ~3 KB | Binary safety systems, gas/fire detection |
| `logistic` | < 0.5 ms | ~1 KB | Linear separable sensor data |
| `naive_bayes` | < 0.3 ms | ~2 KB | Multi-feature classification |
| `random_forest` | ~1 ms | ~42 KB | Higher accuracy, more complex patterns |

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `edgeflow init --domain safenest --target esp32` | Create project scaffold |
| `edgeflow generate-data --domain safenest --samples 500` | Generate synthetic training data |
| `edgeflow train --data data.csv --model decision_tree --target esp32` | Train a model |
| `edgeflow validate --target esp32 --model decision_tree` | Check device compatibility |
| `edgeflow export --output safenest_model --format both` | Export .h and .edgeai files |
| `edgeflow benchmark --data data.csv --runs 100` | Benchmark latency and accuracy |

---

## EdgeFlow vs TensorFlow Lite

| Feature | TensorFlow Lite | EdgeFlow |
|---------|----------------|----------|
| Setup complexity | High — manual optimization | Zero — automatic |
| Model selection | Manual | Automatic based on device |
| Device validation | Manual | Built-in compatibility checker |
| Export format | .tflite (binary only) | .h header + .edgeai binary |
| Target audience | ML engineers | Any IoT developer |
| Minimum device code | ~50 lines setup | 3 lines |
| Custom IoT datasets | Manual pipeline | Built-in toolkit |
| Benchmark tool | Separate tool | Built into framework |
| Decision Tree support | No (neural nets only) | Yes — flagship model |

---

## Demo: SafeNest Integration

SafeNest is an ESP32-based smart home system with an MQ-2 gas sensor and PIR motion sensor.

**Before EdgeFlow** — hardcoded rule:
```cpp
if (gasValue > 2500) {    // fragile, fixed threshold
    gasAlert = true;
}
```

**After EdgeFlow** — learned from data:
```cpp
float input[] = { (float)gasValue, (float)(motion1 || motion2) };
EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);
if (result.label == 0) {  // 0 = DANGER (ML decision)
    gasAlert = true;
    // Confidence: 96.4% | Latency: 0.04ms | Data sent: 0 bytes
}
```

Run the full demo:
```bash
cd examples/safenest_demo
python train_model.py
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Quickstart](docs/quickstart.md) | Get running in 5 minutes |
| [Python API](docs/python-api.md) | Complete Python API reference |
| [C++ API](docs/cpp-api.md) | Complete C++ runtime API reference |
| [Device Support](docs/device-support.md) | RAM limits, model compatibility |
| [Architecture](docs/architecture.md) | How it works internally |

---

## Repository Structure

```
edgeflow/
├── python/          # Python toolkit (pip installable)
├── cpp/             # C++ Arduino library
├── examples/        # End-to-end usage examples
│   └── safenest_demo/
├── docs/            # Documentation
├── legacy/          # EdgeAI v0 prototype
└── README.md
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

*EdgeFlow Framework v1.0*
*VIT Pune EDI 2025-26, Group E-16*
