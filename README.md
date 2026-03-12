# EdgeAI Framework v1.0.0

**Lightweight, deployable Edge AI framework that turns any Python-capable device into an Edge AI device.**

No cloud. No internet. No data transfer. Just local intelligence.

---

## What is EdgeAI Framework?

### The Cloud Problem
- Traditional AI requires constant internet connectivity and cloud servers
- Sensitive data (health, environment, traffic) must be uploaded to remote servers
- High latency, bandwidth costs, and single point of failure

### The Edge Solution
- **EdgeAI** ships pre-trained models *inside* the package — no downloads needed
- All predictions run locally on the device — data **never** leaves the device
- Works on any Python-capable device: laptops, Raspberry Pi, IoT boards, and more

---

## How It Works

```
Device collects sensor data
        ↓
EdgeAI Framework (runs locally on device)
        ↓
Device Profiler detects RAM, CPU, platform
        ↓
Auto-selects best lightweight model for this device
        ↓
Prediction runs locally on device
        ↓
Result → User
        
Zero data sent to cloud. Ever.
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd edgeai

# Install as editable package
pip install -e .
```

---

## Quick Start — 2 Lines

```python
from edgeai import EdgeAI

ai = EdgeAI(domain="healthcare")
result = ai.predict([72, 120, 36.5, 98, 1])
print(result)
```

That's it. Works on any device. No configuration needed.

---

## Supported Domains

| Domain | Features | Use Case |
|---|---|---|
| **healthcare** | heart_rate, blood_pressure, temperature, spo2, activity_level | Patient risk monitoring |
| **smartcity** | vehicle_count, avg_speed, traffic_density, time_of_day, weather | Traffic congestion detection |
| **environment** | temperature, humidity, pm25, pm10, co2 | Air quality hazard alerts |

---

## Device Compatibility

The framework **automatically** detects device hardware and selects the optimal model:

| Device Class | RAM | Model Selected | Example Devices |
|---|---|---|---|
| Ultra Edge | < 256 MB | Logistic Regression | Arduino + MicroPython, ESP32 |
| Edge | < 512 MB | Logistic Regression | Raspberry Pi Zero, low-end IoT |
| Mid Edge | < 2 GB | Random Forest | Raspberry Pi 3/4, Jetson Nano |
| High Edge | ≥ 2 GB | Neural Network (MLP) | Laptops, desktops, Jetson Xavier |

---

## Benchmark Results

| Device Class | Domain | Avg Latency | Model Size | Data Transfer |
|---|---|---|---|---|
| high_edge | healthcare | ~0.5 ms | 89.5 KB | 0 bytes |
| high_edge | smartcity | ~0.4 ms | 89.5 KB | 0 bytes |
| high_edge | environment | ~0.4 ms | 89.5 KB | 0 bytes |
| edge | healthcare | ~1.2 ms | 1.2 KB | 0 bytes |
| edge | smartcity | ~1.0 ms | 1.2 KB | 0 bytes |

> Run `python -m edgeai.benchmark.benchmark` on your device for actual numbers.

---

## Edge AI vs Cloud AI

| Metric | Edge AI | Cloud AI |
|---|---|---|
| Avg Latency | ~0.5 ms | ~7.5 ms |
| Data Transfer | **0 bytes** | ~2 KB/request |
| Internet Required | **NO** | YES |
| Data Privacy | **100% — data stays on device** | At risk |
| Works Offline | **YES** | NO |

---

## Project Structure

```
edgeai/
├── edgeai/                        # Python package
│   ├── __init__.py                # Package init, exports EdgeAI
│   ├── EdgeAI.py                  # Main public API (2-line usage)
│   ├── core/
│   │   ├── device_profiler.py     # Auto-detect RAM, CPU, platform
│   │   ├── model_selector.py      # Pick best model for device
│   │   ├── preprocessor.py        # Validate & normalize inputs
│   │   └── predictor.py           # Load model, run inference
│   ├── models/
│   │   ├── trainer.py             # Train & save .pkl models
│   │   └── pretrained/            # Ships with pre-trained models
│   │       ├── healthcare_*.pkl
│   │       ├── smartcity_*.pkl
│   │       └── environment_*.pkl
│   └── benchmark/
│       └── benchmark.py           # Performance benchmarking
├── data/                          # Training datasets (CSV)
├── demo/
│   └── demo.py                    # Full end-to-end demo
├── research/
│   └── experiments/
│       ├── run_all.py             # Research experiment runner
│       └── results/               # Benchmark CSVs & charts
├── setup.py                       # pip install -e .
├── requirements.txt
└── README.md
```

---

## Research

This framework is developed as part of a research study on the feasibility of deploying lightweight machine learning models on resource-constrained edge devices and IoT hardware.

Key research areas:
- **Device-aware model selection** — automatic adaptation to hardware capabilities
- **Latency benchmarking** across device classes (Ultra Edge → High Edge)
- **Privacy-preserving inference** — zero data transfer architecture
- **Multi-domain feasibility** — healthcare, smart city, and environmental monitoring

Run the full research experiment suite:

```bash
python research/experiments/run_all.py
```

---

## Tech Stack

- **Python 3.8+**
- **scikit-learn** — Logistic Regression, Random Forest, MLP
- **pandas / numpy** — Data processing
- **psutil** — Cross-platform hardware detection
- **matplotlib** — Benchmark visualisation

---

## License

MIT License

---

*Built with the vision that AI should run where the data lives — on the device itself.*
