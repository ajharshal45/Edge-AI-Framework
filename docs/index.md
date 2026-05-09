# EdgeFlow Documentation

> Train on your laptop. Deploy to any IoT device. Run forever offline.

---

## What is EdgeFlow?

EdgeFlow is an open-source **Edge AI framework** that enables any IoT developer to add real Machine Learning inference to ESP32, Raspberry Pi, or any Linux device — with **zero cloud dependency**, **zero internet requirement**, and **zero complex configuration**.

### The One-Line Description

> "The missing link between scikit-learn and your ESP32."

### The Core Problem

Every IoT system today sends sensor data to the cloud for AI decisions:

```
Sensor --> Device --> Internet --> Cloud AI --> Back to Device
```

This means: latency, privacy risk, internet dependency, cloud costs, single point of failure.

EdgeFlow eliminates the cloud step entirely:

```
Sensor --> Device --> EdgeFlow Runtime --> Decision (locally, instantly)
```

---

## Architecture

EdgeFlow has two parts that work together:

| Part | Language | Purpose |
|------|----------|---------|
| Python Toolkit | Python 3.8+ | Train models on your laptop, export to C++ |
| C++ Runtime | C++ / Arduino | Run inference on-device, zero dependencies |

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Quickstart](quickstart.md) | Get running in 5 minutes — install, train, flash to ESP32 |
| [Python API](python-api.md) | Complete Python API reference |
| [C++ API](cpp-api.md) | Complete C++ runtime API reference |
| [Device Support](device-support.md) | Supported devices, RAM limits, model compatibility |
| [Architecture](architecture.md) | How EdgeFlow works internally |

---

## Quick Links

- **GitHub:** https://github.com/ajharshal45/Edge-AI-Framework
- **License:** MIT
- **Version:** 1.0.0
- **Project:** VIT Pune EDI 2025-26, Group E-16
