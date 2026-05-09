# EdgeFlow Python Toolkit

The Python package for training, validating, and exporting ML models for IoT devices.

---

## Installation

```bash
cd python
pip install -e .
```

Verify:
```bash
edgeflow --version
# EdgeFlow 1.0.0
```

---

## Quick API Example

```python
from edgeflow import EdgeFlow

# 1. Create
ef = EdgeFlow(domain="safenest", target="esp32", model="decision_tree")

# 2. Train
report = ef.train("safenest_data.csv")
print(f"Accuracy: {report.accuracy:.1%}")      # e.g. 96.4%
print(f"Label map: {report.label_map}")        # {'DANGER': 0, 'SAFE': 1}

# 3. Validate
result = ef.validate_target()
print(f"Compatible: {result.compatible}")      # True

# 4. Export
ef.export(output="safenest_model", format="both")
# --> safenest_model.h   (for Arduino IDE)
# --> safenest_model.edgeai  (for SPIFFS/SD runtime)

# 5. Benchmark
bench = ef.benchmark(data="safenest_data.csv", n_runs=100)
print(f"Avg latency: {bench.avg_latency_ms:.4f} ms")
print(f"Data sent:   {bench.data_transferred_bytes} bytes")   # always 0
```

---

## CLI Commands

```bash
# Initialize a new project
edgeflow init --domain safenest --target esp32

# Generate synthetic training data
edgeflow generate-data --domain safenest --samples 500 --output data/safenest_data.csv

# Train
edgeflow train --data data/safenest_data.csv --model decision_tree --target esp32 --domain safenest

# Validate device compatibility
edgeflow validate --target esp32 --model decision_tree

# Export
edgeflow export --output models/safenest_model --format both

# Benchmark
edgeflow benchmark --data data/safenest_data.csv --runs 100
```

---

## Supported Models

| Model | Flag | ESP32 Support |
|-------|------|--------------|
| Decision Tree | `--model decision_tree` | Yes (recommended) |
| Logistic Regression | `--model logistic` | Yes |
| Gaussian Naive Bayes | `--model naive_bayes` | Yes |
| Random Forest | `--model random_forest` | Experimental |

---

## Package Structure

```
edgeflow/
├── __init__.py          # EdgeFlow main class
├── core/
│   ├── trainer.py       # Model training engine
│   ├── exporter.py      # .h and .edgeai export
│   ├── validator.py     # Device compatibility check
│   ├── analyzer.py      # Memory/size estimation
│   └── benchmarker.py   # Latency benchmarking
├── datasets/
│   ├── loader.py        # CSV loading and validation
│   ├── formatter.py     # Label encoding, feature scaling
│   └── builtin/         # Built-in dataset generators
├── devices/
│   └── device_profiles.py  # ESP32, Raspberry Pi, Linux specs
└── cli/
    └── commands.py      # CLI entry points
```

---

## Full API Reference

See [docs/python-api.md](../docs/python-api.md)
