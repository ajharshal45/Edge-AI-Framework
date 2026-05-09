# EdgeFlow Python API Reference

Complete reference for the `edgeflow` Python package.

---

## EdgeFlow (main class)

**Import:** `from edgeflow import EdgeFlow`

The single entry point for the EdgeFlow framework. Wraps the full pipeline — training, validation, export, and benchmarking — behind one clean object.

### Constructor

```python
ef = EdgeFlow(
    domain="safenest",        # str  — user-defined project name
    target="esp32",           # str  — target device
    model="decision_tree",    # str  — ML model type
    verbose=True              # bool — print progress messages
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | str | `"custom"` | Name for this use case. Used in export file naming and report headers. |
| `target` | str | `"esp32"` | Target device. One of: `"esp32"`, `"raspberry_pi"`, `"linux"` |
| `model` | str | `"decision_tree"` | Model type. One of: `"logistic"`, `"decision_tree"`, `"naive_bayes"`, `"random_forest"` |
| `verbose` | bool | `True` | Print progress messages during pipeline steps |

---

### `train(data, test_size, random_state)` → `TrainingReport`

Load a CSV dataset, train the model, and store results internally.

```python
report = ef.train(
    data="safenest_data.csv",   # path to training CSV
    test_size=0.2,              # 20% held out for testing
    random_state=42             # reproducibility seed
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | str | required | Path to CSV file. Last column must be `label`. |
| `test_size` | float | `0.2` | Fraction of data reserved for testing (0.0 to 1.0) |
| `random_state` | int | `42` | Random seed for reproducibility |

**Returns:** `TrainingReport` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `accuracy` | float | Test set accuracy (0.0 to 1.0) |
| `model_type` | str | Model type string (e.g. `"decision_tree"`) |
| `model_size_kb` | float | Estimated model size in KB |
| `feature_names` | list[str] | Column names from CSV (excluding label) |
| `label_map` | dict[str, int] | e.g. `{"DANGER": 0, "SAFE": 1}` |
| `n_samples` | int | Total number of training samples |
| `n_features` | int | Number of input features |
| `n_classes` | int | Number of output classes |
| `training_time_ms` | float | Training duration in milliseconds |
| `compatible_with_target` | bool | Whether model fits on target device |
| `warnings` | list[str] | Any compatibility warnings |

**CSV format required:**
```csv
feature_1,feature_2,...,feature_n,label
2800,1,DANGER
1200,0,SAFE
```

---

### `validate_target()` → `ValidationResult`

Check whether the trained model fits on the target device.

```python
result = ef.validate_target()

print(result.compatible)          # True / False
print(result.estimated_ram_kb)    # e.g. 3.2
print(result.estimated_size_kb)   # e.g. 8.1
print(result.warnings)            # list of warning strings
print(result.recommendation)      # suggestion if incompatible
```

**Returns:** `ValidationResult` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `compatible` | bool | True if model fits within device limits |
| `estimated_ram_kb` | float | Estimated RAM usage during inference |
| `estimated_size_kb` | float | Estimated Flash/storage usage |
| `warnings` | list[str] | Compatibility issues found |
| `recommendation` | str | Alternative model suggestion if incompatible |
| `device_ram_limit_kb` | float | Device's total RAM limit |

---

### `export(output, format)` → `list[str]`

Export the trained model to C++ header and/or binary format.

```python
paths = ef.export(
    output="safenest_model",    # filename without extension
    format="both"               # "header", "binary", or "both"
)
# paths = ["safenest_model.h", "safenest_model.edgeai"]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output` | str | `"edgeflow_model"` | Output filename without extension |
| `format` | str | `"header"` | `"header"` (.h only), `"binary"` (.edgeai only), `"both"` |

**Returns:** List of file paths written.

**Raises:** `RuntimeError` if called before `train()`.

---

### `benchmark(data, n_runs)` → `BenchmarkReport`

Benchmark prediction latency and accuracy over multiple passes.

```python
bench = ef.benchmark(
    data="safenest_data.csv",   # CSV test data (same format as training)
    n_runs=100                  # number of full passes
)

print(bench.avg_latency_ms)     # e.g. 0.12
print(bench.accuracy)           # e.g. 0.964
print(bench.data_transferred_bytes)  # always 0
```

**Returns:** `BenchmarkReport` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `avg_latency_ms` | float | Mean prediction time in ms |
| `min_latency_ms` | float | Fastest prediction |
| `max_latency_ms` | float | Slowest prediction |
| `accuracy` | float | Accuracy on test data (0.0 to 1.0) |
| `model_size_kb` | float | Estimated model size |
| `n_predictions` | int | Total predictions made (n_runs x n_samples) |
| `data_transferred_bytes` | int | Always `0` — offline inference |
| `internet_used` | str | Always `"None"` |

---

## DeviceValidator (standalone)

Use without training a model first.

```python
from edgeflow.core.validator import DeviceValidator

v = DeviceValidator(verbose=True)
result = v.validate(model_type="decision_tree", target_device="esp32")
```

---

## EdgeFlowTrainer (standalone)

Access the trainer directly for custom pipelines.

```python
from edgeflow.core.trainer import EdgeFlowTrainer

trainer = EdgeFlowTrainer(model="logistic", target="raspberry_pi", domain="env")
report  = trainer.train("environment_data.csv")
model   = trainer.get_fitted_model()
fmt     = trainer.get_formatter()
```

---

## EdgeFlowExporter (standalone)

```python
from edgeflow.core.exporter import EdgeFlowExporter

exporter = EdgeFlowExporter()
paths = exporter.export(
    fitted_model=model,
    formatter=fmt,
    report=report,
    output="my_model",
    format="header"
)
```

---

## Built-in Dataset Generators

```python
from edgeflow.datasets.builtin.generate_safenest import generate_safenest_dataset

df = generate_safenest_dataset(n_samples=500, output_path="data.csv")
```

Generator rules:
- **DANGER**: `mq2_raw > 2200` (with Gaussian noise), `pir_state` random
- **SAFE**: `mq2_raw < 1800` (with Gaussian noise), `pir_state` random

---

## Complete Example

```python
from edgeflow import EdgeFlow

# 1. Create
ef = EdgeFlow(domain="safenest", target="esp32", model="decision_tree", verbose=True)

# 2. Train
report = ef.train("safenest_data.csv", test_size=0.2)
print(f"Accuracy: {report.accuracy:.1%}")

# 3. Validate
result = ef.validate_target()
assert result.compatible, "Model does not fit on ESP32!"

# 4. Export
ef.export(output="safenest_model", format="both")

# 5. Benchmark
bench = ef.benchmark(data="safenest_data.csv", n_runs=100)
print(f"Avg latency: {bench.avg_latency_ms:.4f} ms")
print(f"Data transferred: {bench.data_transferred_bytes} bytes")  # always 0
```
