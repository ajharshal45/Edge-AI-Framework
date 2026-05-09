# EdgeFlow Framework — Complete Architecture Blueprint
## Version 1.0 | For AI Agent Implementation

---

## 1. PROJECT OVERVIEW

### What is EdgeFlow?
EdgeFlow is an open-source framework that enables any IoT developer to add real Machine Learning inference to any IoT device — ESP32, Raspberry Pi, or any Linux system — with zero cloud dependency, zero internet requirement, and zero complex configuration.

### The One-Line Description
> "Train on your laptop. Deploy to any IoT device. Run forever offline."

### The Core Problem Solved
Every IoT system today sends sensor data to the cloud for AI decisions:
```
Sensor → Device → Internet → Cloud AI → Back to Device
```
This means: latency, privacy risk, internet dependency, cloud costs, single point of failure.

EdgeFlow eliminates the cloud step:
```
Sensor → Device → EdgeFlow Runtime → Decision (locally, instantly)
```

### Real Demo (SafeNest Integration)
The framework is demonstrated live using SafeNest — an ESP32-based smart home system with:
- MQ-2 gas sensor (ADC 0–4095)
- PIR motion sensor (0/1)

**Before EdgeFlow (current SafeNest):**
```cpp
if (gasValue > 2500) {  // hardcoded rule
    gasAlert = true;
}
```

**After EdgeFlow:**
```cpp
float input[] = {gasValue, pirState};
EdgeFlowResult result = EdgeFlow.predict(input);
if (result.label == DANGER) {
    gasAlert = true;
    // ML decision using learned patterns, not hardcoded rules
}
```

---

## 2. REPOSITORY STRUCTURE

**GitHub:** https://github.com/ajharshal45/Edge-AI-Framework

**Rename repo to:** `edgeflow` (or keep URL, change display name)

```
edgeflow/
│
├── README.md                          ← Main project readme
├── LICENSE                            ← MIT License
├── .gitignore
│
├── python/                            ← Python Toolkit (pip installable)
│   ├── README.md
│   ├── setup.py
│   ├── requirements.txt
│   ├── edgeflow/                      ← Python package
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py             ← Model training engine
│   │   │   ├── exporter.py            ← Export to .h and .edgeai
│   │   │   ├── validator.py           ← Device compatibility validator
│   │   │   ├── analyzer.py            ← Memory/size estimator
│   │   │   └── benchmarker.py         ← Performance benchmarking
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── logistic.py            ← Logistic Regression wrapper
│   │   │   ├── decision_tree.py       ← Decision Tree wrapper
│   │   │   ├── naive_bayes.py         ← Naive Bayes wrapper
│   │   │   └── random_forest.py       ← Small Random Forest (experimental)
│   │   ├── devices/
│   │   │   ├── __init__.py
│   │   │   ├── device_profiles.py     ← RAM/Flash limits per device
│   │   │   └── esp32.py               ← ESP32 specific config
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py              ← Load and validate CSV datasets
│   │   │   ├── formatter.py           ← Format raw sensor data
│   │   │   └── builtin/               ← Built-in example datasets
│   │   │       ├── safenest_sample.csv
│   │   │       ├── environment_sample.csv
│   │   │       └── healthcare_sample.csv
│   │   └── cli/
│   │       ├── __init__.py
│   │       └── commands.py            ← CLI entry points
│   └── tests/
│       ├── test_trainer.py
│       ├── test_exporter.py
│       └── test_validator.py
│
├── cpp/                               ← C++ Runtime (Arduino library)
│   ├── README.md
│   ├── library.properties             ← Arduino library metadata
│   ├── src/
│   │   ├── EdgeFlow.h                 ← Main header (developer includes this)
│   │   ├── EdgeFlow.cpp               ← Runtime implementation
│   │   ├── EdgeFlowResult.h           ← Result struct definition
│   │   ├── models/
│   │   │   ├── LogisticRuntime.h      ← LR inference in C++
│   │   │   ├── DecisionTreeRuntime.h  ← DT inference in C++
│   │   │   └── NaiveBayesRuntime.h    ← NB inference in C++
│   │   └── utils/
│   │       ├── MemoryUtils.h          ← RAM measurement on device
│   │       └── TimingUtils.h          ← Latency measurement
│   └── examples/
│       ├── BasicPredict/
│       │   └── BasicPredict.ino       ← Simplest possible example
│       ├── SafeNestIntegration/
│       │   └── SafeNestIntegration.ino ← Full SafeNest demo
│       └── BenchmarkDevice/
│           └── BenchmarkDevice.ino    ← Benchmark on any device
│
├── examples/                          ← End-to-end usage examples
│   ├── safenest_demo/
│   │   ├── README.md
│   │   ├── record_data.py             ← Record sensor data from ESP32
│   │   ├── train_model.py             ← Train SafeNest model
│   │   ├── export_model.py            ← Export for ESP32
│   │   └── safenest_model.h           ← Pre-exported ready to use
│   ├── environment_demo/
│   └── healthcare_demo/
│
├── docs/                              ← Documentation
│   ├── index.md
│   ├── quickstart.md
│   ├── python-api.md
│   ├── cpp-api.md
│   ├── device-support.md
│   └── architecture.md
│
├── research/                          ← Benchmarks and analysis
│   ├── benchmarks/
│   │   ├── esp32_results.csv
│   │   └── rpi_results.csv
│   └── analysis/
│       └── latency_comparison.py
│
└── legacy/                            ← Previous EdgeAI v0 code
    └── edgeai_v0/                     ← Current 3000-line codebase moved here
```

---

## 3. DEVICE PROFILES

These are the hardware limits the validator uses. Defined in `python/edgeflow/devices/device_profiles.py`.

```python
DEVICE_PROFILES = {
    "esp32": {
        "name": "ESP32 DevKit",
        "ram_kb": 320,           # usable SRAM
        "flash_kb": 4096,        # total flash
        "cpu_mhz": 240,
        "has_fpu": False,        # no floating point unit
        "supported_models": ["logistic", "decision_tree", "naive_bayes"],
        "experimental_models": ["random_forest_small"],
        "max_model_size_kb": 50, # safe limit for model in RAM
        "architecture": "xtensa_lx6",
    },
    "raspberry_pi": {
        "name": "Raspberry Pi 3/4",
        "ram_kb": 1024 * 1024,   # 1GB+
        "flash_kb": None,        # SD card, unlimited
        "cpu_mhz": 1400,
        "has_fpu": True,
        "supported_models": ["logistic", "decision_tree", "naive_bayes",
                              "random_forest", "mlp"],
        "experimental_models": [],
        "max_model_size_kb": None,
        "architecture": "arm_cortex_a",
    },
    "linux": {
        "name": "Generic Linux",
        "ram_kb": None,
        "flash_kb": None,
        "cpu_mhz": None,
        "has_fpu": True,
        "supported_models": ["logistic", "decision_tree", "naive_bayes",
                              "random_forest", "mlp"],
        "experimental_models": [],
        "max_model_size_kb": None,
        "architecture": "x86_64",
    },
}
```

---

## 4. DATA FORMAT

### Training CSV Format
Every dataset used by EdgeFlow must follow this format:

```csv
feature_1, feature_2, ..., feature_n, label
2800, 1, DANGER
1200, 0, SAFE
3100, 1, DANGER
890,  0, SAFE
```

Rules:
- Last column is always `label`
- Labels are strings (SAFE, DANGER, etc.) — framework converts to integers internally
- Feature names are defined by the user
- No missing values allowed
- Minimum 50 rows recommended

### SafeNest Dataset Format
```csv
mq2_raw, pir_state, label
2800, 1, DANGER
1200, 0, SAFE
3100, 1, DANGER
890,  0, SAFE
450,  1, SAFE
2900, 0, DANGER
```

### Built-in Dataset Generator
For the SafeNest demo, EdgeFlow includes a synthetic data generator that produces realistic sensor readings:

```python
from edgeflow.datasets import generate_safenest_dataset

df = generate_safenest_dataset(n_samples=500)
df.to_csv("safenest_data.csv", index=False)
```

Generator rules:
- DANGER: mq2_raw > 2200 (with noise), pir_state = random
- SAFE: mq2_raw < 1800 (with noise), pir_state = random
- Add gaussian noise to make it realistic, not perfectly separable

---

## 5. PYTHON TOOLKIT — COMPLETE API

### 5.1 Main Class: EdgeFlow

**File:** `python/edgeflow/__init__.py` and `python/edgeflow/core/trainer.py`

```python
from edgeflow import EdgeFlow

ef = EdgeFlow(
    domain="safenest",        # string — user defined name
    target="esp32",           # device target
    model="decision_tree",    # model type
    verbose=True              # print progress
)
```

**Constructor Parameters:**
- `domain` (str): Name for this use case. Used in export file naming.
- `target` (str): One of "esp32", "raspberry_pi", "linux". Default: "linux"
- `model` (str): One of "logistic", "decision_tree", "naive_bayes", "random_forest". Default: "decision_tree"
- `verbose` (bool): Print progress messages. Default: True

---

### 5.2 train()

```python
report = ef.train(
    data="safenest_data.csv",   # path to CSV
    test_size=0.2,              # train/test split
    random_state=42
)

# report contains:
# report.accuracy        → float (0.0 to 1.0)
# report.model_size_kb   → float
# report.feature_names   → list of strings
# report.label_map       → dict {"SAFE": 0, "DANGER": 1}
# report.n_samples       → int
# report.training_time_ms → float
```

**Internal steps:**
1. Load CSV using `datasets/loader.py`
2. Validate format (correct columns, no missing values, enough samples)
3. Split features (all columns except last) and labels (last column)
4. Encode string labels to integers, store mapping
5. Train/test split
6. Fit StandardScaler on X_train only
7. Train selected model
8. Evaluate accuracy on X_test
9. Store trained model, scaler, feature names, label map internally
10. Return TrainingReport object

---

### 5.3 validate_target()

```python
result = ef.validate_target()

# result contains:
# result.compatible      → bool
# result.warnings        → list of strings
# result.estimated_ram_kb → float
# result.estimated_size_kb → float
# result.recommendation  → str (suggested model if incompatible)
```

**Device Compatibility Validator Output:**
```
[EdgeFlow Validator]
═══════════════════════════════════════
Target Device   : ESP32
Model           : decision_tree
Estimated RAM   : 3.2 KB
Estimated Flash : 8.1 KB
ESP32 RAM Limit : 320 KB
═══════════════════════════════════════
✅ Compatible — Safe to deploy
```

Or if incompatible:
```
[EdgeFlow Validator]
═══════════════════════════════════════
Target Device   : ESP32
Model           : random_forest (100 trees)
Estimated RAM   : 420 KB
ESP32 RAM Limit : 320 KB
═══════════════════════════════════════
⚠️  WARNING: Model exceeds ESP32 RAM limit
Suggested alternatives:
  → decision_tree    (estimated 3.2 KB)
  → logistic         (estimated 1.1 KB)
Continue anyway? [y/n]
```

---

### 5.4 export()

```python
ef.export(
    output="safenest_model",    # filename without extension
    format="header",            # "header", "binary", or "both"
)
```

**Format "header" generates:** `safenest_model.h`
**Format "binary" generates:** `safenest_model.edgeai`
**Format "both" generates:** both files

---

### 5.5 benchmark()

```python
report = ef.benchmark(
    data="safenest_data.csv",   # test data
    n_runs=100                  # repetitions
)

# report contains:
# report.avg_latency_ms
# report.min_latency_ms
# report.max_latency_ms
# report.accuracy
# report.model_size_kb
# report.data_transferred_bytes → always 0
```

---

## 6. EXPORT PIPELINE — EXACT FORMATS

### 6.1 Header File Format (.h)

**File:** `python/edgeflow/core/exporter.py` generates this

```cpp
// ============================================
// EdgeFlow Framework v1.0
// Auto-generated model header
// DO NOT EDIT MANUALLY
// ============================================
// Domain      : safenest
// Model       : decision_tree
// Target      : esp32
// Features    : 2 (mq2_raw, pir_state)
// Classes     : 2 (SAFE=0, DANGER=1)
// Accuracy    : 96.40%
// Generated   : 2026-05-09T10:30:00Z
// ============================================

#ifndef EDGEFLOW_MODEL_H
#define EDGEFLOW_MODEL_H

#define EDGEFLOW_N_FEATURES 2
#define EDGEFLOW_N_CLASSES  2
#define EDGEFLOW_MODEL_TYPE 1   // 0=logistic, 1=decision_tree, 2=naive_bayes

// Feature names (for reference)
// Feature 0: mq2_raw
// Feature 1: pir_state

// Label map
// 0 = SAFE
// 1 = DANGER

// StandardScaler parameters (stored in Flash)
const float EDGEFLOW_SCALER_MEAN[]  PROGMEM = {1450.5, 0.5};
const float EDGEFLOW_SCALER_SCALE[] PROGMEM = {820.3, 0.5};

// Decision Tree nodes (stored in Flash)
// Format: {feature_index, threshold, left_child, right_child, leaf_value}
// leaf_value = -1 means not a leaf, use children
// For leaf nodes: left_child = right_child = -1
const int EDGEFLOW_TREE_NODES = 7;
const float EDGEFLOW_TREE[][5] PROGMEM = {
    {0, 1800.5, 1, 2, -1},   // node 0: if mq2_raw <= 1800.5 go left(1) else right(2)
    {1, 0.5,   3, 4, -1},    // node 1: if pir_state <= 0.5 go left(3) else right(4)
    {-1, -1,  -1,-1,  1},    // node 2: leaf → DANGER
    {-1, -1,  -1,-1,  0},    // node 3: leaf → SAFE
    {-1, -1,  -1,-1,  0},    // node 4: leaf → SAFE
    // ... more nodes
};

#endif // EDGEFLOW_MODEL_H
```

---

### 6.2 Binary Format (.edgeai)

**File:** `python/edgeflow/core/exporter.py`

Binary format structure (bytes):
```
[0:4]   Magic bytes: "EDGF"
[4:8]   Version: uint32 (100 = v1.0.0)
[8:12]  Model type: uint32 (0=logistic, 1=decision_tree, 2=naive_bayes)
[12:16] N features: uint32
[16:20] N classes: uint32
[20:24] N scaler params: uint32
[24:X]  Scaler means: float32 array
[X:Y]   Scaler scales: float32 array
[Y:Z]   Model weights/nodes: float32 array
[Z:]    Metadata JSON: UTF-8 string (domain, feature names, label map, accuracy, timestamp)
```

This format is loaded from SPIFFS/SD card on ESP32 at runtime.

---

## 7. C++ RUNTIME — COMPLETE API

### 7.1 EdgeFlowResult Struct

**File:** `cpp/src/EdgeFlowResult.h`

```cpp
#ifndef EDGEFLOW_RESULT_H
#define EDGEFLOW_RESULT_H

struct EdgeFlowResult {
    int   label;             // predicted class index (0, 1, 2...)
    float confidence;        // confidence percentage (0.0 to 100.0)
    float latency_ms;        // inference time in milliseconds
    float memory_used_kb;    // RAM used during inference in KB
    bool  success;           // true if prediction succeeded
};

#endif
```

---

### 7.2 Main EdgeFlow Class

**File:** `cpp/src/EdgeFlow.h`

```cpp
#ifndef EDGEFLOW_H
#define EDGEFLOW_H

#include "EdgeFlowResult.h"

class EdgeFlowClass {
public:
    // Initialize with header file model (compile-time)
    void begin();

    // Run prediction on input array
    // input: float array of sensor readings (already in correct order)
    // n_features: number of features (must match model)
    EdgeFlowResult predict(float* input, int n_features);

    // Load model from .edgeai binary file (SPIFFS or SD)
    bool loadModel(const char* filepath);

    // Get framework version
    const char* version();

    // Print model info to Serial
    void printModelInfo();

private:
    // Internal inference methods per model type
    EdgeFlowResult _predictDecisionTree(float* scaled_input);
    EdgeFlowResult _predictLogistic(float* scaled_input);
    EdgeFlowResult _predictNaiveBayes(float* scaled_input);

    // Preprocessing
    void _scaleInput(float* input, float* output, int n);

    // Memory measurement
    float _measureMemoryKB();

    int   _modelType;
    int   _nFeatures;
    int   _nClasses;
    bool  _modelLoaded;
};

extern EdgeFlowClass EdgeFlow;

#endif
```

---

### 7.3 Developer Usage on ESP32

**Simplest possible usage:**
```cpp
#include <EdgeFlow.h>
#include "safenest_model.h"   // auto-generated by Python toolkit

void setup() {
    Serial.begin(115200);
    EdgeFlow.begin();
    EdgeFlow.printModelInfo();
}

void loop() {
    float input[] = {
        (float)analogRead(MQ2_PIN),
        (float)digitalRead(PIR1_PIN)
    };

    EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);

    Serial.print("Label     : "); Serial.println(result.label);
    Serial.print("Confidence: "); Serial.println(result.confidence);
    Serial.print("Latency   : "); Serial.println(result.latency_ms);
    Serial.print("Memory    : "); Serial.println(result.memory_used_kb);

    if (result.label == 1) {  // DANGER
        // trigger gas alert
    }

    delay(500);
}
```

---

### 7.4 SafeNest Integration Example

**File:** `cpp/examples/SafeNestIntegration/SafeNestIntegration.ino`

This replaces the hardcoded `if (gasValue > 2500)` in the original SafeNest code with:

```cpp
#include <EdgeFlow.h>
#include "safenest_model.h"

// Inside gasSafetySystem() — replaces the old rule-based check
void gasSafetySystem() {
    float input[] = {
        (float)gasValue,
        (float)(motion1 || motion2)
    };

    EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);

    if (result.label == 1) {  // DANGER
        gasAlert = true;
        digitalWrite(GREEN_LED, LOW);
        digitalWrite(RED_LED, HIGH);
        Serial.print("EdgeFlow DANGER detected | Confidence: ");
        Serial.print(result.confidence);
        Serial.print("% | Latency: ");
        Serial.print(result.latency_ms);
        Serial.println("ms");
    } else {
        gasAlert = false;
        digitalWrite(GREEN_LED, HIGH);
        digitalWrite(RED_LED, LOW);
    }
}
```

---

## 8. CLI COMMANDS — COMPLETE SPECIFICATION

**File:** `python/edgeflow/cli/commands.py`

All CLI commands call the same Python core engine internally.

### edgeflow init
```bash
edgeflow init --domain safenest --target esp32
```
Creates:
- `edgeflow_config.yaml` with domain, target, model settings
- `data/` folder
- `models/` folder
- Prints quickstart instructions

---

### edgeflow train
```bash
edgeflow train \
  --data data/safenest_data.csv \
  --model decision_tree \
  --target esp32 \
  --domain safenest
```
Output:
```
[EdgeFlow Trainer]
═══════════════════════════════════════
Loading data...     ✅ 500 samples loaded
Validating...       ✅ Format correct
Training...         ✅ Decision Tree fitted
Accuracy...         ✅ 96.4%
Model size...       ✅ 8.2 KB
═══════════════════════════════════════
Training complete. Run 'edgeflow export' next.
```

---

### edgeflow validate
```bash
edgeflow validate --target esp32
```
Runs device compatibility validator. Shows RAM/Flash estimates. Warns if incompatible.

---

### edgeflow export
```bash
edgeflow export \
  --output models/safenest_model \
  --format both
```
Output:
```
[EdgeFlow Exporter]
═══════════════════════════════════════
Exporting header...  ✅ models/safenest_model.h
Exporting binary...  ✅ models/safenest_model.edgeai
═══════════════════════════════════════
Copy safenest_model.h to your Arduino project.
Include EdgeFlow.h and safenest_model.h in your sketch.
```

---

### edgeflow benchmark
```bash
edgeflow benchmark \
  --data data/safenest_data.csv \
  --runs 100
```
Output:
```
[EdgeFlow Benchmark]
═══════════════════════════════════════
Model         : decision_tree
Samples       : 100 runs × 5 samples
Avg Latency   : 0.12 ms
Min Latency   : 0.08 ms
Max Latency   : 0.31 ms
Accuracy      : 96.4%
Model Size    : 8.2 KB
Data Transfer : 0 bytes
Internet Used : None
═══════════════════════════════════════
```

---

### edgeflow generate-data
```bash
edgeflow generate-data \
  --domain safenest \
  --samples 500 \
  --output data/safenest_data.csv
```
Generates synthetic but realistic sensor data for the specified domain.

---

## 9. IMPLEMENTATION ORDER FOR AI AGENT

Tell the agent to build in this exact order. Each step is testable before moving to next.

### Phase 1 — Python Core (Week 1)
```
Step 1: Repository structure setup
Step 2: device_profiles.py — device limits dictionary
Step 3: loader.py — CSV loading and validation
Step 4: formatter.py — label encoding, feature extraction
Step 5: trainer.py — train() method for all 3 models
Step 6: validator.py — device compatibility check
Step 7: analyzer.py — memory/size estimation
Step 8: exporter.py — generate .h header file
Step 9: exporter.py — generate .edgeai binary file
Step 10: benchmarker.py — latency benchmarking
Step 11: __init__.py — EdgeFlow class combining all above
Step 12: setup.py — pip installable package
```

### Phase 2 — CLI (Week 1 end)
```
Step 13: commands.py — all CLI commands calling Phase 1 core
Step 14: Test full Python workflow end to end
```

### Phase 3 — C++ Runtime (Week 2)
```
Step 15: EdgeFlowResult.h — result struct
Step 16: TimingUtils.h — latency measurement using micros()
Step 17: MemoryUtils.h — RAM measurement using ESP.getFreeHeap()
Step 18: DecisionTreeRuntime.h — DT inference reading PROGMEM
Step 19: LogisticRuntime.h — LR inference reading PROGMEM
Step 20: NaiveBayesRuntime.h — NB inference reading PROGMEM
Step 21: EdgeFlow.h + EdgeFlow.cpp — main class
Step 22: library.properties — Arduino library metadata
```

### Phase 4 — Examples and Demo (Week 2-3)
```
Step 23: BasicPredict.ino — simplest example
Step 24: generate safenest_data.csv using generate-data command
Step 25: Train SafeNest model using CLI
Step 26: Export safenest_model.h
Step 27: SafeNestIntegration.ino — full integration with original SafeNest code
Step 28: BenchmarkDevice.ino — benchmark sketch
```

### Phase 5 — Polish (Week 3-4)
```
Step 29: Move current EdgeAI v0 code to legacy/
Step 30: Write documentation in docs/
Step 31: Write main README.md
Step 32: Publish to PyPI
Step 33: Submit as Arduino library
Step 34: Record demo video
```

---

## 10. KEY TECHNICAL DECISIONS FOR AGENT

### Decision Tree Export Strategy
Decision Trees are exported as a flat array of nodes stored in PROGMEM (Flash) on ESP32.
Each node: `{feature_index, threshold, left_child_index, right_child_index, leaf_label}`
Leaf nodes have feature_index = -1 and leaf_label = class index.
Inference: recursive traversal from node 0.

### Logistic Regression Export Strategy
Weights matrix and bias vector stored as PROGMEM float arrays.
Inference: dot product of scaled input with weights, add bias, apply sigmoid, return argmax.
For binary: single weight vector. For multiclass: one weight vector per class.

### Naive Bayes Export Strategy
Store class priors and feature likelihoods (mean and variance per feature per class) in PROGMEM.
Inference: compute log-likelihood for each class, return argmax.

### Scaling on ESP32
StandardScaler parameters (mean and scale per feature) stored in PROGMEM.
Apply scaling before inference: `scaled = (raw - mean) / scale`
Use Flash-stored values via pgm_read_float_near() on AVR or direct pointer on ESP32.

### Memory Measurement on ESP32
```cpp
float _measureMemoryKB() {
    uint32_t free_before = ESP.getFreeHeap();
    // ... inference happens ...
    uint32_t free_after = ESP.getFreeHeap();
    return (free_before - free_after) / 1024.0f;
}
```

### Latency Measurement on ESP32
```cpp
unsigned long start = micros();
// ... inference ...
float latency_ms = (micros() - start) / 1000.0f;
```

---

## 11. WHAT MAKES THIS DIFFERENT FROM TFLITE

This is what you tell reviewers and judges:

| Feature | TensorFlow Lite | EdgeFlow |
|---|---|---|
| Setup complexity | High — manual optimization | Zero — automatic |
| Model selection | Manual | Automatic based on device |
| Device validation | Manual | Built-in compatibility checker |
| Export format | .tflite (binary only) | .h header + .edgeai binary |
| Target audience | ML engineers | Any IoT developer |
| Minimum code (device) | ~50 lines setup | 3 lines |
| Custom IoT datasets | Manual pipeline | Built-in toolkit |
| Benchmark tool | Separate tool | Built into framework |
| Decision Tree support | No (neural nets only) | Yes — flagship model |

---

## 12. VIVA / REVIEW ANSWERS

**Q: Why not just use TensorFlow Lite?**
A: TFLite requires manual model optimization, quantization expertise, and only supports neural networks. EdgeFlow supports Decision Trees which are actually better for ESP32 — they require zero floating point math and fit in under 10KB of Flash.

**Q: Is this real AI or just rules?**
A: The model is trained on real sensor data using supervised machine learning. The Decision Tree learned the decision boundaries from data — we did not write those if/else conditions. They were discovered by the training algorithm.

**Q: Why Decision Tree as flagship model?**
A: Because when exported to C++, a Decision Tree becomes native device logic — pure integer comparisons, no matrix math, no floating point. It runs in under 0.5ms on ESP32, uses under 5KB RAM, and is fully explainable. That is ideal for embedded safety systems.

**Q: What is novel about EdgeFlow?**
A: The combination of automatic device-aware model selection + Device Compatibility Validator + dual export format (.h for simplicity, .edgeai for flexibility) + built-in IoT dataset toolkit — in one unified framework targeting the underserved gap between raw Arduino code and heavy ML frameworks.

---

## 13. SUCCESS METRICS FOR REVIEW

The final demo should show:

1. ✅ `pip install edgeflow` works
2. ✅ `edgeflow train --data safenest_data.csv --target esp32` works
3. ✅ `edgeflow validate --target esp32` shows compatibility report
4. ✅ `edgeflow export --format both` generates .h and .edgeai files
5. ✅ Generated .h file compiles and runs on ESP32
6. ✅ ESP32 SafeNest demo runs EdgeFlow prediction in real time
7. ✅ Serial monitor shows: label, confidence, latency_ms, memory_used_kb
8. ✅ Benchmark shows <1ms latency on ESP32
9. ✅ Data transferred = 0 bytes (verified)
10. ✅ GitHub repo is public, clean, and documented

---

*EdgeFlow Framework Blueprint v1.0*
*Prepared for AI Agent Implementation*
*Project: VIT Pune EDI 2025-26, Group E-16*
