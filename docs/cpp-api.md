# EdgeFlow C++ API Reference

Complete reference for the EdgeFlow Arduino/ESP32 C++ runtime.

---

## Installation

Copy the `cpp/` folder into your Arduino libraries directory:

```
Documents/Arduino/libraries/EdgeFlow/
```

Or place the source directly in your sketch folder.

---

## Include Order

**Critical:** always include your model header BEFORE `EdgeFlow.h`.  
The model header defines `EDGEFLOW_MODEL_TYPE` which selects the correct inference engine at compile time.

```cpp
#include "safenest_model.h"   // generated model header — MUST be first
#include <EdgeFlow.h>          // EdgeFlow runtime
```

---

## EdgeFlowResult Struct

Returned by every `EdgeFlow.predict()` call.

```cpp
struct EdgeFlowResult {
    int   label;             // predicted class index (0, 1, 2 ...)
    float confidence;        // confidence percentage (0.0 to 100.0)
    float latency_ms;        // inference time in milliseconds
    float memory_used_kb;    // RAM used during inference in KB
    bool  success;           // true if prediction succeeded
};
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | `int` | Predicted class integer. Check your label map to interpret. For SafeNest: `0=DANGER`, `1=SAFE` |
| `confidence` | `float` | Confidence as a percentage (0–100). For Decision Tree: uniform = `100/n_classes`. For LR/NB: probability-based. |
| `latency_ms` | `float` | Wall-clock inference time in milliseconds, measured with `micros()` |
| `memory_used_kb` | `float` | Heap consumed during inference in KB. Measured with `ESP.getFreeHeap()`. Near zero for Decision Tree. |
| `success` | `bool` | `true` if prediction succeeded. `false` if no model header was included or an error occurred. |

---

## EdgeFlowClass

Global singleton accessible as `EdgeFlow` (lowercase usage: `EdgeFlow.predict(...)`).

### `begin()`

Initialize EdgeFlow and print the version banner to Serial.  
Call once in `setup()` after `Serial.begin()`.

```cpp
void setup() {
    Serial.begin(115200);
    EdgeFlow.begin();
}
```

**Output:**
```
=====================================
 EdgeFlow Framework v1.0
 Edge AI -- Zero Cloud -- Zero Internet
=====================================
 Model type : decision_tree
 Features   : 2
 Classes    : 2
=====================================
```

---

### `predict(input, n_features)` → `EdgeFlowResult`

Run ML inference on an array of raw sensor readings.

```cpp
float input[] = { 2800.0f, 1.0f };   // raw sensor values
EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `float*` | Pointer to array of raw (unscaled) sensor readings. Length must equal `n_features`. |
| `n_features` | `int` | Number of features. Use the constant `EDGEFLOW_N_FEATURES` from your model header. |

**Returns:** `EdgeFlowResult` — see struct definition above.

**What happens internally:**
1. StandardScaler is applied: `scaled[i] = (input[i] - mean[i]) / scale[i]`
2. The correct inference engine is called based on `EDGEFLOW_MODEL_TYPE`
3. Latency and memory are measured and stored in the result
4. Result is returned

**Error handling:** If no model header is included, returns `EdgeFlowResult` with `success=false` and `label=-1`. Always check `result.success` in safety-critical applications.

---

### `printModelInfo()`

Print model metadata to Serial Monitor. Useful for debugging.

```cpp
EdgeFlow.printModelInfo();
```

**Output:**
```
[EdgeFlow Model Info]
-------------------------------------
Model type  : decision_tree
N features  : 2
N classes   : 2
Runtime ver : 1.0.0
-------------------------------------
```

---

### `version()` → `const char*`

Return the EdgeFlow framework version string.

```cpp
Serial.println(EdgeFlow.version());   // "1.0.0"
```

---

## Model Header Constants

The generated `.h` file defines these constants for use in your sketch:

| Constant | Type | Description |
|----------|------|-------------|
| `EDGEFLOW_N_FEATURES` | `int` | Number of input features |
| `EDGEFLOW_N_CLASSES` | `int` | Number of output classes |
| `EDGEFLOW_MODEL_TYPE` | `int` | 0=logistic, 1=decision_tree, 2=naive_bayes |
| `EDGEFLOW_SCALER_MEAN[]` | `float[]` PROGMEM | StandardScaler mean per feature |
| `EDGEFLOW_SCALER_SCALE[]` | `float[]` PROGMEM | StandardScaler scale per feature |
| `EDGEFLOW_TREE[][5]` | `float[][5]` PROGMEM | Decision Tree node table (DT only) |
| `EDGEFLOW_LR_WEIGHTS[][]` | `float[][]` PROGMEM | LR weight matrix (LR only) |
| `EDGEFLOW_LR_BIAS[]` | `float[]` PROGMEM | LR bias vector (LR only) |
| `EDGEFLOW_NB_PRIORS[]` | `float[]` PROGMEM | NB class priors (NB only) |
| `EDGEFLOW_NB_MEANS[][]` | `float[][]` PROGMEM | NB feature means (NB only) |
| `EDGEFLOW_NB_VARS[][]` | `float[][]` PROGMEM | NB feature variances (NB only) |

---

## Timing Utilities

Available in `utils/TimingUtils.h` (auto-included via `EdgeFlow.h`):

```cpp
unsigned long t = edgeflow_start_timer();
// ... your code ...
float ms = edgeflow_elapsed_ms(t);
```

---

## Memory Utilities

Available in `utils/MemoryUtils.h` (auto-included via `EdgeFlow.h`):

```cpp
float heap_kb = edgeflow_free_heap_kb();         // current free heap in KB
float used_kb = edgeflow_memory_used_kb(before); // KB consumed since snapshot
```

On non-ESP32 platforms, `edgeflow_free_heap_kb()` returns `0.0` safely.

---

## Complete Example

```cpp
#include "safenest_model.h"
#include <EdgeFlow.h>

#define MQ2_PIN  34
#define PIR_PIN  27

void setup() {
    Serial.begin(115200);
    pinMode(MQ2_PIN, INPUT);
    pinMode(PIR_PIN, INPUT);

    EdgeFlow.begin();
    EdgeFlow.printModelInfo();
}

void loop() {
    float input[EDGEFLOW_N_FEATURES] = {
        (float)analogRead(MQ2_PIN),
        (float)digitalRead(PIR_PIN)
    };

    EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);

    if (result.success) {
        // label map: 0=DANGER, 1=SAFE
        if (result.label == 0) {
            Serial.print("DANGER | Confidence: ");
            Serial.print(result.confidence);
            Serial.print("% | Latency: ");
            Serial.print(result.latency_ms);
            Serial.println("ms");
        }
    }

    delay(500);
}
```
