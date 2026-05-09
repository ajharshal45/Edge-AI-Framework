# EdgeFlow Architecture

How EdgeFlow works internally — from raw sensor data to on-device inference.

---

## Overview

EdgeFlow has two separate runtimes that hand off through an exported model file:

```
+------------------+        +-------------------+
|  Python Toolkit  |        |   C++ Runtime     |
|  (your laptop)   |        |   (your ESP32)    |
|                  |        |                   |
|  loader          |        |  EdgeFlow.h       |
|  formatter       | -----> |  DecisionTree     |
|  trainer         | .h/.edgeai |  LogisticRuntime  |
|  validator       | file   |  NaiveBayes       |
|  exporter        |        |  TimingUtils      |
|  benchmarker     |        |  MemoryUtils      |
+------------------+        +-------------------+
```

---

## Training Pipeline (Python)

```
CSV File
   |
   v
[loader.py]
   Load CSV, validate format, extract feature names
   Reject: missing values, wrong column order, < 50 rows
   |
   v
[formatter.py]
   LabelEncoder: DANGER -> 0, SAFE -> 1  (alphabetical sort)
   StandardScaler: fit on X_train only
   Store: mean[], scale[], label_map
   |
   v
[trainer.py]
   train_test_split(stratified, 80/20)
   Fit sklearn model: DecisionTreeClassifier / LogisticRegression / GaussianNB
   Evaluate accuracy on X_test
   |
   v
[TrainingReport]
   accuracy, feature_names, label_map, model_size_kb, n_samples
   |
   v
[validator.py]                    [analyzer.py]
   Check: model in device profile    Estimate: RAM and Flash KB
   Check: estimated RAM < limit      Check: fits within device limits
   |
   v
[exporter.py]
   Build .h header string
   Build .edgeai binary bytes
   Write to disk
```

---

## Export Pipeline — Header File

The Python exporter traverses the fitted sklearn model and emits C arrays stored in PROGMEM (Flash):

### Decision Tree Export

```
sklearn DecisionTreeClassifier
   tree_.feature[]          -> feature index per node
   tree_.threshold[]        -> split threshold per node
   tree_.children_left[]    -> left child index
   tree_.children_right[]   -> right child index
   tree_.value[]            -> class distribution at each node

Exported as flat 5-column array:
   EDGEFLOW_TREE[][5] PROGMEM = {
       {feature, threshold, left, right, leaf_label},
       ...
   }
   Leaf nodes: feature=-1, threshold=-1, left=-1, right=-1
```

### Logistic Regression Export

```
sklearn LogisticRegression
   coef_[]         -> weight matrix (n_coef_rows x n_features)
   intercept_[]    -> bias vector   (n_coef_rows)
   Binary: 1 coef row
   Multiclass: n_classes coef rows

Exported as:
   EDGEFLOW_LR_WEIGHTS[][N_FEATURES] PROGMEM
   EDGEFLOW_LR_BIAS[] PROGMEM
```

### Naive Bayes Export

```
sklearn GaussianNB
   class_prior_[]  -> class probabilities
   theta_[][]      -> per-class feature means
   var_[][]        -> per-class feature variances

Exported as:
   EDGEFLOW_NB_PRIORS[] PROGMEM
   EDGEFLOW_NB_MEANS[][N_FEATURES] PROGMEM
   EDGEFLOW_NB_VARS[][N_FEATURES] PROGMEM
```

---

## C++ Runtime Inference Flow

```
EdgeFlow.predict(input[], n_features)
   |
   +-- Check EDGEFLOW_MODEL_TYPE (compile-time constant)
   |
   +-- [EDGEFLOW_TYPE_DECISION_TREE == 1]
   |       dt_predict()
   |         1. Scale: scaled[i] = (input[i] - mean[i]) / scale[i]
   |            (mean/scale read from PROGMEM via pgm_read_float_near)
   |         2. Start at node 0
   |         3. Loop:
   |              if node.feature == -1: return leaf_label
   |              if scaled[feature] <= threshold: go left
   |              else: go right
   |         4. Fill result: label, confidence=100/n_classes, latency, memory
   |
   +-- [EDGEFLOW_TYPE_LOGISTIC == 0]
   |       lr_predict()
   |         1. Scale input
   |         2. dot(weights[c], scaled) + bias[c] for each coef row
   |         3. Binary: sigmoid(score) -> label + confidence
   |            Multiclass: softmax(scores) -> argmax + confidence
   |
   +-- [EDGEFLOW_TYPE_NAIVE_BAYES == 2]
           nb_predict()
             1. Scale input
             2. log_score[c] = log(prior[c]) + sum_f log_gaussian(scaled[f], mu_cf, var_cf)
             3. label = argmax(log_scores)
             4. confidence = softmax(log_scores)[label] * 100
```

---

## Latency Measurement

```cpp
unsigned long t = micros();       // start timer (microseconds)
// ... inference logic ...
float ms = (micros() - t) / 1000.0f;   // convert to ms
```

`micros()` has 1 microsecond resolution on ESP32 at 240 MHz, giving sub-millisecond latency measurement.

---

## Memory Measurement

```cpp
float before = ESP.getFreeHeap() / 1024.0f;
// ... inference ...
float after  = ESP.getFreeHeap() / 1024.0f;
float used   = before - after;    // KB consumed
```

For Decision Tree with stack-allocated `scaled[]` array: typically near **0 KB** heap used.

---

## Binary Format (.edgeai)

For runtime loading from SPIFFS or SD card:

```
Bytes      Field
[0:4]      Magic: "EDGF"
[4:8]      Version: uint32 (100 = v1.0.0)
[8:12]     Model type: uint32 (0/1/2)
[12:16]    N features: uint32
[16:20]    N classes: uint32
[20:24]    N scaler params: uint32
[24:X]     Scaler means: float32[] (one per feature)
[X:Y]      Scaler scales: float32[] (one per feature)
[Y:Z]      Model weights: float32[] (layout depends on model type)
[Z:]       Metadata JSON: UTF-8 (domain, feature_names, label_map, accuracy, timestamp)
```

---

## Compile-Time Dispatch

The C++ runtime uses `#if EDGEFLOW_MODEL_TYPE` preprocessor conditionals rather than runtime `if/else`. This means:

- **Zero runtime branching** on the inference critical path
- **Dead code elimination**: the compiler strips out the two unused runtimes entirely
- **Smaller binary**: only the selected inference engine is compiled in

```cpp
// In EdgeFlow.cpp — resolved at compile time, not runtime
#if EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_DECISION_TREE
    return dt_predict(...);
#elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_LOGISTIC
    return lr_predict(...);
#elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_NAIVE_BAYES
    return nb_predict(...);
#endif
```

---

## Why Decision Tree is the Flagship for ESP32

| Property | Decision Tree | Logistic Regression | Neural Net (TFLite) |
|----------|--------------|---------------------|---------------------|
| Inference math | Integer comparisons | Dot product + exp() | Matrix multiply |
| Float operations | Minimal (scaling only) | Heavy | Very heavy |
| Memory (typical) | 3 KB | 1 KB | 100 KB+ |
| Latency on ESP32 | < 0.1 ms | < 0.5 ms | 10–100 ms |
| Explainability | Full — you can read the tree | Partial | None |
| Quantization needed | No | No | Yes |
