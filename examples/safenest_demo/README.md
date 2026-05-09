# SafeNest + EdgeFlow — Complete Integration Guide

> Train a gas-hazard ML model on your laptop. Deploy to ESP32. Run forever offline.

---

## Overview

This demo shows how to replace the original SafeNest hardcoded gas rule:

```cpp
// BEFORE — hardcoded threshold (fragile, fixed, not adaptive)
if (gasValue > 2500) {
    gasAlert = true;
}
```

with a trained Machine Learning model:

```cpp
// AFTER — EdgeFlow ML inference (learned from real sensor data)
float input[] = { (float)gasValue, (float)(motion1 || motion2) };
EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);
if (result.label == 0) {   // 0 = DANGER
    gasAlert = true;
}
```

---

## Prerequisites

### Python (on your laptop)
```bash
cd c:\edi\edgeflow\python
pip install -e .
```

### Arduino IDE (on your laptop)
- Install ESP32 board package: https://docs.espressif.com/projects/arduino-esp32/
- Copy `c:\edi\edgeflow\cpp\` into your Arduino libraries folder

---

## Step-by-Step Workflow

### Option A — Automated (run everything at once)

```bash
cd c:\edi\edgeflow\examples\safenest_demo
python train_model.py
```

This runs all 5 steps automatically and prints upload instructions.

---

### Option B — Manual (using CLI commands)

#### Step 1 — Generate synthetic SafeNest training data
```bash
cd c:\edi\edgeflow\python
edgeflow generate-data --domain safenest --samples 500 --output safenest_data.csv
```

Output: `safenest_data.csv` with columns: `mq2_raw, pir_state, label`

| mq2_raw | pir_state | label  |
|---------|-----------|--------|
| 2742    | 0         | DANGER |
| 1212    | 1         | SAFE   |
| 3512    | 0         | DANGER |

---

#### Step 2 — Train the model
```bash
edgeflow train \
  --data safenest_data.csv \
  --model decision_tree \
  --target esp32 \
  --domain safenest
```

Expected output:
```
[EdgeFlow Trainer]
=======================================
Loading data...     [OK] 500 samples loaded
Validating...       [OK] Format correct
Training...         [OK] Decision Tree fitted
Accuracy...         [OK] 96.4%
Model size...       [OK] 8.0 KB
=======================================
Training complete. Run 'edgeflow export' next.
```

---

#### Step 3 — Validate ESP32 compatibility
```bash
edgeflow validate --target esp32 --model decision_tree
```

Expected output:
```
[EdgeFlow Validator]
=======================================
Target Device   : ESP32 DevKit
Model           : decision_tree
Estimated RAM   : 3.2 KB
Estimated Flash : 8.1 KB
ESP32 DEVKIT RAM Limit : 320 KB
=======================================
[OK] Compatible -- Safe to deploy
```

---

#### Step 4 — Export to C++ header
```bash
edgeflow export --output safenest_model --format both
```

Generates:
- `safenest_model.h` — include in Arduino sketch
- `safenest_model.edgeai` — for SPIFFS/SD runtime loading

---

#### Step 5 — Load onto ESP32

1. Copy `safenest_model.h` into your Arduino sketch folder:
   ```
   c:\edi\edgeflow\cpp\examples\SafeNestIntegration\
   ```

2. Open `SafeNestIntegration.ino` in Arduino IDE

3. Verify the includes at the top:
   ```cpp
   #include "safenest_model.h"   // <-- generated model
   #include <EdgeFlow.h>          // <-- EdgeFlow runtime
   ```

4. Select **Board**: `ESP32 Dev Module`  
   Select **Port**: your COM port  
   Click **Upload**

5. Open Serial Monitor at **115200 baud**

---

## Expected Serial Monitor Output

```
=====================================
 EdgeFlow Framework v1.0
 Edge AI -- Zero Cloud -- Zero Internet
=====================================
 Model type : decision_tree
 Features   : 2
 Classes    : 2
=====================================

[EdgeFlow Model Info]
-------------------------------------
Model type  : decision_tree
N features  : 2
N classes   : 2
Runtime ver : 1.0.0
-------------------------------------

SafeNest + EdgeFlow ready.
Monitoring gas and motion sensors...

GAS: 890   PIR1: no   PIR2: no   ALERT: safe
GAS: 2800  PIR1: YES  PIR2: no   ALERT: DANGER
[EdgeFlow] DANGER detected | Confidence: 50.0% | Latency: 0.040ms | Memory: 0.00KB
GAS: 1100  PIR1: no   PIR2: no   ALERT: safe
```

---

## Label Map

The EdgeFlow Python trainer sorts labels alphabetically. For SafeNest:

| Label  | Integer | Meaning                          |
|--------|---------|----------------------------------|
| DANGER | **0**   | Hazardous gas levels detected    |
| SAFE   | **1**   | Normal air quality               |

> **Important:** Always check `result.label == 0` for DANGER, not `1`.

---

## Performance (ESP32 @ 240 MHz)

| Metric              | Value        |
|---------------------|--------------|
| Inference latency   | < 0.1 ms     |
| Model Flash usage   | ~3 KB        |
| Model RAM usage     | ~3.2 KB      |
| Internet required   | **None**     |
| Cloud dependency    | **None**     |
| Data transferred    | **0 bytes**  |

---

## File Reference

| File | Description |
|------|-------------|
| `train_model.py` | Complete automated 5-step pipeline |
| `safenest_data.csv` | Generated training data (after running step 1) |
| `safenest_model.h` | Generated C++ header (after export) |
| `safenest_model.edgeai` | Generated binary (after export) |
| `../../cpp/examples/SafeNestIntegration/SafeNestIntegration.ino` | Full Arduino sketch |

---

## Architecture

```
[Laptop]                              [ESP32]
   |                                     |
edgeflow generate-data                   |
edgeflow train            ------>   safenest_model.h
edgeflow export                    (copy once via USB)
                                         |
                                    EdgeFlow.predict()
                                    runs locally, instantly
                                    zero internet, zero cloud
```

---

*EdgeFlow Framework v1.0 — VIT Pune EDI Group E-16*
