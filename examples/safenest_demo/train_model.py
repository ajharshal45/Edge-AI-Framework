"""
EdgeFlow SafeNest Demo — Complete Training Pipeline
====================================================
Runs all 5 steps of the EdgeFlow workflow for the SafeNest domain:

  Step 1: Generate 500 synthetic SafeNest sensor readings
  Step 2: Train a Decision Tree model for ESP32
  Step 3: Validate device compatibility
  Step 4: Export model as .h header and .edgeai binary
  Step 5: Print instructions for loading onto ESP32

Usage:
    cd examples/safenest_demo
    python train_model.py

Outputs:
    safenest_data.csv       — training data
    safenest_model.h        — C++ header for Arduino IDE
    safenest_model.edgeai   — binary for SPIFFS/SD runtime loading
"""

import os
import sys

# Allow running from anywhere by resolving the python package path
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_PYTHON_DIR   = os.path.join(_PROJECT_ROOT, "python")

if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOMAIN       = "safenest"
TARGET       = "esp32"
MODEL        = "decision_tree"
N_SAMPLES    = 500
DATA_PATH    = os.path.join(_SCRIPT_DIR, "safenest_data.csv")
OUTPUT_PATH  = os.path.join(_SCRIPT_DIR, "safenest_model")  # no extension

SEP = "=" * 50


def banner(title: str):
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# Step 1 — Generate SafeNest dataset
# ---------------------------------------------------------------------------
banner("Step 1 — Generate SafeNest Dataset")

from edgeflow.datasets.builtin.generate_safenest import generate_safenest_dataset

df = generate_safenest_dataset(n_samples=N_SAMPLES, output_path=DATA_PATH)

print(f"[OK] Generated {len(df)} samples -> {DATA_PATH}")
print(f"     SAFE   : {(df['label'] == 'SAFE').sum()} samples")
print(f"     DANGER : {(df['label'] == 'DANGER').sum()} samples")
print()
print("Preview (first 5 rows):")
print(df.head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Step 2 — Train Decision Tree for ESP32
# ---------------------------------------------------------------------------
banner("Step 2 — Train Decision Tree for ESP32")

from edgeflow import EdgeFlow

ef = EdgeFlow(
    domain=DOMAIN,
    target=TARGET,
    model=MODEL,
    verbose=True,
)

report = ef.train(data=DATA_PATH, test_size=0.2, random_state=42)

print()
print(f"[OK] Training complete")
print(f"     Accuracy   : {report.accuracy * 100:.2f}%")
print(f"     Samples    : {report.n_samples}")
print(f"     Features   : {report.feature_names}")
print(f"     Label map  : {report.label_map}")
print(f"     Model size : {report.model_size_kb} KB (estimated)")
print(f"     Train time : {report.training_time_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Step 3 — Validate device compatibility
# ---------------------------------------------------------------------------
banner("Step 3 — Validate Device Compatibility")

result = ef.validate_target()

if result.compatible:
    print(f"[OK] {MODEL} is compatible with {TARGET}")
    print(f"     Estimated RAM   : {result.estimated_ram_kb} KB")
    print(f"     Estimated Flash : {result.estimated_size_kb} KB")
else:
    print(f"[WARNING] Compatibility issues detected:")
    for w in result.warnings:
        print(f"   * {w}")
    print(f"   Recommendation: {result.recommendation}")


# ---------------------------------------------------------------------------
# Step 4 — Export header (.h) and binary (.edgeai)
# ---------------------------------------------------------------------------
banner("Step 4 — Export Model Files")

paths = ef.export(output=OUTPUT_PATH, format="both")

print(f"[OK] Exported {len(paths)} file(s):")
for p in paths:
    size_bytes = os.path.getsize(p)
    print(f"     {p}  ({size_bytes} bytes)")


# ---------------------------------------------------------------------------
# Step 5 — Next steps
# ---------------------------------------------------------------------------
banner("Step 5 — Next Steps: Load onto ESP32")

h_file    = next((p for p in paths if p.endswith(".h")),      "safenest_model.h")
edgeai    = next((p for p in paths if p.endswith(".edgeai")), "safenest_model.edgeai")
h_name    = os.path.basename(h_file)
ino_path  = os.path.join(
    _PROJECT_ROOT, "cpp", "examples", "SafeNestIntegration", "SafeNestIntegration.ino"
)

print(f"""
Option A — Header file (recommended, simplest):
  1. Copy {h_name} into your Arduino sketch folder:
       {os.path.join(_PROJECT_ROOT, 'cpp', 'examples', 'SafeNestIntegration')}

  2. Open SafeNestIntegration.ino in Arduino IDE:
       {ino_path}

  3. Ensure the first include reads:
       #include "{h_name}"
       #include <EdgeFlow.h>

  4. Select Board: ESP32 Dev Module
     Select Port:  your COM port

  5. Upload and open Serial Monitor at 115200 baud

  6. Expected output:
       [EdgeFlow] SAFE     GAS: 890   PIR1: no   PIR2: no
       [EdgeFlow] DANGER   GAS: 2800  PIR1: YES  PIR2: no   Confidence: 50.0%  Latency: 0.04ms

Option B — Binary file (SPIFFS/SD runtime loading):
  Copy {os.path.basename(edgeai)} to ESP32 SPIFFS filesystem.
  Refer to EdgeFlow SPIFFS loader documentation.

Data transferred to cloud : 0 bytes
Internet required         : None
""")

print(SEP)
print("  SafeNest + EdgeFlow pipeline complete!")
print(SEP)
print()
