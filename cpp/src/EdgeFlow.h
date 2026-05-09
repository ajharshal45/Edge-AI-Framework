/**
 * EdgeFlow Framework v1.0
 * EdgeFlow.h — Main C++ Runtime API
 *
 * Single-header entry point for the EdgeFlow inference library.
 * Include this file in your Arduino sketch along with the generated
 * model header (e.g. safenest_model.h).
 *
 * Usage:
 *   #include <EdgeFlow.h>
 *   #include "safenest_model.h"
 *
 *   void setup() {
 *       Serial.begin(115200);
 *       EdgeFlow.begin();
 *       EdgeFlow.printModelInfo();
 *   }
 *
 *   void loop() {
 *       float input[] = {2800.0f, 1.0f};
 *       EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);
 *       Serial.print("Label: ");      Serial.println(result.label);
 *       Serial.print("Confidence: "); Serial.println(result.confidence);
 *       Serial.print("Latency ms: "); Serial.println(result.latency_ms);
 *       delay(1000);
 *   }
 *
 * Model type constants (set by generated .h file via #define EDGEFLOW_MODEL_TYPE):
 *   0 = logistic
 *   1 = decision_tree
 *   2 = naive_bayes
 *
 * Section 7.2 of the EdgeFlow Blueprint.
 */

#ifndef EDGEFLOW_H
#define EDGEFLOW_H

#include <Arduino.h>

// Result struct
#include "EdgeFlowResult.h"

// Timing and memory utilities
#include "utils/TimingUtils.h"
#include "utils/MemoryUtils.h"

// Model inference runtimes
#include "models/DecisionTreeRuntime.h"
#include "models/LogisticRuntime.h"
#include "models/NaiveBayesRuntime.h"

// Model type constants — must match Python exporter MODEL_TYPE_CODES
#define EDGEFLOW_TYPE_LOGISTIC       0
#define EDGEFLOW_TYPE_DECISION_TREE  1
#define EDGEFLOW_TYPE_NAIVE_BAYES    2


/**
 * EdgeFlowClass
 *
 * Singleton class that wraps all three inference runtimes behind a
 * single predict() call. The correct runtime is selected at compile
 * time via the EDGEFLOW_MODEL_TYPE define from the generated model
 * header — no runtime branching overhead on the critical path.
 */
class EdgeFlowClass {
public:

    /**
     * Initialize EdgeFlow and print version banner to Serial.
     * Call once in setup() after Serial.begin().
     */
    void begin();

    /**
     * Run inference on a raw sensor input array.
     *
     * Automatically selects the correct runtime based on
     * EDGEFLOW_MODEL_TYPE (defined in the generated model .h file).
     * Scales input using EDGEFLOW_SCALER_MEAN and EDGEFLOW_SCALER_SCALE
     * before forwarding to the model runtime.
     *
     * @param input      Pointer to float array of raw sensor readings.
     *                   Length must equal n_features.
     * @param n_features Number of features (use EDGEFLOW_N_FEATURES).
     *
     * @return EdgeFlowResult {label, confidence, latency_ms,
     *                         memory_used_kb, success}
     */
    EdgeFlowResult predict(float* input, int n_features);

    /**
     * Print model metadata to Serial.
     * Outputs domain, model type, n_features, n_classes, accuracy.
     */
    void printModelInfo();

    /**
     * Return the EdgeFlow framework version string.
     *
     * @return "1.0.0"
     */
    const char* version();
};

// Global singleton — use EdgeFlow.predict(...) directly in sketches
extern EdgeFlowClass EdgeFlow;

#endif // EDGEFLOW_H
