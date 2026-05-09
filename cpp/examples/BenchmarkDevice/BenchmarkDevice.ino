/**
 * EdgeFlow Framework v1.0 — BenchmarkDevice Example
 * ===================================================
 * Runs 100 prediction cycles on the ESP32 and reports latency
 * statistics to the Serial Monitor.
 *
 * WHAT IT MEASURES:
 *   - Average inference latency (ms) — lower is better
 *   - Minimum inference latency (ms) — best-case single call
 *   - Maximum inference latency (ms) — worst-case single call
 *   - Confidence % of the last prediction
 *   - Memory used during inference (KB) — should be near zero for DT
 *
 * USAGE:
 *   1. Run:  edgeflow export --output safenest_model --format header
 *   2. Copy  safenest_model.h into the same folder as this .ino
 *   3. Upload and open Serial Monitor at 115200 baud
 *   4. Results print once after 100 runs, then stop
 *
 * EXPECTED OUTPUT (SafeNest, Decision Tree):
 *   ===================================
 *   EdgeFlow Device Benchmark
 *   ===================================
 *   Model type  : decision_tree
 *   N features  : 2
 *   N classes   : 2
 *   N runs      : 100
 *   -----------------------------------
 *   Avg latency : 0.0400 ms
 *   Min latency : 0.0300 ms
 *   Max latency : 0.1500 ms
 *   Last label  : 0
 *   Confidence  : 50.00 %
 *   Memory used : 0.00 KB
 *   Data sent   : 0 bytes
 *   Internet    : None
 *   ===================================
 *   Benchmark complete.
 */

#include "safenest_model.h"    // <-- replace with your generated model filename
#include <EdgeFlow.h>

// ---- Benchmark settings ----
#define N_RUNS       100      // number of prediction cycles to measure
#define SAMPLE_MQ2   2800.0f  // fixed test input — mq2_raw reading
#define SAMPLE_PIR   1.0f     // fixed test input — pir_state reading

void setup() {
    Serial.begin(115200);
    delay(500);

    EdgeFlow.begin();

    // ----------------------------------------------------------------
    // Run benchmark
    // ----------------------------------------------------------------
    float input[EDGEFLOW_N_FEATURES] = { SAMPLE_MQ2, SAMPLE_PIR };

    float total_latency = 0.0f;
    float min_latency   = 1e9f;
    float max_latency   = 0.0f;
    EdgeFlowResult last_result;

    Serial.println(F("Running benchmark..."));

    for (int i = 0; i < N_RUNS; i++) {
        EdgeFlowResult r = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);

        if (r.success) {
            total_latency += r.latency_ms;
            if (r.latency_ms < min_latency) min_latency = r.latency_ms;
            if (r.latency_ms > max_latency) max_latency = r.latency_ms;
            last_result = r;
        }

        // Brief yield to prevent watchdog reset on long runs
        if (i % 10 == 0) delay(1);
    }

    float avg_latency = total_latency / (float)N_RUNS;

    // ----------------------------------------------------------------
    // Print report
    // ----------------------------------------------------------------
    Serial.println(F("==================================="));
    Serial.println(F("EdgeFlow Device Benchmark"));
    Serial.println(F("==================================="));

#if   EDGEFLOW_MODEL_TYPE == 0
    Serial.println(F("Model type  : logistic"));
#elif EDGEFLOW_MODEL_TYPE == 1
    Serial.println(F("Model type  : decision_tree"));
#elif EDGEFLOW_MODEL_TYPE == 2
    Serial.println(F("Model type  : naive_bayes"));
#endif

    Serial.print(F("N features  : ")); Serial.println(EDGEFLOW_N_FEATURES);
    Serial.print(F("N classes   : ")); Serial.println(EDGEFLOW_N_CLASSES);
    Serial.print(F("N runs      : ")); Serial.println(N_RUNS);

    Serial.println(F("-----------------------------------"));

    Serial.print(F("Avg latency : ")); Serial.print(avg_latency, 4);   Serial.println(F(" ms"));
    Serial.print(F("Min latency : ")); Serial.print(min_latency, 4);   Serial.println(F(" ms"));
    Serial.print(F("Max latency : ")); Serial.print(max_latency, 4);   Serial.println(F(" ms"));
    Serial.print(F("Last label  : ")); Serial.println(last_result.label);
    Serial.print(F("Confidence  : ")); Serial.print(last_result.confidence, 2); Serial.println(F(" %"));
    Serial.print(F("Memory used : ")); Serial.print(last_result.memory_used_kb, 2); Serial.println(F(" KB"));
    Serial.println(F("Data sent   : 0 bytes"));
    Serial.println(F("Internet    : None"));
    Serial.println(F("==================================="));
    Serial.println(F("Benchmark complete."));
}

void loop() {
    // Benchmark runs once in setup() — nothing to do here
    delay(10000);
}
