/**
 * EdgeFlow Framework v1.0 — BasicPredict Example
 * ================================================
 * The simplest possible EdgeFlow sketch.
 *
 * WHAT YOU NEED:
 *   1. A generated model header (replace "your_model.h" below with the real name,
 *      e.g. "safenest_model.h" after running: edgeflow export --format header)
 *   2. Arduino IDE with ESP32 board package installed
 *   3. EdgeFlow library installed (copy cpp/ into your Arduino/libraries folder)
 *
 * STEPS:
 *   Step 1  Generate training data  : edgeflow generate-data --domain safenest --samples 500
 *   Step 2  Train model             : edgeflow train --data safenest_data.csv --target esp32
 *   Step 3  Export header           : edgeflow export --output safenest_model --format header
 *   Step 4  Copy safenest_model.h into the same folder as this .ino file
 *   Step 5  Replace "your_model.h" below with "safenest_model.h"
 *   Step 6  Upload and open Serial Monitor at 115200 baud
 *
 * SERIAL OUTPUT EXPECTED:
 *   =====================================
 *    EdgeFlow Framework v1.0
 *    Edge AI -- Zero Cloud -- Zero Internet
 *   =====================================
 *   [EdgeFlow] Label      : 0
 *   [EdgeFlow] Confidence : 50.00 %
 *   [EdgeFlow] Latency    : 0.04 ms
 *   [EdgeFlow] Memory     : 0.00 KB
 */

// ============================================================
// Step 1: Include the model header BEFORE EdgeFlow.h
//         The model header defines EDGEFLOW_MODEL_TYPE which
//         tells EdgeFlow which inference engine to compile in.
// ============================================================
#include "your_model.h"    // <-- REPLACE with your generated model filename

// Step 2: Include the main EdgeFlow runtime
#include <EdgeFlow.h>

// ============================================================
// setup() — runs once on power-on / reset
// ============================================================
void setup() {
    // Step 3: Initialize serial port
    Serial.begin(115200);
    delay(500);   // brief pause so the Serial Monitor can connect

    // Step 4: Initialize EdgeFlow — prints version banner to Serial
    EdgeFlow.begin();

    // Step 5: (Optional) print the model metadata
    EdgeFlow.printModelInfo();
}

// ============================================================
// loop() — runs forever
// ============================================================
void loop() {
    // Step 6: Build a float input array — one value per feature,
    //         in the same order as your training CSV columns
    //         (everything except the last "label" column).
    //
    //         SafeNest example: [mq2_raw, pir_state]
    //         Replace these constants with real sensor reads:
    //           float input[] = { (float)analogRead(34), (float)digitalRead(27) };
    float input[] = { 2800.0f, 1.0f };   // <-- replace with real sensor values

    // Step 7: Run inference — auto-dispatches to the correct model runtime
    EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);

    // Step 8: Use the result
    if (result.success) {
        Serial.print(F("[EdgeFlow] Label      : ")); Serial.println(result.label);
        Serial.print(F("[EdgeFlow] Confidence : ")); Serial.print(result.confidence, 2);
        Serial.println(F(" %"));
        Serial.print(F("[EdgeFlow] Latency    : ")); Serial.print(result.latency_ms, 4);
        Serial.println(F(" ms"));
        Serial.print(F("[EdgeFlow] Memory     : ")); Serial.print(result.memory_used_kb, 2);
        Serial.println(F(" KB"));
        Serial.println();
    } else {
        Serial.println(F("[EdgeFlow] ERROR: prediction failed — check your model header."));
    }

    // Wait 2 seconds before next prediction
    delay(2000);
}
