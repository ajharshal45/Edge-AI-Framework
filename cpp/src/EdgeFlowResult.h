/**
 * EdgeFlow Framework v1.0
 * EdgeFlowResult.h
 *
 * Result struct returned by every EdgeFlow prediction call.
 * Matches Section 7.1 of the EdgeFlow Blueprint exactly.
 *
 * Usage:
 *   EdgeFlowResult result = EdgeFlow.predict(input, EDGEFLOW_N_FEATURES);
 *   Serial.println(result.label);         // predicted class index
 *   Serial.println(result.confidence);    // 0.0 to 100.0
 *   Serial.println(result.latency_ms);    // inference time in ms
 *   Serial.println(result.memory_used_kb); // RAM used during inference in KB
 */

#ifndef EDGEFLOW_RESULT_H
#define EDGEFLOW_RESULT_H

struct EdgeFlowResult {
    int   label;             // predicted class index (0, 1, 2 ...)
    float confidence;        // confidence percentage (0.0 to 100.0)
    float latency_ms;        // inference time in milliseconds
    float memory_used_kb;    // RAM used during inference in KB
    bool  success;           // true if prediction succeeded
};

#endif // EDGEFLOW_RESULT_H
