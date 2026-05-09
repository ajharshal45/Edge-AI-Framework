// ============================================
// EdgeFlow Framework v1.0
// Auto-generated model header
// DO NOT EDIT MANUALLY
// ============================================
// Domain      : safenest
// Model       : decision_tree
// Target      : esp32
// Features    : 2 (mq2_raw, pir_state)
// Classes     : 2 (0=DANGER, 1=SAFE)
// Accuracy    : 100.00%
// Generated   : 2026-05-09T20:11:40Z
// ============================================

#ifndef EDGEFLOW_SAFENEST_DECISION_TREE_H
#define EDGEFLOW_SAFENEST_DECISION_TREE_H

#define EDGEFLOW_N_FEATURES 2
#define EDGEFLOW_N_CLASSES  2
#define EDGEFLOW_MODEL_TYPE 1   // 0=logistic, 1=decision_tree, 2=naive_bayes

// Feature names (for reference)
// Feature 0: mq2_raw
// Feature 1: pir_state

// Label map
// 0 = DANGER
// 1 = SAFE

// StandardScaler parameters (stored in Flash)
const float EDGEFLOW_SCALER_MEAN[]  PROGMEM = {1857.010000, 0.457500};
const float EDGEFLOW_SCALER_SCALE[] PROGMEM = {982.089685, 0.498190};

// Decision Tree nodes (stored in Flash)
// Format: {feature_index, threshold, left_child, right_child, leaf_label}
// leaf_label = -1 means internal node -- use children to traverse
// For leaf nodes: feature_index = threshold = left = right = -1
const int EDGEFLOW_TREE_NODES = 3;
const float EDGEFLOW_TREE[][5] PROGMEM = {
    {0, 0.079921, 1, 2, -1},   // node 0: if feature[0] <= 0.0799 -> left(1) else right(2)
    {-1, -1.000000, -1, -1, 1},   // node 1: leaf -> class 1
    {-1, -1.000000, -1, -1, 0}    // node 2: leaf -> class 0
};

#endif // EDGEFLOW_SAFENEST_DECISION_TREE_H
