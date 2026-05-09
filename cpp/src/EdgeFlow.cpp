/**
 * EdgeFlow Framework v1.0
 * EdgeFlow.cpp — Main C++ Runtime Implementation
 *
 * Implements EdgeFlowClass methods and defines the global singleton.
 * Dispatch logic selects the correct inference runtime from
 * EDGEFLOW_MODEL_TYPE, which is set by the generated model header.
 */

#include "EdgeFlow.h"

// ----------------------------------------------------------------
// begin()
// ----------------------------------------------------------------
void EdgeFlowClass::begin() {
    Serial.println();
    Serial.println(F("====================================="));
    Serial.println(F(" EdgeFlow Framework v1.0"));
    Serial.println(F(" Edge AI -- Zero Cloud -- Zero Internet"));
    Serial.println(F("====================================="));

#if defined(EDGEFLOW_MODEL_TYPE)
    Serial.print(F(" Model type : "));

    #if   EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_LOGISTIC
        Serial.println(F("Logistic Regression"));
    #elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_DECISION_TREE
        Serial.println(F("Decision Tree"));
    #elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_NAIVE_BAYES
        Serial.println(F("Naive Bayes"));
    #else
        Serial.println(F("Unknown"));
    #endif

    #if defined(EDGEFLOW_N_FEATURES)
        Serial.print(F(" Features   : "));
        Serial.println(EDGEFLOW_N_FEATURES);
    #endif
    #if defined(EDGEFLOW_N_CLASSES)
        Serial.print(F(" Classes    : "));
        Serial.println(EDGEFLOW_N_CLASSES);
    #endif
#else
    Serial.println(F(" [WARNING] No model header included."));
    Serial.println(F("   #include your_model.h before EdgeFlow.h"));
#endif

    Serial.println(F("====================================="));
    Serial.println();
}

// ----------------------------------------------------------------
// predict()
// ----------------------------------------------------------------
EdgeFlowResult EdgeFlowClass::predict(float* input, int n_features) {

    EdgeFlowResult error_result;
    error_result.label          = -1;
    error_result.confidence     = 0.0f;
    error_result.latency_ms     = 0.0f;
    error_result.memory_used_kb = 0.0f;
    error_result.success        = false;

#if !defined(EDGEFLOW_MODEL_TYPE)
    // No model header included — return error result
    return error_result;
#else

    unsigned long t = edgeflow_start_timer();

    // ------------------------------------------------------------------
    // Dispatch to the correct runtime based on EDGEFLOW_MODEL_TYPE.
    // The compiler resolves this at compile time — zero runtime overhead.
    // ------------------------------------------------------------------

    #if EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_DECISION_TREE
    // ---- Decision Tree ----
    #if defined(EDGEFLOW_TREE) && defined(EDGEFLOW_TREE_NODES) && \
        defined(EDGEFLOW_SCALER_MEAN) && defined(EDGEFLOW_SCALER_SCALE)
        return dt_predict(
            input,
            n_features,
            EDGEFLOW_SCALER_MEAN,
            EDGEFLOW_SCALER_SCALE,
            EDGEFLOW_TREE,
            EDGEFLOW_TREE_NODES,
            EDGEFLOW_N_CLASSES,
            t
        );
    #else
        return error_result;
    #endif

    #elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_LOGISTIC
    // ---- Logistic Regression ----
    #if defined(EDGEFLOW_LR_WEIGHTS) && defined(EDGEFLOW_LR_BIAS) && \
        defined(EDGEFLOW_SCALER_MEAN) && defined(EDGEFLOW_SCALER_SCALE)
        return lr_predict(
            input,
            n_features,
            EDGEFLOW_SCALER_MEAN,
            EDGEFLOW_SCALER_SCALE,
            (const float*)EDGEFLOW_LR_WEIGHTS,
            EDGEFLOW_LR_BIAS,
            EDGEFLOW_N_CLASSES,
            t
        );
    #else
        return error_result;
    #endif

    #elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_NAIVE_BAYES
    // ---- Naive Bayes ----
    #if defined(EDGEFLOW_NB_PRIORS) && defined(EDGEFLOW_NB_MEANS) && \
        defined(EDGEFLOW_NB_VARS) && defined(EDGEFLOW_SCALER_MEAN) && \
        defined(EDGEFLOW_SCALER_SCALE)
        return nb_predict(
            input,
            n_features,
            EDGEFLOW_SCALER_MEAN,
            EDGEFLOW_SCALER_SCALE,
            EDGEFLOW_NB_PRIORS,
            (const float*)EDGEFLOW_NB_MEANS,
            (const float*)EDGEFLOW_NB_VARS,
            EDGEFLOW_N_CLASSES,
            t
        );
    #else
        return error_result;
    #endif

    #else
        // Unknown model type
        return error_result;
    #endif

#endif // EDGEFLOW_MODEL_TYPE
}

// ----------------------------------------------------------------
// printModelInfo()
// ----------------------------------------------------------------
void EdgeFlowClass::printModelInfo() {
    Serial.println(F("[EdgeFlow Model Info]"));
    Serial.println(F("-------------------------------------"));

#if defined(EDGEFLOW_MODEL_TYPE)
    Serial.print(F("Model type  : "));
    #if   EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_LOGISTIC
        Serial.println(F("logistic"));
    #elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_DECISION_TREE
        Serial.println(F("decision_tree"));
    #elif EDGEFLOW_MODEL_TYPE == EDGEFLOW_TYPE_NAIVE_BAYES
        Serial.println(F("naive_bayes"));
    #else
        Serial.print(EDGEFLOW_MODEL_TYPE);
        Serial.println(F(" (unknown)"));
    #endif
#else
    Serial.println(F("Model type  : (no model loaded)"));
#endif

#if defined(EDGEFLOW_N_FEATURES)
    Serial.print(F("N features  : "));
    Serial.println(EDGEFLOW_N_FEATURES);
#endif

#if defined(EDGEFLOW_N_CLASSES)
    Serial.print(F("N classes   : "));
    Serial.println(EDGEFLOW_N_CLASSES);
#endif

    Serial.print(F("Runtime ver : "));
    Serial.println(version());

    Serial.println(F("-------------------------------------"));
}

// ----------------------------------------------------------------
// version()
// ----------------------------------------------------------------
const char* EdgeFlowClass::version() {
    return "1.0.0";
}

// ----------------------------------------------------------------
// Global singleton instance
// ----------------------------------------------------------------
EdgeFlowClass EdgeFlow;
