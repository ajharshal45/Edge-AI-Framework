/**
 * EdgeFlow Framework v1.0
 * utils/MemoryUtils.h
 *
 * Free heap measurement for memory-used-during-inference reporting.
 *
 * On ESP32 : uses ESP.getFreeHeap() (accurate, native API).
 * On other : returns 0.0 as a safe fallback so the code compiles
 *            universally without platform-specific linker errors.
 *
 * Usage:
 *   float before = edgeflow_free_heap_kb();
 *   // ... inference ...
 *   float used_kb = before - edgeflow_free_heap_kb();
 */

#ifndef EDGEFLOW_MEMORY_UTILS_H
#define EDGEFLOW_MEMORY_UTILS_H

/**
 * Return current free heap in kilobytes.
 *
 * @return  Free heap in KB on ESP32, 0.0 on unsupported platforms.
 */
inline float edgeflow_free_heap_kb() {
#ifdef ESP32
    return (float)ESP.getFreeHeap() / 1024.0f;
#elif defined(ESP8266)
    return (float)ESP.getFreeHeap() / 1024.0f;
#else
    // Non-ESP platform: return 0.0 — no heap API available
    return 0.0f;
#endif
}

/**
 * Measure RAM consumed between two heap snapshots.
 *
 * @param heap_before  Snapshot taken before inference (KB).
 * @return             Kilobytes consumed = before - current free heap.
 *                     Negative values indicate heap grew (unlikely but safe).
 */
inline float edgeflow_memory_used_kb(float heap_before) {
    float heap_after = edgeflow_free_heap_kb();
    float used = heap_before - heap_after;
    return (used > 0.0f) ? used : 0.0f;
}

#endif // EDGEFLOW_MEMORY_UTILS_H
