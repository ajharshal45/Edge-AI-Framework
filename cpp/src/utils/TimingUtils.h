/**
 * EdgeFlow Framework v1.0
 * utils/TimingUtils.h
 *
 * Microsecond-precision latency measurement using Arduino micros().
 *
 * Usage:
 *   unsigned long t = edgeflow_start_timer();
 *   // ... inference ...
 *   float ms = edgeflow_elapsed_ms(t);
 */

#ifndef EDGEFLOW_TIMING_UTILS_H
#define EDGEFLOW_TIMING_UTILS_H

/**
 * Record the current time in microseconds.
 * Call this immediately before the inference you want to time.
 *
 * @return  Current micros() timestamp.
 */
inline unsigned long edgeflow_start_timer() {
    return micros();
}

/**
 * Compute elapsed time in milliseconds since start.
 *
 * @param start  Timestamp returned by edgeflow_start_timer().
 * @return       Elapsed time in milliseconds (float, sub-ms precision).
 */
inline float edgeflow_elapsed_ms(unsigned long start) {
    return (float)(micros() - start) / 1000.0f;
}

#endif // EDGEFLOW_TIMING_UTILS_H
