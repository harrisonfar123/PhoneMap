/**
 * SlipStream — Performance Profiler
 *
 * Tracks timing for tokenization, prefill, per-token generation,
 * and writes a detailed report to a file.
 */

#ifndef SS_PROFILER_H
#define SS_PROFILER_H

#include <stdint.h>
#include <stdbool.h>

// Maximum tokens we track individually
#define SS_PROF_MAX_TOKENS 1024

typedef struct {
    // Model info
    char model_name[256];
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t num_heads;
    uint32_t num_kv_heads;

    // Timing (in seconds)
    double model_load_time;
    double tokenize_time;
    double prefill_time;
    double total_generate_time;

    // Token tracking
    int32_t num_prompt_tokens;
    int32_t num_generated_tokens;
    double  token_times[SS_PROF_MAX_TOKENS];  // per-token generation time
    int32_t token_ids[SS_PROF_MAX_TOKENS];

    // Memory
    uint64_t model_file_size;
    uint64_t kv_cache_size;

    // State
    bool active;
} ss_profiler_t;

/**
 * Initialize the profiler.
 */
void ss_profiler_init(ss_profiler_t *prof);

/**
 * Start a timer. Returns current time in seconds.
 */
double ss_profiler_now(void);

/**
 * Record a generated token timing.
 */
void ss_profiler_record_token(ss_profiler_t *prof, int32_t token_id, double elapsed);

/**
 * Write the performance report to a file.
 * Returns the path written to (static buffer).
 */
const char *ss_profiler_write_report(const ss_profiler_t *prof, const char *output_dir);

/**
 * Print a summary to stdout.
 */
void ss_profiler_print_summary(const ss_profiler_t *prof);

#endif // SS_PROFILER_H
