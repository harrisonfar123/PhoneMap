/**
 * SlipStream — Performance Profiler Implementation
 */

#include "utils/profiler.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

void ss_profiler_init(ss_profiler_t *prof) {
    memset(prof, 0, sizeof(ss_profiler_t));
    prof->active = true;
}

double ss_profiler_now(void) {
#ifdef __APPLE__
    // High-precision timer on Apple platforms
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t t = mach_absolute_time();
    return (double)(t * info.numer) / (double)(info.denom * 1000000000ULL);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

void ss_profiler_record_token(ss_profiler_t *prof, int32_t token_id, double elapsed) {
    if (!prof || !prof->active) return;
    if (prof->num_generated_tokens < SS_PROF_MAX_TOKENS) {
        int idx = prof->num_generated_tokens;
        prof->token_times[idx] = elapsed;
        prof->token_ids[idx] = token_id;
    }
    prof->num_generated_tokens++;
}

const char *ss_profiler_write_report(const ss_profiler_t *prof, const char *output_dir) {
    if (!prof) return NULL;

    static char path[512];
    
    // Generate timestamped filename
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm);
    
    snprintf(path, sizeof(path), "%s/slipstream_perf_%s.txt", output_dir, timestamp);

    FILE *f = fopen(path, "w");
    if (!f) {
        printf("[PROFILER] Failed to write report to: %s\n", path);
        return NULL;
    }

    // ── Header ──
    fprintf(f, "═══════════════════════════════════════════════════════════\n");
    fprintf(f, "  SlipStream Performance Report\n");
    fprintf(f, "═══════════════════════════════════════════════════════════\n\n");

    // ── Model Info ──
    fprintf(f, "Model: %s\n", prof->model_name);
    fprintf(f, "Layers: %u\n", prof->num_layers);
    fprintf(f, "Hidden: %u\n", prof->hidden_size);
    fprintf(f, "Heads: %u (KV: %u)\n", prof->num_heads, prof->num_kv_heads);
    fprintf(f, "Vocab: %u\n", prof->vocab_size);
    if (prof->model_file_size > 0) {
        fprintf(f, "File Size: %.1f MB\n", (double)prof->model_file_size / (1024.0 * 1024.0));
    }
    if (prof->kv_cache_size > 0) {
        fprintf(f, "KV Cache: %.1f MB\n", (double)prof->kv_cache_size / (1024.0 * 1024.0));
    }
    fprintf(f, "\n");

    // ── Timing Summary ──
    fprintf(f, "───────────────────────────────────────────────────────────\n");
    fprintf(f, "  Timing Summary\n");
    fprintf(f, "───────────────────────────────────────────────────────────\n\n");

    fprintf(f, "Model Load:    %8.3f s\n", prof->model_load_time);
    fprintf(f, "Tokenization:  %8.3f ms\n", prof->tokenize_time * 1000.0);
    fprintf(f, "Prefill:       %8.3f s  (%d tokens)\n", prof->prefill_time, prof->num_prompt_tokens);
    if (prof->num_prompt_tokens > 0) {
        fprintf(f, "  → Prompt throughput: %.1f tokens/sec\n",
                prof->num_prompt_tokens / prof->prefill_time);
    }
    fprintf(f, "Generation:    %8.3f s  (%d tokens)\n",
            prof->total_generate_time, prof->num_generated_tokens);
    if (prof->num_generated_tokens > 0 && prof->total_generate_time > 0) {
        fprintf(f, "  → Decode throughput: %.2f tokens/sec\n",
                prof->num_generated_tokens / prof->total_generate_time);
        fprintf(f, "  → Avg per token:     %.0f ms\n",
                (prof->total_generate_time / prof->num_generated_tokens) * 1000.0);
    }
    fprintf(f, "\n");
    fprintf(f, "Total (end-to-end): %.3f s\n",
            prof->model_load_time + prof->tokenize_time + prof->prefill_time + prof->total_generate_time);
    fprintf(f, "\n");

    // ── Per-Token Breakdown ──
    int32_t n_tokens = prof->num_generated_tokens;
    if (n_tokens > SS_PROF_MAX_TOKENS) n_tokens = SS_PROF_MAX_TOKENS;

    if (n_tokens > 0) {
        fprintf(f, "───────────────────────────────────────────────────────────\n");
        fprintf(f, "  Per-Token Timing (first %d tokens)\n", n_tokens);
        fprintf(f, "───────────────────────────────────────────────────────────\n\n");

        fprintf(f, "  Token#   TokenID     Time(ms)\n");
        fprintf(f, "  ──────   ────────    ────────\n");

        double min_t = 1e9, max_t = 0, sum_t = 0;
        double first_token = prof->token_times[0];

        for (int32_t i = 0; i < n_tokens; i++) {
            double ms = prof->token_times[i] * 1000.0;
            fprintf(f, "  %5d    %6d      %7.1f\n", i, prof->token_ids[i], ms);
            if (ms < min_t) min_t = ms;
            if (ms > max_t) max_t = ms;
            sum_t += ms;
        }

        fprintf(f, "\n");
        fprintf(f, "  First token (TTFT): %7.1f ms\n", first_token * 1000.0);
        fprintf(f, "  Min:                %7.1f ms\n", min_t);
        fprintf(f, "  Max:                %7.1f ms\n", max_t);
        fprintf(f, "  Avg:                %7.1f ms\n", sum_t / n_tokens);
        fprintf(f, "\n");
    }

    fprintf(f, "═══════════════════════════════════════════════════════════\n");

    fclose(f);
    printf("[PROFILER] Report written to: %s\n", path);
    return path;
}

void ss_profiler_print_summary(const ss_profiler_t *prof) {
    if (!prof) return;
    printf("\n");
    printf("┌─────────────────────────────────────┐\n");
    printf("│  SlipStream Performance Summary     │\n");
    printf("├─────────────────────────────────────┤\n");
    printf("│ Load:    %7.2f s                   │\n", prof->model_load_time);
    printf("│ Prefill: %7.2f s (%3d tok)         │\n", prof->prefill_time, prof->num_prompt_tokens);
    printf("│ Decode:  %7.2f s (%3d tok)         │\n", prof->total_generate_time, prof->num_generated_tokens);
    if (prof->num_generated_tokens > 0 && prof->total_generate_time > 0) {
        printf("│ Speed:   %7.2f tok/s              │\n",
               prof->num_generated_tokens / prof->total_generate_time);
    }
    printf("└─────────────────────────────────────┘\n\n");
}
