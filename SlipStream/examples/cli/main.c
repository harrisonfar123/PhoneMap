/**
 * SlipStream CLI — Interactive Example
 *
 * A simple command-line interface for testing SlipStream inference.
 *
 * Usage:
 *   slipstream-cli <model.gguf> [prompt]
 */

#include "slipstream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

static ss_model_t *g_model = NULL;

// ─── Signal handler for clean shutdown ───────────────────────────────────────

static void signal_handler(int sig) {
    (void)sig;
    fprintf(stderr, "\n[Interrupted]\n");
    if (g_model) {
        ss_cancel(g_model);
    }
}

// ─── Token callback — prints each token as it's generated ────────────────────

static bool on_token(const char *text, uint32_t token_id, void *user_data) {
    (void)token_id;
    (void)user_data;
    printf("%s", text);
    fflush(stdout);
    return true;  // Continue generation
}

// ─── Progress callback ──────────────────────────────────────────────────────

static void on_progress(uint32_t current, uint32_t total,
                         const char *phase, void *user_data) {
    (void)user_data;
    fprintf(stderr, "\r  [%s] %u/%u", phase, current, total);
    if (current == total) {
        fprintf(stderr, "\n");
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "🌊 SlipStream v%s — Layer-Streaming LLM Inference\n"
            "\n"
            "Usage: %s <model.gguf> [prompt]\n"
            "\n"
            "If no prompt is given, enters interactive mode.\n"
            "\n"
            "Options set via environment variables:\n"
            "  SS_THREADS=N      Number of CPU threads (default: auto)\n"
            "  SS_CONTEXT=N      Context window size (default: model default)\n"
            "  SS_TEMP=0.7       Temperature (default: 0.7)\n"
            "  SS_MAX_TOKENS=N   Max tokens to generate (default: 512)\n"
            "  SS_BACKEND=cpu    Force backend: auto, cpu, metal\n"
            "\n",
            ss_version(), argv[0]
        );
        return 1;
    }

    // Set up signal handler
    signal(SIGINT, signal_handler);

    const char *model_path = argv[1];

    // ── Parse config from environment ──
    ss_model_config_t config = ss_default_config();

    const char *env_threads = getenv("SS_THREADS");
    if (env_threads) config.n_threads = (uint32_t)atoi(env_threads);

    const char *env_ctx = getenv("SS_CONTEXT");
    if (env_ctx) config.context_size = (uint32_t)atoi(env_ctx);

    const char *env_backend = getenv("SS_BACKEND");
    if (env_backend) {
        if (strcmp(env_backend, "cpu") == 0) config.backend = SS_BACKEND_CPU;
        else if (strcmp(env_backend, "metal") == 0) config.backend = SS_BACKEND_METAL;
    }

    // ── Estimate memory ──
    uint64_t peak_mem = 0;
    if (ss_estimate_memory(model_path, config.context_size, &peak_mem) == SS_OK) {
        fprintf(stderr, "Estimated peak memory: %.1f MB\n",
                (double)peak_mem / (1024.0 * 1024.0));
    }

    // ── Load model ──
    fprintf(stderr, "Loading model: %s\n", model_path);
    g_model = ss_model_load(model_path, &config);
    if (!g_model) {
        fprintf(stderr, "Error: Failed to load model — %s\n",
                ss_error_string(ss_last_error()));
        return 1;
    }

    // Set progress callback
    ss_set_progress_callback(g_model, on_progress, NULL);

    // ── Parse generation params ──
    ss_generate_params_t params = ss_default_params();

    const char *env_temp = getenv("SS_TEMP");
    if (env_temp) params.temperature = (float)atof(env_temp);

    const char *env_max = getenv("SS_MAX_TOKENS");
    if (env_max) params.max_tokens = (uint32_t)atoi(env_max);

    // ── Generate ──
    if (argc >= 3) {
        // Single prompt mode
        const char *prompt = argv[2];
        fprintf(stderr, "\n--- Generating ---\n\n");
        printf("%s", prompt);

        ss_error_t err = ss_generate(g_model, prompt, &params, on_token, NULL);
        printf("\n");

        if (err != SS_OK && err != SS_ERROR_CANCELLED) {
            fprintf(stderr, "Error: %s\n", ss_error_string(err));
        }
    } else {
        // Interactive mode
        fprintf(stderr, "\nInteractive mode. Type your prompt and press Enter.\n");
        fprintf(stderr, "Type 'quit' or Ctrl+C to exit.\n\n");

        char prompt[4096];
        while (true) {
            fprintf(stderr, ">>> ");
            if (!fgets(prompt, sizeof(prompt), stdin)) break;

            // Remove trailing newline
            size_t len = strlen(prompt);
            if (len > 0 && prompt[len - 1] == '\n') prompt[len - 1] = '\0';

            if (strcmp(prompt, "quit") == 0 || strcmp(prompt, "exit") == 0) break;
            if (strlen(prompt) == 0) continue;

            printf("\n");
            ss_error_t err = ss_generate(g_model, prompt, &params, on_token, NULL);
            printf("\n\n");

            if (err != SS_OK && err != SS_ERROR_CANCELLED) {
                fprintf(stderr, "Error: %s\n", ss_error_string(err));
            }
        }
    }

    // ── Cleanup ──
    ss_model_free(g_model);
    g_model = NULL;

    fprintf(stderr, "Done.\n");
    return 0;
}
