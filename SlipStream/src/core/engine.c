/**
 * SlipStream — Main Engine Implementation
 *
 * Implements the public C API defined in slipstream.h.
 * This is the entry point that ties together the GGUF loader,
 * layer scheduler, tokenizer, and compute backends.
 */

#include "core/engine.h"
#include "utils/memory.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// ─── Thread-local error ──────────────────────────────────────────────────────

static __thread ss_error_t g_last_error = SS_OK;

// ─── Default Configurations ──────────────────────────────────────────────────

ss_model_config_t ss_default_config(void) {
    return (ss_model_config_t){
        .backend      = SS_BACKEND_AUTO,
        .n_threads    = 0,      // auto
        .max_memory   = 0,      // no limit
        .use_mmap     = true,
        .prefetch     = true,
        .context_size = 0,      // model default
    };
}

ss_generate_params_t ss_default_params(void) {
    return (ss_generate_params_t){
        .max_tokens     = 512,
        .temperature    = 0.7f,
        .top_p          = 0.9f,
        .top_k          = 40,
        .repeat_penalty = 1.1f,
        .seed           = 0,
        .stop_sequence  = NULL,
    };
}

// ─── Model Loading ───────────────────────────────────────────────────────────

ss_model_t *ss_model_load(const char *path, const ss_model_config_t *config) {
    if (!path) {
        g_last_error = SS_ERROR_INVALID_PARAMS;
        return NULL;
    }

    ss_model_t *model = (ss_model_t *)calloc(1, sizeof(ss_model_t));
    if (!model) {
        g_last_error = SS_ERROR_OUT_OF_MEMORY;
        return NULL;
    }

    // Apply config
    model->config = config ? *config : ss_default_config();

    // ── Open GGUF file ──
    model->gguf = ss_gguf_open(path);
    if (!model->gguf) {
        g_last_error = SS_ERROR_INVALID_MODEL;
        free(model);
        return NULL;
    }

    // ── Initialize tokenizer ──
    model->tokenizer = ss_tokenizer_create(model->gguf->mmap_base,
                                            model->gguf->file_size);

    // ── Initialize scheduler ──
    uint32_t ctx_size = model->config.context_size > 0
                        ? model->config.context_size
                        : model->gguf->context_length;

    model->scheduler = ss_scheduler_init(model->gguf, ctx_size,
                                          model->config.prefetch);
    if (!model->scheduler) {
        g_last_error = SS_ERROR_OUT_OF_MEMORY;
        ss_model_free(model);
        return NULL;
    }

    // ── Initialize thread pool ──
    uint32_t n_threads = model->config.n_threads > 0
                         ? model->config.n_threads
                         : ss_get_cpu_count();
    model->thread_pool = ss_thread_pool_create(n_threads);

    // ── Initialize Metal backend ──
#ifdef SS_HAS_METAL
    if (model->config.backend == SS_BACKEND_AUTO ||
        model->config.backend == SS_BACKEND_METAL) {
        model->metal = ss_metal_init();
        if (model->metal) {
            fprintf(stderr, "SlipStream: Metal GPU backend initialized\n");
        }
    }
#endif

    fprintf(stderr, "SlipStream: Model loaded — %s\n", path);
    fprintf(stderr, "  Architecture: %s\n", model->gguf->arch);
    fprintf(stderr, "  Layers: %u\n", model->gguf->num_layers);
    fprintf(stderr, "  Hidden size: %u\n", model->gguf->hidden_size);
    fprintf(stderr, "  Heads: %u (KV: %u)\n", model->gguf->num_heads,
            model->gguf->num_kv_heads);
    fprintf(stderr, "  Vocab: %u\n", model->gguf->vocab_size);
    fprintf(stderr, "  Context: %u\n", ctx_size);

    g_last_error = SS_OK;
    return model;
}

// ─── Model Info ──────────────────────────────────────────────────────────────

ss_error_t ss_model_get_info(const ss_model_t *model, ss_model_info_t *info) {
    if (!model || !info) return SS_ERROR_INVALID_PARAMS;

    memset(info, 0, sizeof(ss_model_info_t));
    strncpy(info->architecture, model->gguf->arch, sizeof(info->architecture) - 1);
    info->num_layers = model->gguf->num_layers;
    info->hidden_size = model->gguf->hidden_size;
    info->num_heads = model->gguf->num_heads;
    info->num_kv_heads = model->gguf->num_kv_heads;
    info->vocab_size = model->gguf->vocab_size;
    info->context_length = model->gguf->context_length;
    info->intermediate_size = model->gguf->intermediate_size;
    info->file_size = model->gguf->file_size;

    // Compute per-layer size
    if (model->gguf->layers && model->gguf->num_layers > 0) {
        info->per_layer_size = (uint64_t)model->gguf->layers[0].total_size;
    }

    return SS_OK;
}

// ─── Text Generation ─────────────────────────────────────────────────────────

ss_error_t ss_generate(
    ss_model_t                *model,
    const char                *prompt,
    const ss_generate_params_t *params,
    ss_token_callback_t        token_cb,
    void                      *user_data
) {
    if (!model || !prompt) {
        g_last_error = SS_ERROR_INVALID_PARAMS;
        return SS_ERROR_INVALID_PARAMS;
    }

    ss_generate_params_t p = params ? *params : ss_default_params();
    model->cancelled = false;

    // Reset scheduler for new generation
    ss_scheduler_reset(model->scheduler);

    // ── Tokenize prompt ──
    int32_t prompt_tokens[4096];
    int32_t n_prompt = ss_tokenizer_encode(model->tokenizer, prompt,
                                            prompt_tokens, 4096);
    if (n_prompt <= 0) {
        g_last_error = SS_ERROR_TOKENIZER;
        return SS_ERROR_TOKENIZER;
    }

    // ── Initialize RNG ──
    uint64_t rng_state = p.seed > 0 ? (uint64_t)p.seed : (uint64_t)time(NULL);

    // ── Process prompt tokens (prefill) ──
    if (model->progress_cb) {
        model->progress_cb(0, (uint32_t)n_prompt, "prefill",
                           model->progress_user_data);
    }

    float *logits = NULL;
    for (int32_t i = 0; i < n_prompt; i++) {
        if (model->cancelled) {
            g_last_error = SS_ERROR_CANCELLED;
            return SS_ERROR_CANCELLED;
        }

        logits = ss_scheduler_forward(model->scheduler, prompt_tokens[i], i);

        if (model->progress_cb) {
            model->progress_cb((uint32_t)(i + 1), (uint32_t)n_prompt, "prefill",
                               model->progress_user_data);
        }
    }

    if (!logits) {
        g_last_error = SS_ERROR_BACKEND;
        return SS_ERROR_BACKEND;
    }

    // ── Generate tokens ──
    int32_t eos_id = ss_tokenizer_eos_id(model->tokenizer);
    int32_t generated = 0;
    int32_t pos = n_prompt;

    // For stop sequence detection
    char output_buf[8192] = {0};
    int32_t output_len = 0;

    for (uint32_t t = 0; t < p.max_tokens; t++) {
        if (model->cancelled) {
            g_last_error = SS_ERROR_CANCELLED;
            return SS_ERROR_CANCELLED;
        }

        // Sample next token
        int32_t token_id = ss_sample(logits, (int32_t)model->gguf->vocab_size,
                                      p.temperature, p.top_p, p.top_k,
                                      &rng_state);

        // Check for EOS
        if (token_id == eos_id) break;

        // Decode token
        const char *token_text = ss_tokenizer_decode(model->tokenizer, token_id);

        // Check stop sequence
        if (p.stop_sequence && token_text) {
            size_t text_len = strlen(token_text);
            if (output_len + (int32_t)text_len < (int32_t)sizeof(output_buf)) {
                memcpy(output_buf + output_len, token_text, text_len);
                output_len += (int32_t)text_len;
                output_buf[output_len] = '\0';

                if (strstr(output_buf, p.stop_sequence)) break;
            }
        }

        // Token callback
        if (token_cb) {
            bool should_continue = token_cb(token_text, (uint32_t)token_id,
                                             user_data);
            if (!should_continue) break;
        }

        // Forward pass for next token
        logits = ss_scheduler_forward(model->scheduler, token_id, pos);
        if (!logits) break;

        pos++;
        generated++;
    }

    g_last_error = SS_OK;
    return SS_OK;
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

ss_error_t ss_embed(
    ss_model_t *model,
    const char *prompt,
    float     **out_embedding,
    uint32_t   *out_dim
) {
    if (!model || !prompt || !out_embedding || !out_dim) {
        g_last_error = SS_ERROR_INVALID_PARAMS;
        return SS_ERROR_INVALID_PARAMS;
    }

    model->cancelled = false;
    ss_scheduler_reset(model->scheduler);

    // ── Tokenize prompt ──
    int32_t prompt_tokens[4096];
    int32_t n_prompt = ss_tokenizer_encode(model->tokenizer, prompt,
                                           prompt_tokens, 4096);
    if (n_prompt <= 0) {
        g_last_error = SS_ERROR_TOKENIZER;
        return SS_ERROR_TOKENIZER;
    }

    if (model->progress_cb) {
        model->progress_cb(0, (uint32_t)n_prompt, "embedding",
                           model->progress_user_data);
    }

    // ── Pre-fill embedding generation ──
    float *embedding = ss_scheduler_embed(model->scheduler, prompt_tokens, n_prompt);
    
    if (!embedding) {
        g_last_error = SS_ERROR_BACKEND;
        return SS_ERROR_BACKEND;
    }

    // Output dynamic evaluation bindings
    *out_embedding = embedding;
    *out_dim = (uint32_t)model->gguf->hidden_size;

    g_last_error = SS_OK;
    return SS_OK;
}

// ─── Embedding Generation ────────────────────────────────────────────────────

ss_error_t ss_embed(
    ss_model_t *model,
    const char *prompt,
    float     **out_embedding,
    uint32_t   *out_dim
) {
    if (!model || !prompt || !out_embedding || !out_dim) {
        g_last_error = SS_ERROR_INVALID_PARAMS;
        return SS_ERROR_INVALID_PARAMS;
    }

    model->cancelled = false;
    ss_scheduler_reset(model->scheduler);

    // ── Tokenize ──
    int32_t prompt_tokens[4096];
    int32_t n_prompt = ss_tokenizer_encode(model->tokenizer, prompt,
                                            prompt_tokens, 4096);
    if (n_prompt <= 0) {
        g_last_error = SS_ERROR_TOKENIZER;
        return SS_ERROR_TOKENIZER;
    }

    // ── Process ──
    float *result = ss_scheduler_prefill(model->scheduler, prompt_tokens, n_prompt, 0);
    if (!result) {
        g_last_error = SS_ERROR_BACKEND;
        return SS_ERROR_BACKEND;
    }

    // Embeddings are extracted from the last token's hidden state.
    *out_embedding = model->scheduler->state.hidden;
    *out_dim = model->scheduler->state.hidden_size;

    g_last_error = SS_OK;
    return SS_OK;
}

// ─── Cancel ──────────────────────────────────────────────────────────────────

void ss_cancel(ss_model_t *model) {
    if (model) {
        model->cancelled = true;
        if (model->scheduler) {
            model->scheduler->cancelled = true;
        }
    }
}

// ─── Progress ────────────────────────────────────────────────────────────────

void ss_set_progress_callback(ss_model_t *model,
                               ss_progress_callback_t callback,
                               void *user_data) {
    if (!model) return;
    model->progress_cb = callback;
    model->progress_user_data = user_data;
    if (model->scheduler) {
        model->scheduler->progress_cb = callback;
        model->scheduler->progress_user_data = user_data;
    }
}

// ─── Error Handling ──────────────────────────────────────────────────────────

ss_error_t ss_last_error(void) {
    return g_last_error;
}

const char *ss_error_string(ss_error_t error) {
    switch (error) {
        case SS_OK:                    return "OK";
        case SS_ERROR_FILE_NOT_FOUND:  return "File not found";
        case SS_ERROR_INVALID_MODEL:   return "Invalid GGUF model file";
        case SS_ERROR_OUT_OF_MEMORY:   return "Out of memory";
        case SS_ERROR_MMAP_FAILED:     return "Memory mapping failed";
        case SS_ERROR_INVALID_PARAMS:  return "Invalid parameters";
        case SS_ERROR_CANCELLED:       return "Operation cancelled";
        case SS_ERROR_BACKEND:         return "Compute backend error";
        case SS_ERROR_TOKENIZER:       return "Tokenizer error";
        case SS_ERROR_UNKNOWN:
        default:                       return "Unknown error";
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

const char *ss_version(void) {
    return SS_VERSION_STRING;
}

ss_error_t ss_estimate_memory(const char *path, uint32_t context_size,
                               uint64_t *peak_memory) {
    if (!path || !peak_memory) return SS_ERROR_INVALID_PARAMS;

    ss_gguf_file_t *gguf = ss_gguf_open(path);
    if (!gguf) return SS_ERROR_INVALID_MODEL;

    // Peak memory estimation:
    // 1 layer weights + KV cache + activations + working buffers
    uint64_t layer_size = 0;
    if (gguf->layers && gguf->num_layers > 0) {
        // Use the largest layer
        for (uint32_t i = 0; i < gguf->num_layers; i++) {
            if (gguf->layers[i].total_size > layer_size) {
                layer_size = gguf->layers[i].total_size;
            }
        }
    }

    uint32_t ctx = context_size > 0 ? context_size : gguf->context_length;
    uint32_t num_kv_heads = gguf->num_kv_heads > 0 ? gguf->num_kv_heads
                                                     : gguf->num_heads;
    uint32_t head_dim = gguf->hidden_size / gguf->num_heads;
    uint32_t kv_dim = num_kv_heads * head_dim;

    // KV cache: 2 * num_layers * max_seq * kv_dim * sizeof(float)
    uint64_t kv_size = 2ULL * gguf->num_layers * ctx * kv_dim * sizeof(float);

    // Working buffers: hidden + residual + attn + ffn + q/k/v
    uint64_t buffers = (uint64_t)gguf->hidden_size * 4 * sizeof(float)
                     + (uint64_t)gguf->intermediate_size * 2 * sizeof(float)
                     + (uint64_t)gguf->vocab_size * sizeof(float);

    *peak_memory = layer_size + kv_size + buffers;

    ss_gguf_close(gguf);
    return SS_OK;
}

// ─── Cleanup ─────────────────────────────────────────────────────────────────

void ss_model_free(ss_model_t *model) {
    if (!model) return;

#ifdef SS_HAS_METAL
    ss_metal_free(model->metal);
#endif

    ss_thread_pool_destroy(model->thread_pool);
    ss_scheduler_free(model->scheduler);
    ss_tokenizer_free(model->tokenizer);
    ss_gguf_close(model->gguf);

    free(model);
}
