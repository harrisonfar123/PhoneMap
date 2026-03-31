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

#include <sys/time.h>

static inline double ss_get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

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
    model->tokenizer = ss_tokenizer_create(model->gguf->vocab,
                                            model->gguf->vocab_scores,
                                            model->gguf->vocab_size,
                                            model->gguf->merges,
                                            model->gguf->n_merges,
                                            model->gguf->arch);

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

    // ── Initialize scheduler ──
    uint32_t ctx_size = model->config.context_size > 0 
                      ? model->config.context_size 
                      : model->gguf->context_length;
                      
    if (ctx_size == 0 || ctx_size > 8192) {
        fprintf(stderr, "SlipStream Warning: Model requested context size %u, severely capped to 8192 to prevent extreme iOS iOS memory/LZ4 compression locks!\n", ctx_size);
        ctx_size = 8192;
    }

    model->scheduler = ss_scheduler_init(
        model->gguf,
        ctx_size,
        model->config.prefetch,
        model->metal
    );
    if (!model->scheduler) {
        fprintf(stderr, "SlipStream Error: Failed to initialize scheduler. (OOM or context sizing constraint)\n");
        ss_model_free(model);
        return NULL;
    }

    // ── Initialize thread pool ──
    uint32_t n_threads = model->config.n_threads > 0
                         ? model->config.n_threads
                         : ss_get_cpu_count();
    model->thread_pool = ss_thread_pool_create(n_threads);

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
    
    const char *detected_chat = "unknown";
    if (strstr(model->gguf->arch, "gemma")) detected_chat = "gemma";
    else if (strstr(model->gguf->arch, "phi")) detected_chat = "phi3";
    else if (strstr(model->gguf->arch, "qwen")) detected_chat = "chatml";
    else if (strstr(model->gguf->arch, "llama")) detected_chat = "llama3";
    
    strncpy(info->chat_format, detected_chat, sizeof(info->chat_format) - 1);
    
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
    
    double t_start_prefill = ss_get_time_sec();
    float *logits = ss_scheduler_prefill(model->scheduler, prompt_tokens, n_prompt, 0);
    double t_end_prefill = ss_get_time_sec();
    
    if (model->progress_cb) {
        model->progress_cb((uint32_t)n_prompt, (uint32_t)n_prompt, "prefill",
                           model->progress_user_data);
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

    double t_start_decode = ss_get_time_sec();

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

        // Check stop sequence and robustly detect implicit EOS outputs
        if (token_text && 
            (strcmp(token_text, "<|im_end|>") == 0 ||
             strcmp(token_text, "<|eot_id|>") == 0 ||
             strcmp(token_text, "<|end|>") == 0 ||
             strcmp(token_text, "<|endoftext|>") == 0 ||
             strcmp(token_text, "<|user|>") == 0 ||
             strcmp(token_text, "</s>") == 0 ||
             strcmp(token_text, "<end_of_turn>") == 0 ||
             strcmp(token_text, "<eos>") == 0)) {
            break;
        }

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
    
    double t_end_decode = ss_get_time_sec();
    double prefill_sec = t_end_prefill - t_start_prefill;
    double decode_sec = t_end_decode - t_start_decode;
    
    fprintf(stderr, "\n=== SlipStream Diagnostic Recap ===\n");
    if (prefill_sec > 0.0) {
        fprintf(stderr, "Prefill: %d tokens in %.2f seconds (%.2f tok/s)\n", n_prompt, prefill_sec, (double)n_prompt / prefill_sec);
    }
    if (generated > 0 && decode_sec > 0.0) {
        fprintf(stderr, "Decode : %d tokens in %.2f seconds (%.2f tok/s)\n", generated, decode_sec, (double)generated / decode_sec);
    }
    fprintf(stderr, "===================================\n\n");

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

// ─── Profiler ────────────────────────────────────────────────────────────────
const char *ss_get_last_profile_path(void) {
    return NULL;
}

void ss_set_profile_output_dir(const char *dir) {
    (void)dir;
}

// ─── Throttle ────────────────────────────────────────────────────────────────
void ss_set_throttle(ss_model_t *model, uint32_t delay_us) {
    if (model) {
        // Dummy implementation since struct is inaccessible publicly without modifying engine.h
    }
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
