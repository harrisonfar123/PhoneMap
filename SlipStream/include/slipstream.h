/**
 * SlipStream — Layer-Streaming LLM Inference for Mobile
 *
 * Public C API. Include this single header to use SlipStream.
 *
 * Key concept: Instead of loading the entire model into memory,
 * SlipStream loads one transformer layer at a time from disk,
 * computes its forward pass, then evicts it before loading the next.
 * Peak memory = 1 layer + KV cache + activations.
 */

#ifndef SLIPSTREAM_H
#define SLIPSTREAM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Version ──────────────────────────────────────────────────────────────────

#define SS_VERSION_MAJOR 0
#define SS_VERSION_MINOR 1
#define SS_VERSION_PATCH 0
#define SS_VERSION_STRING "0.1.0"

// ─── Error Codes ──────────────────────────────────────────────────────────────

typedef enum {
    SS_OK                    = 0,
    SS_ERROR_FILE_NOT_FOUND  = -1,
    SS_ERROR_INVALID_MODEL   = -2,
    SS_ERROR_OUT_OF_MEMORY   = -3,
    SS_ERROR_MMAP_FAILED     = -4,
    SS_ERROR_INVALID_PARAMS  = -5,
    SS_ERROR_CANCELLED       = -6,
    SS_ERROR_BACKEND         = -7,
    SS_ERROR_TOKENIZER       = -8,
    SS_ERROR_UNKNOWN         = -99,
} ss_error_t;

// ─── Compute Backend ──────────────────────────────────────────────────────────

typedef enum {
    SS_BACKEND_AUTO   = 0,  // Auto-select best available
    SS_BACKEND_CPU    = 1,  // ARM NEON / scalar CPU
    SS_BACKEND_METAL  = 2,  // Apple Metal GPU
} ss_backend_t;

// ─── Quantization Types ───────────────────────────────────────────────────────

typedef enum {
    SS_QUANT_F32   = 0,
    SS_QUANT_F16   = 1,
    SS_QUANT_Q8_0  = 2,
    SS_QUANT_Q4_0  = 3,
    SS_QUANT_Q4_1  = 4,
} ss_quant_type_t;

// ─── Model Info ───────────────────────────────────────────────────────────────

typedef struct {
    char         model_name[256];
    char         architecture[64];
    uint32_t     num_layers;
    uint32_t     hidden_size;
    uint32_t     num_heads;
    uint32_t     num_kv_heads;
    uint32_t     vocab_size;
    uint32_t     context_length;
    uint32_t     intermediate_size;
    ss_quant_type_t quant_type;
    uint64_t     file_size;
    uint64_t     per_layer_size;
} ss_model_info_t;

// ─── Opaque Model Handle ─────────────────────────────────────────────────────

typedef struct ss_model ss_model_t;

// ─── Model Configuration ─────────────────────────────────────────────────────

typedef struct {
    ss_backend_t backend;       // Which compute backend to use
    uint32_t     n_threads;     // Number of CPU threads (0 = auto)
    uint64_t     max_memory;    // Max memory budget in bytes (0 = no limit)
    bool         use_mmap;      // Use memory-mapped I/O (default: true)
    bool         prefetch;      // Prefetch next layer while computing (default: true)
    uint32_t     context_size;  // Max context window (0 = model default)
} ss_model_config_t;

// ─── Generation Parameters ───────────────────────────────────────────────────

typedef struct {
    uint32_t    max_tokens;     // Max tokens to generate
    float       temperature;    // Sampling temperature (0.0 = greedy)
    float       top_p;          // Top-p (nucleus) sampling
    uint32_t    top_k;          // Top-k sampling
    float       repeat_penalty; // Repetition penalty
    uint32_t    seed;           // Random seed (0 = random)
    const char *stop_sequence;  // Stop generation at this string (NULL = none)
} ss_generate_params_t;

// ─── Callback Types ──────────────────────────────────────────────────────────

/**
 * Token callback — called for each generated token.
 *
 * @param token_text  The decoded token string (UTF-8)
 * @param token_id    The token ID
 * @param user_data   User-provided context pointer
 * @return            true to continue generation, false to stop
 */
typedef bool (*ss_token_callback_t)(
    const char *token_text,
    uint32_t    token_id,
    void       *user_data
);

/**
 * Progress callback — called during model loading and layer processing.
 *
 * @param current     Current progress (e.g., current layer index)
 * @param total       Total items (e.g., total layers)
 * @param phase       Description string ("loading", "computing layer 5/32", etc.)
 * @param user_data   User-provided context pointer
 */
typedef void (*ss_progress_callback_t)(
    uint32_t    current,
    uint32_t    total,
    const char *phase,
    void       *user_data
);

// ─── Default Configurations ──────────────────────────────────────────────────

/**
 * Returns a model config with sensible defaults.
 */
ss_model_config_t ss_default_config(void);

/**
 * Returns generation params with sensible defaults.
 * temperature=0.7, top_p=0.9, top_k=40, max_tokens=512
 */
ss_generate_params_t ss_default_params(void);

// ─── Model Lifecycle ─────────────────────────────────────────────────────────

/**
 * Load a GGUF model from disk.
 * This parses the header and builds the layer index but does NOT
 * load weights into memory — that happens during generation.
 *
 * @param path    Path to the .gguf file
 * @param config  Model configuration (NULL for defaults)
 * @return        Model handle, or NULL on failure (check ss_last_error())
 */
ss_model_t *ss_model_load(const char *path, const ss_model_config_t *config);

/**
 * Get model information (architecture, layer count, sizes, etc.)
 */
ss_error_t ss_model_get_info(const ss_model_t *model, ss_model_info_t *info);

/**
 * Free a loaded model and all associated resources.
 */
void ss_model_free(ss_model_t *model);

// ─── Text Generation ─────────────────────────────────────────────────────────

/**
 * Generate text from a prompt using layer-streaming inference.
 *
 * For each token generated:
 *   1. For each layer 0..N:
 *      a. mmap layer weights from GGUF
 *      b. Dequantize if needed
 *      c. Compute layer forward pass
 *      d. Evict layer from memory
 *   2. Sample next token
 *   3. Call token_callback with the new token
 *
 * @param model         Loaded model handle
 * @param prompt        Input prompt (UTF-8 string)
 * @param params        Generation parameters (NULL for defaults)
 * @param token_cb      Callback for each token (NULL to generate silently)
 * @param user_data     User data passed to callbacks
 * @return              SS_OK on success, error code on failure
 */
ss_error_t ss_generate(
    ss_model_t                *model,
    const char                *prompt,
    const ss_generate_params_t *params,
    ss_token_callback_t        token_cb,
    void                      *user_data
);

/**
 * Cancel an in-progress generation (thread-safe).
 */
void ss_cancel(ss_model_t *model);

// ─── Embedding Generation ──────────────────────────────────────────────────────

/**
 * Generate a dense vector embedding for the input text.
 * The model must support embeddings (e.g., nomic-embed).
 * This will return a pointer to the internal hidden state representation.
 * 
 * @param model         Loaded model handle
 * @param prompt        Input text to embed
 * @param out_embedding Pointer to a float array that will receive the embedding
 * @param out_dim       Pointer to an integer receiving the vector dimension
 * @return              SS_OK on success, error code on failure
 */
ss_error_t ss_embed(
    ss_model_t *model,
    const char *prompt,
    float     **out_embedding,
    uint32_t   *out_dim
);

// ─── Progress Monitoring ─────────────────────────────────────────────────────

/**
 * Set a progress callback for model loading and inference.
 */
void ss_set_progress_callback(
    ss_model_t              *model,
    ss_progress_callback_t   callback,
    void                    *user_data
);

// ─── Error Handling ──────────────────────────────────────────────────────────

/**
 * Get the last error that occurred.
 */
ss_error_t ss_last_error(void);

/**
 * Get a human-readable description of an error code.
 */
const char *ss_error_string(ss_error_t error);

// ─── Utility ─────────────────────────────────────────────────────────────────

/**
 * Get the SlipStream version string.
 */
const char *ss_version(void);

/**
 * Estimate peak memory usage for a given model file.
 * Useful for checking if a model can run on the current device.
 *
 * @param path           Path to the .gguf file
 * @param context_size   Desired context window size
 * @param peak_memory    Output: estimated peak memory in bytes
 * @return               SS_OK on success
 */
ss_error_t ss_estimate_memory(
    const char *path,
    uint32_t    context_size,
    uint64_t   *peak_memory
);

#ifdef __cplusplus
}
#endif

#endif // SLIPSTREAM_H
