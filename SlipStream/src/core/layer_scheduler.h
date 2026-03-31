/**
 * SlipStream — Layer Scheduler
 *
 * Orchestrates the load → compute → evict cycle for layer-streaming inference.
 */

#ifndef SS_LAYER_SCHEDULER_H
#define SS_LAYER_SCHEDULER_H

#include "core/gguf_loader.h"
#include "core/tensor.h"
#include "core/quantize.h"
#include "slipstream.h"

#include <stdbool.h>

// ─── KV Cache ────────────────────────────────────────────────────────────────

typedef struct {
    float   *key_cache;     // [num_layers, max_seq_len, num_kv_heads * head_dim]
    float   *value_cache;   // [num_layers, max_seq_len, num_kv_heads * head_dim]
    int32_t  max_seq_len;
    int32_t  num_layers;
    int32_t  kv_dim;        // num_kv_heads * head_dim
    int32_t  seq_len;       // Current sequence length

    // ── LZ4 Compressed KV Cache ──
    // Only 1 layer is uncompressed at a time. All others are LZ4-compressed.
    uint8_t **compressed_keys;      // [num_layers] compressed key data
    uint8_t **compressed_values;    // [num_layers] compressed value data
    size_t   *compressed_key_sizes; // [num_layers] compressed sizes
    size_t   *compressed_val_sizes; // [num_layers] compressed sizes
    float    *work_key;             // Working buffer for 1 layer's keys
    float    *work_value;           // Working buffer for 1 layer's values
    size_t    layer_bytes;          // Uncompressed size per layer = max_seq_len * kv_dim * sizeof(float)
    size_t    compress_bound;       // Max compressed size per layer
    int32_t   active_layer;         // Which layer is currently decompressed (-1 = none)
    bool      lz4_enabled;         // Whether LZ4 compression is active
} ss_kv_cache_t;

// ─── Layer State ─────────────────────────────────────────────────────────────

typedef struct {
    // Working buffers (reused across layers)
    float *hidden;          // [hidden_size] — current hidden state
    float *residual;        // [hidden_size] — residual connection
    float *attn_out;        // [hidden_size] — attention output
    float *ffn_buf;         // [intermediate_size] — FFN intermediate
    float *ffn_gate;        // [intermediate_size] — FFN gate
    float *q_buf;           // [num_heads * head_dim] — query projection
    float *k_buf;           // [num_kv_heads * head_dim] — key projection
    float *v_buf;           // [num_kv_heads * head_dim] — value projection
    float *qkv_buf;         // Fused QKV output buffer for architectures with attn_qkv
    float *head_out;        // [head_dim] — single head attention output
    float *logits;          // [vocab_size] — output logits

    float *ffn_up_buf;      // Fused Gate/Up buffer for architectures like Phi-3

    float *scores_buf;      // Pre-allocated attention scores [max_seq_len]
    int32_t scores_buf_size;

    // Dimensions
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t rope_dim;
    int32_t vocab_size;
} ss_layer_state_t;

// ─── Scheduler ───────────────────────────────────────────────────────────────

typedef struct {
    ss_gguf_file_t   *gguf;
    ss_kv_cache_t     kv_cache;
    ss_layer_state_t  state;

    // Config
    int32_t           num_layers;
    float             rms_norm_eps;
    float             rope_theta;
    bool              use_prefetch;
    bool              cancelled;
    void             *metal_ctx;
    bool              is_neox_rope;
    bool              is_gemma;

    // Progress callback
    ss_progress_callback_t progress_cb;
    void                  *progress_user_data;
} ss_scheduler_t;

// ─── API ─────────────────────────────────────────────────────────────────────

/**
 * Initialize the scheduler with a loaded GGUF file.
 * Allocates KV cache and working buffers.
 */
ss_scheduler_t *ss_scheduler_init(ss_gguf_file_t *gguf, uint32_t context_size,
                                   bool prefetch, void *metal_ctx);

/**
 * Process a single token through ALL layers (the core streaming loop).
 * This is the main AirLLM innovation — loads each layer, computes, evicts.
 *
 * @param sched     Scheduler
 * @param token_id  Input token ID
 * @param pos       Position in sequence
 * @return          Pointer to logits [vocab_size] (valid until next call)
 */
float *ss_scheduler_forward(ss_scheduler_t *sched, int32_t token_id, int32_t pos);

/**
 * Process multiple tokens through ALL layers simultaneously.
 * Used during the prefill phase to massively accelerate SSD I/O.
 *
 * @param sched      Scheduler
 * @param tokens     Array of token IDs
 * @param num_tokens Number of tokens
 * @param start_pos  Starting position
 * @return           Pointer to logits for the LAST token [vocab_size]
 */
float *ss_scheduler_prefill(ss_scheduler_t *sched, const int32_t *tokens, int32_t num_tokens, int32_t start_pos);

/**
 * Perform a full forward pass across all tokens to extract Dense Vector Embeddings.
 * Computes mean-pooling over the final hidden states correctly normalized.
 * Returns a dynamically allocated float array of `hidden_size` length.
 * The caller is responsible for freeing this array.
 */
float *ss_scheduler_embed(ss_scheduler_t *sched, const int32_t *tokens, int32_t num_tokens);

/**
 * Reset the scheduler state (clear KV cache, etc.)
 */
void ss_scheduler_reset(ss_scheduler_t *sched);

/**
 * Free the scheduler and all allocated resources.
 */
void ss_scheduler_free(ss_scheduler_t *sched);

#endif // SS_LAYER_SCHEDULER_H
