/**
 * SlipStream — Layer Scheduler Implementation
 *
 * The core layer-streaming inference loop:
 *   For each token → For each layer:
 *     1. mmap layer weights
 *     2. Dequantize if quantized
 *     3. Compute: Attention → FFN
 *     4. Evict layer weights (madvise DONTNEED)
 *     5. Prefetch next layer (madvise WILLNEED)
 */

#include "core/layer_scheduler.h"
#include "utils/memory.h"
#include "backends/metal_backend.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

// ─── Helper: Find tensor in layer by name suffix ─────────────────────────────

static const ss_tensor_info_t *find_tensor(const ss_layer_info_t *layer,
                                            const char *suffix) {
    for (uint32_t i = 0; i < layer->num_tensors; i++) {
        const char *name = layer->tensors[i].name;
        size_t name_len = strlen(name);
        size_t suffix_len = strlen(suffix);
        if (name_len >= suffix_len &&
            strcmp(name + name_len - suffix_len, suffix) == 0) {
            return &layer->tensors[i];
        }
    }
    return NULL;
}

// Find tensor in the global tensor list by exact name
static const ss_tensor_info_t *find_global_tensor(const ss_gguf_file_t *gguf,
                                                    const char *name) {
    for (uint64_t i = 0; i < gguf->tensor_count; i++) {
        if (strcmp(gguf->tensors[i].name, name) == 0) {
            return &gguf->tensors[i];
        }
    }
    return NULL;
}

// ─── KV Cache with LZ4 Compression ──────────────────────────────────────────
// Only 1 layer is uncompressed at a time. All others are LZ4-compressed.
// Apple's native compression.h provides COMPRESSION_LZ4 for zero-dependency use.

static bool kv_cache_init(ss_kv_cache_t *cache, int32_t num_layers,
                           int32_t max_seq_len, int32_t kv_dim) {
    cache->num_layers = num_layers;
    cache->max_seq_len = max_seq_len;
    cache->kv_dim = kv_dim;
    cache->seq_len = 0;

    size_t total_vectors = (size_t)num_layers * max_seq_len;
    size_t total_int8 = total_vectors * kv_dim;
    
    cache->q_key_cache = (int8_t *)calloc(total_int8, sizeof(int8_t));
    cache->q_val_cache = (int8_t *)calloc(total_int8, sizeof(int8_t));
    cache->k_scales = (float *)calloc(total_vectors, sizeof(float));
    cache->v_scales = (float *)calloc(total_vectors, sizeof(float));
    
    cache->work_key = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
    cache->work_value = (float *)calloc(max_seq_len * kv_dim, sizeof(float));

    if (!cache->q_key_cache || !cache->q_val_cache ||
        !cache->k_scales || !cache->v_scales ||
        !cache->work_key || !cache->work_value) {
        return false;
    }

    return true;
}

static void kv_cache_free(ss_kv_cache_t *cache) {
    free(cache->q_key_cache);
    free(cache->q_val_cache);
    free(cache->k_scales);
    free(cache->v_scales);
    free(cache->work_key);
    free(cache->work_value);
    
    cache->q_key_cache = NULL;
    cache->q_val_cache = NULL;
    cache->k_scales = NULL;
    cache->v_scales = NULL;
    cache->work_key = NULL;
    cache->work_value = NULL;
}

// ─── TurboQuant Dynamic Quantization ─────────────────────────────────────────

static inline void kv_turboquant_encode(const float *src, int8_t *dst_q, float *dst_s, int32_t len) {
    // 1. Find absolute max to determine structural grid scalar boundary
    float max_val = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        float v = fabsf(src[i]);
        if (v > max_val) max_val = v;
    }
    
    // 2. Compute symmetric 8-bit mapping scale (-127 to +127)
    float scale = max_val / 127.0f;
    *dst_s = scale;
    
    // 3. Compress directly to int8_t
    float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int32_t i = 0; i < len; i++) {
        dst_q[i] = (int8_t)roundf(src[i] * inv_scale);
    }
}

static inline void kv_turboquant_decode(const int8_t *src_q, float src_s, float *dst, int32_t len) {
    // De-quantize cleanly directly back to Floating Point math
    for (int32_t i = 0; i < len; i++) {
        dst[i] = (float)src_q[i] * src_s;
    }
}

// ─── Initialize ──────────────────────────────────────────────────────────────

ss_scheduler_t *ss_scheduler_init(ss_gguf_file_t *gguf, uint32_t context_size,
                                   bool prefetch, void *metal_ctx) {
    if (!gguf) return NULL;

    ss_scheduler_t *sched = (ss_scheduler_t *)calloc(1, sizeof(ss_scheduler_t));
    if (!sched) return NULL;

    sched->gguf = gguf;
    sched->num_layers = (int32_t)gguf->num_layers;
    sched->rms_norm_eps = gguf->rms_norm_eps;
    sched->rope_theta = gguf->rope_theta;
    sched->use_prefetch = prefetch;
    sched->cancelled = false;
    sched->metal_ctx = metal_ctx;
    
    // Set RoPE style based on architecture (Qwen, Falcon, Phi, DeepSeek use Neox-style)
    sched->is_neox_rope = false;
    if (strstr(gguf->arch, "qwen") || strstr(gguf->arch, "falcon") || 
        strstr(gguf->arch, "phi") || strstr(gguf->arch, "deepseek")) {
        sched->is_neox_rope = true;
    }
    
    // Detect Gemma architectures for distinct activation and scaling behaviors
    sched->is_gemma = false;
    if (strstr(gguf->arch, "gemma")) {
        sched->is_gemma = true;
    }

    int32_t hidden_size = (int32_t)gguf->hidden_size;
    int32_t num_heads = (int32_t)gguf->num_heads;
    int32_t num_kv_heads = (int32_t)gguf->num_kv_heads;
    if (num_kv_heads == 0) num_kv_heads = num_heads;
    int32_t head_dim = 0;
    if (gguf->head_dim > 0) {
        head_dim = (int32_t)gguf->head_dim;
    } else {
        // Find blk.0.attn_q.weight or wqkv to deduce head_dim reliably
        for (uint64_t i = 0; i < gguf->tensor_count; i++) {
            if (strcmp(gguf->tensors[i].name, "blk.0.attn_q.weight") == 0) {
                head_dim = (int32_t)(gguf->tensors[i].dims[1] / num_heads);
                break;
            } else if (strcmp(gguf->tensors[i].name, "blk.0.attn_qkv.weight") == 0) {
                int32_t total_heads = num_heads + 2 * num_kv_heads;
                head_dim = (int32_t)(gguf->tensors[i].dims[1] / total_heads);
                break;
            }
        }
    }
    
    // Absolute fallback
    if (head_dim == 0) {
        head_dim = hidden_size / num_heads; 
    }

    int32_t rope_dim = gguf->rope_dim > 0 ? (int32_t)gguf->rope_dim : head_dim;
    int32_t intermediate_size = (int32_t)gguf->intermediate_size;
    int32_t vocab_size = (int32_t)gguf->vocab_size;
    
    // Some quantized/padded GGUFs have mismatched metadata vocab_size.
    // Derive the true addressable logits space directly from output projection bounds.
    for (uint64_t i = 0; i < gguf->tensor_count; i++) {
        if ((strcmp(gguf->tensors[i].name, "output.weight") == 0 ||
             strcmp(gguf->tensors[i].name, "token_embd.weight") == 0) &&
            gguf->tensors[i].n_dims == 2) {
            if ((int32_t)gguf->tensors[i].dims[1] < vocab_size) {
                vocab_size = (int32_t)gguf->tensors[i].dims[1];
            }
        }
    }

    int32_t max_seq = context_size > 0 ? (int32_t)context_size
                                        : (int32_t)gguf->context_length;
    // iOS safety cap: KV cache at 512 ctx ≈ 128MB. Beyond this, Jetsam kills the process.
    if (max_seq > 512) max_seq = 512;

    // Initialize KV cache
    int32_t kv_dim = num_kv_heads * head_dim;
    if (!kv_cache_init(&sched->kv_cache, sched->num_layers, max_seq, kv_dim)) {
        free(sched);
        return NULL;
    }

    // Initialize working state
    ss_layer_state_t *s = &sched->state;
    s->hidden_size = hidden_size;
    s->intermediate_size = intermediate_size;
    s->num_heads = num_heads;
    s->num_kv_heads = num_kv_heads;
    s->head_dim = head_dim;
    s->rope_dim = rope_dim;
    s->vocab_size = vocab_size;

    s->hidden   = (float *)calloc(hidden_size, sizeof(float));
    s->residual = (float *)calloc(hidden_size, sizeof(float));
    s->attn_out = (float *)calloc(hidden_size, sizeof(float));
    s->ffn_buf    = (float *)calloc(intermediate_size, sizeof(float));
    s->ffn_gate   = (float *)calloc(intermediate_size, sizeof(float));
    s->ffn_up_buf = (float *)calloc(intermediate_size * 2, sizeof(float));
    s->q_buf      = (float *)calloc(num_heads * head_dim, sizeof(float));
    s->k_buf      = (float *)calloc(kv_dim, sizeof(float));
    s->v_buf      = (float *)calloc(kv_dim, sizeof(float));
    s->qkv_buf    = (float *)calloc(num_heads * head_dim + 2 * kv_dim, sizeof(float));
    s->head_out   = (float *)calloc(head_dim, sizeof(float));
    s->logits     = (float *)calloc(vocab_size, sizeof(float));
    s->scores_buf = (float *)calloc(max_seq, sizeof(float));
    s->scores_buf_size = max_seq;

    if (!s->hidden || !s->residual || !s->attn_out || !s->ffn_buf ||
        !s->ffn_gate || !s->ffn_up_buf || !s->q_buf || !s->k_buf || !s->v_buf || 
        !s->qkv_buf || !s->head_out || !s->logits || !s->scores_buf) {
        ss_scheduler_free(sched);
        return NULL;
    }

    return sched;
}

// ─── Dispatch Helper ────────────────────────────────────────────────────────

static void ss_matmul_dispatch(ss_scheduler_t *sched, float *out, const void *mat, const float *vec,
                               int32_t rows, int32_t cols, ggml_type_t type) {
#ifdef SS_HAS_METAL
    if (sched->metal_ctx && ss_metal_should_use_gpu(rows, cols)) {
        bool gpu_success = false;
        if (type == GGML_TYPE_Q4_0) {
            gpu_success = ss_metal_matvec_q4_0(sched->metal_ctx, out, mat, vec, rows, cols);
        } else if (type == GGML_TYPE_Q4_K) {
            gpu_success = ss_metal_matvec_q4_K(sched->metal_ctx, out, mat, vec, rows, cols);
        } else if (type == GGML_TYPE_F32) {
            gpu_success = ss_metal_matvec(sched->metal_ctx, out, (const float *)mat, vec, rows, cols);
        }
        
        // If GPU execution succeeded we are done!
        if (gpu_success) return;
        // Otherwise, execution was aborted (e.g. background suspension, or unsupported format), fall through to CPU NEON.
    }
#endif

    // Fallback: execute on CPU NEON natively.
    static int printed_cpu_fallback = 0;
    if (printed_cpu_fallback < 25) {
        fprintf(stderr, "SlipStream CPU Fallback Warning: CPU taking over tensor evaluation for ggml_type_t enum: %d (rows: %d, cols: %d)\n", type, rows, cols);
        printed_cpu_fallback++;
    }

    if (type == GGML_TYPE_F32) {
        ss_matvec(out, (const float *)mat, vec, rows, cols);
    } else {
        ss_dequant_matvec(out, mat, vec, rows, cols, type);
    }
}

// ─── Core: Process One Layer ─────────────────────────────────────────────────

// Helper to safely load bias/norm data handling both F32 and F16
static inline float safe_load_f32(const void *data, int32_t idx, ggml_type_t type) {
    if (type == GGML_TYPE_F16) {
        return ss_f16_to_f32(((const uint16_t *)data)[idx]);
    }
    return ((const float *)data)[idx];
}

static void process_layer(ss_scheduler_t *sched, int32_t layer_idx, int32_t pos) {
    ss_gguf_file_t *gguf = sched->gguf;
    ss_layer_state_t *s = &sched->state;
    ss_layer_info_t *layer = &gguf->layers[layer_idx];

    // ── Find layer tensors ──
    const ss_tensor_info_t *attn_norm   = find_tensor(layer, "attn_norm.weight");
    const ss_tensor_info_t *ffn_norm    = find_tensor(layer, "ffn_norm.weight");
    const ss_tensor_info_t *wqkv        = find_tensor(layer, "attn_qkv.weight");
    const ss_tensor_info_t *wq          = find_tensor(layer, "attn_q.weight");
    const ss_tensor_info_t *wk          = find_tensor(layer, "attn_k.weight");
    const ss_tensor_info_t *wv          = find_tensor(layer, "attn_v.weight");
    const ss_tensor_info_t *wo          = find_tensor(layer, "attn_output.weight");
    const ss_tensor_info_t *w_gate      = find_tensor(layer, "ffn_gate.weight");
    const ss_tensor_info_t *w_up        = find_tensor(layer, "ffn_up.weight");
    const ss_tensor_info_t *w_down      = find_tensor(layer, "ffn_down.weight");
    
    // Gemma 2 specific architecture features
    const ss_tensor_info_t *post_attn_norm = find_tensor(layer, "post_attention_norm.weight");
    const ss_tensor_info_t *post_ffn_norm  = find_tensor(layer, "post_ffw_norm.weight");

    // Qwen2/2.5 has attention bias tensors (optional for other architectures)
    const ss_tensor_info_t *bq          = find_tensor(layer, "attn_q.bias");
    const ss_tensor_info_t *bk          = find_tensor(layer, "attn_k.bias");
    const ss_tensor_info_t *bv          = find_tensor(layer, "attn_v.bias");
    const ss_tensor_info_t *bqkv        = find_tensor(layer, "attn_qkv.bias");

    if (!attn_norm || (!wqkv && (!wq || !wk || !wv)) || !wo) return;

    // ── 1. Attention Pre-Norm ──
    const float *norm_weight = (const float *)ss_gguf_tensor_data(gguf, attn_norm);
    if (!norm_weight) return;

    // Save residual
    memcpy(s->residual, s->hidden, s->hidden_size * sizeof(float));

    // RMSNorm
    if (sched->is_gemma) {
        ss_gemma_rmsnorm(s->hidden, s->residual, norm_weight, s->hidden_size,
                         sched->rms_norm_eps, attn_norm->type);
    } else {
        ss_rmsnorm(s->hidden, s->residual, norm_weight, s->hidden_size,
                   sched->rms_norm_eps, attn_norm->type);
    }

    // ── 2. QKV Projections ──
    if (wqkv) {
        const void *wqkv_data = ss_gguf_tensor_data(gguf, wqkv);
        int32_t total_qkv_rows = s->num_heads * s->head_dim + 2 * s->num_kv_heads * s->head_dim;
        ss_matmul_dispatch(sched, s->qkv_buf, wqkv_data, s->hidden, total_qkv_rows, s->hidden_size, wqkv->type);
        
        int32_t q_size = s->num_heads * s->head_dim;
        int32_t kv_size = s->num_kv_heads * s->head_dim;
        memcpy(s->q_buf, s->qkv_buf, q_size * sizeof(float));
        memcpy(s->k_buf, s->qkv_buf + q_size, kv_size * sizeof(float));
        memcpy(s->v_buf, s->qkv_buf + q_size + kv_size, kv_size * sizeof(float));
    } else {
        const void *wq_data = ss_gguf_tensor_data(gguf, wq);
        const void *wk_data = ss_gguf_tensor_data(gguf, wk);
        const void *wv_data = ss_gguf_tensor_data(gguf, wv);

#ifdef SS_HAS_METAL
        if (sched->metal_ctx) ss_metal_begin_batch(sched->metal_ctx);
#endif

        ss_matmul_dispatch(sched, s->q_buf, wq_data, s->hidden, s->num_heads * s->head_dim, s->hidden_size, wq->type);
        ss_matmul_dispatch(sched, s->k_buf, wk_data, s->hidden, s->num_kv_heads * s->head_dim, s->hidden_size, wk->type);
        ss_matmul_dispatch(sched, s->v_buf, wv_data, s->hidden, s->num_kv_heads * s->head_dim, s->hidden_size, wv->type);

#ifdef SS_HAS_METAL
        if (sched->metal_ctx) ss_metal_end_batch(sched->metal_ctx);
#endif
    }

    // Apply QKV bias if present (required for Qwen2/2.5)
    // Vectorized with vDSP_vadd for F32 biases (the common case)
    if (bq) {
        const void *bq_data = ss_gguf_tensor_data(gguf, bq);
        if (bq_data) {
            int32_t q_size = s->num_heads * s->head_dim;
            if (bq->type == GGML_TYPE_F32) {
                vDSP_vadd(s->q_buf, 1, (const float *)bq_data, 1, s->q_buf, 1, q_size);
            } else {
                for (int32_t i = 0; i < q_size; i++)
                    s->q_buf[i] += safe_load_f32(bq_data, i, bq->type);
            }
        }
    }
    if (bk) {
        const void *bk_data = ss_gguf_tensor_data(gguf, bk);
        if (bk_data) {
            int32_t kv_size = s->num_kv_heads * s->head_dim;
            if (bk->type == GGML_TYPE_F32) {
                vDSP_vadd(s->k_buf, 1, (const float *)bk_data, 1, s->k_buf, 1, kv_size);
            } else {
                for (int32_t i = 0; i < kv_size; i++)
                    s->k_buf[i] += safe_load_f32(bk_data, i, bk->type);
            }
        }
    }
    if (bv) {
        const void *bv_data = ss_gguf_tensor_data(gguf, bv);
        if (bv_data) {
            int32_t kv_size = s->num_kv_heads * s->head_dim;
            if (bv->type == GGML_TYPE_F32) {
                vDSP_vadd(s->v_buf, 1, (const float *)bv_data, 1, s->v_buf, 1, kv_size);
            } else {
                for (int32_t i = 0; i < kv_size; i++)
                    s->v_buf[i] += safe_load_f32(bv_data, i, bv->type);
            }
        }
    }

    // ── 3. RoPE ──
    ss_rope(s->q_buf, s->k_buf, s->num_heads, s->num_kv_heads,
            s->head_dim, s->rope_dim, pos, sched->rope_theta, sched->is_neox_rope);

    // ── 4. TurboQuant Dynamic Cache Initialization ──
    int32_t kv_size = s->num_kv_heads * s->head_dim;
    size_t vector_idx = (size_t)layer_idx * sched->kv_cache.max_seq_len + pos;
    
    kv_turboquant_encode(s->k_buf, sched->kv_cache.q_key_cache + vector_idx * kv_size, &sched->kv_cache.k_scales[vector_idx], kv_size);
    kv_turboquant_encode(s->v_buf, sched->kv_cache.q_val_cache + vector_idx * kv_size, &sched->kv_cache.v_scales[vector_idx], kv_size);

    // ── 5. Multi-Head Attention ──
    int32_t seq_len = pos + 1;
    int32_t heads_per_kv = s->num_heads / s->num_kv_heads;

    memset(s->attn_out, 0, s->hidden_size * sizeof(float));

    float scale = 1.0f / sqrtf((float)s->head_dim);

    // Parallel attention heads: each head writes to a non-overlapping region of attn_out
    dispatch_apply(s->num_heads, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t h_idx) {
        int32_t h = (int32_t)h_idx;
        float *q = s->q_buf + h * s->head_dim;
        int32_t kv_head = h / heads_per_kv;

        // Per-thread secure localized arrays! Max seq 512, Max Head Dim 256
        float scores[512]; 
        float tmp_kv[256]; 

        // Q @ K^T for all positions — vectorized 8-Bit Decompression
        for (int32_t t = 0; t < seq_len; t++) {
            size_t v_idx = (size_t)layer_idx * sched->kv_cache.max_seq_len + t;
            int8_t *q_k = sched->kv_cache.q_key_cache + v_idx * kv_size + kv_head * s->head_dim;
            float k_scale = sched->kv_cache.k_scales[v_idx];
            
            kv_turboquant_decode(q_k, k_scale, tmp_kv, s->head_dim);
            
            float dot = 0.0f;
            vDSP_dotpr(q, 1, tmp_kv, 1, &dot, s->head_dim);
            scores[t] = dot * scale;
        }

        if (sched->gguf->attn_logit_softcapping > 0.0f) {
            ss_softcap(scores, seq_len, sched->gguf->attn_logit_softcapping);
        }

        // Softmax
        ss_softmax(scores, seq_len);

        // Weighted sum of 8-Bit Dynamic Values → directly into attn_out slice
        float *head_dst = s->attn_out + h * s->head_dim;
        memset(head_dst, 0, s->head_dim * sizeof(float));
        for (int32_t t = 0; t < seq_len; t++) {
            float w = scores[t];
            
            size_t v_idx = (size_t)layer_idx * sched->kv_cache.max_seq_len + t;
            int8_t *q_v = sched->kv_cache.q_val_cache + v_idx * kv_size + kv_head * s->head_dim;
            float v_scale = sched->kv_cache.v_scales[v_idx];
            
            kv_turboquant_decode(q_v, v_scale, tmp_kv, s->head_dim);
            
            // cblas_saxpy: head_dst += w * vt (vectorized NEON/AMX)
            cblas_saxpy(s->head_dim, w, tmp_kv, 1, head_dst, 1);
        }
    });

    // ── 6. Output Projection ──
    const void *wo_data = ss_gguf_tensor_data(gguf, wo);
    ss_matmul_dispatch(sched, s->hidden, wo_data, s->attn_out, s->hidden_size, s->num_heads * s->head_dim, wo->type);

    // Apply Gemma 2 Post-Attention Norm natively before the residual
    if (post_attn_norm) {
        const float *post_attn_w = (const float *)ss_gguf_tensor_data(gguf, post_attn_norm);
        if (post_attn_w) {
            ss_gemma_rmsnorm(s->hidden, s->hidden, post_attn_w, s->hidden_size,
                             sched->rms_norm_eps, post_attn_norm->type);
        }
    }

    // Add residual (vectorized)
    vDSP_vadd(s->hidden, 1, s->residual, 1, s->hidden, 1, s->hidden_size);

    // ── 8. FFN Pre-Norm ──
    if (!ffn_norm || (!w_gate && !w_up) || !w_down) return;

    const float *ffn_norm_weight = (const float *)ss_gguf_tensor_data(gguf, ffn_norm);
    memcpy(s->residual, s->hidden, s->hidden_size * sizeof(float));

    if (sched->is_gemma) {
        ss_gemma_rmsnorm(s->hidden, s->residual, ffn_norm_weight, s->hidden_size,
                         sched->rms_norm_eps, ffn_norm->type);
    } else {
        ss_rmsnorm(s->hidden, s->residual, ffn_norm_weight, s->hidden_size,
                   sched->rms_norm_eps, ffn_norm->type);
    }

    // ── 9. FFN: SwiGLU / GeGLU ──
    if (w_gate && w_up) {
        const void *gate_data = ss_gguf_tensor_data(gguf, w_gate);
        const void *up_data   = ss_gguf_tensor_data(gguf, w_up);

        // Execute sequentially to avoid nested GCD thread-pool starvation
        // since ss_matmul_dispatch already parallelizes across cores.

#ifdef SS_HAS_METAL
        if (sched->metal_ctx) ss_metal_begin_batch(sched->metal_ctx);
#endif
        
        ss_matmul_dispatch(sched, s->ffn_gate, gate_data, s->hidden, s->intermediate_size, s->hidden_size, w_gate->type);
        ss_matmul_dispatch(sched, s->ffn_buf, up_data, s->hidden, s->intermediate_size, s->hidden_size, w_up->type);

#ifdef SS_HAS_METAL
        if (sched->metal_ctx) ss_metal_end_batch(sched->metal_ctx);
#endif
    } else if (w_up) {
        const void *up_data = ss_gguf_tensor_data(gguf, w_up);
        int32_t fused_rows = 2 * s->intermediate_size;
        ss_matmul_dispatch(sched, s->ffn_up_buf, up_data, s->hidden, fused_rows, s->hidden_size, w_up->type);

        memcpy(s->ffn_gate, s->ffn_up_buf, s->intermediate_size * sizeof(float));
        memcpy(s->ffn_buf, s->ffn_up_buf + s->intermediate_size, s->intermediate_size * sizeof(float));
    } else {
        return;
    }

    if (sched->is_gemma) {
        ss_geglu(s->ffn_gate, s->intermediate_size);
        ss_elementwise_mul(s->ffn_buf, s->ffn_buf, s->ffn_gate, s->intermediate_size);
    } else {
        // Fused SiLU(gate) * up — single pass to halve memory traffic
        {
            int32_t n = s->intermediate_size;
            int32_t i = 0;
            for (; i + 3 < n; i += 4) {
                float g0 = s->ffn_gate[i],     g1 = s->ffn_gate[i+1],
                      g2 = s->ffn_gate[i+2],   g3 = s->ffn_gate[i+3];
                s->ffn_buf[i]   *= g0 / (1.0f + expf(-g0));
                s->ffn_buf[i+1] *= g1 / (1.0f + expf(-g1));
                s->ffn_buf[i+2] *= g2 / (1.0f + expf(-g2));
                s->ffn_buf[i+3] *= g3 / (1.0f + expf(-g3));
            }
            for (; i < n; i++) {
                float g = s->ffn_gate[i];
                s->ffn_buf[i] *= g / (1.0f + expf(-g));
            }
        }
    }

    // Down projection
    const void *down_data = ss_gguf_tensor_data(gguf, w_down);
    ss_matmul_dispatch(sched, s->hidden, down_data, s->ffn_buf, s->hidden_size, s->intermediate_size, w_down->type);

    // Apply Gemma 2 Post-FFN Norm natively before residual bridging
    if (post_ffn_norm) {
        const float *post_ffn_w = (const float *)ss_gguf_tensor_data(gguf, post_ffn_norm);
        if (post_ffn_w) {
            ss_gemma_rmsnorm(s->hidden, s->hidden, post_ffn_w, s->hidden_size,
                             sched->rms_norm_eps, post_ffn_norm->type);
        }
    }

    // ── 10. Residual Add ──
    ss_elementwise_add(s->hidden, s->hidden, s->residual, s->hidden_size);
}

// ─── Core: Full Forward Pass ────────────────────────────────────────────────

float *ss_scheduler_forward(ss_scheduler_t *sched, int32_t token_id, int32_t pos) {
    if (!sched || sched->cancelled) return NULL;

    ss_gguf_file_t *gguf = sched->gguf;
    ss_layer_state_t *s = &sched->state;

    // ── Embedding lookup ──
    const ss_tensor_info_t *embed = find_global_tensor(gguf, "token_embd.weight");
    if (!embed) return NULL;

    const void *embed_data = ss_gguf_tensor_data(gguf, embed);
    if (!embed_data) return NULL;

    // Extract embedding for this token
    if (embed->type == GGML_TYPE_F32) {
        const float *emb = (const float *)embed_data + (size_t)token_id * s->hidden_size;
        memcpy(s->hidden, emb, s->hidden_size * sizeof(float));
    } else {
        // Dequantize just this token's embedding
        size_t type_sz = ss_ggml_type_size(embed->type);
        uint32_t block_sz = ss_ggml_block_size(embed->type);
        size_t row_bytes = ((size_t)s->hidden_size / block_sz) * type_sz;
        const uint8_t *row = (const uint8_t *)embed_data + (size_t)token_id * row_bytes;
        ss_dequantize(row, s->hidden, s->hidden_size, embed->type);
    }

    if (sched->is_gemma) {
        float sq = sqrtf((float)s->hidden_size);
        for (int32_t i = 0; i < s->hidden_size; i++) {
            s->hidden[i] *= sq;
        }
    }

    // ── Process all layers (THE STREAMING LOOP) ──
    for (int32_t l = 0; l < sched->num_layers; l++) {
        if (sched->cancelled) return NULL;

        // Prefetch next 2 layers asynchronously for better I/O overlap on large models.
        // For 7B+ models, each layer can be 200MB+. Starting the madvise early ensures
        // the kernel begins paging in while we compute the current layer.
        if (sched->use_prefetch) {
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                if (l + 1 < sched->num_layers) ss_gguf_prefetch_layer(gguf, (uint32_t)(l + 1));
                if (l + 2 < sched->num_layers) ss_gguf_prefetch_layer(gguf, (uint32_t)(l + 2));
            });
        }

        // TurboQuant removes LZ4 un-compression step natively.
        // Process this layer
        process_layer(sched, l, pos);

        // Evict this layer's weights from page cache
        ss_gguf_evict_layer(gguf, (uint32_t)l);

        // Progress callback
        if (sched->progress_cb) {
            char phase[64];
            snprintf(phase, sizeof(phase), "layer %d/%d", l + 1, sched->num_layers);
            sched->progress_cb((uint32_t)(l + 1), (uint32_t)sched->num_layers,
                               phase, sched->progress_user_data);
        }
    }

    // ── Final RMS Norm ──
    const ss_tensor_info_t *final_norm = find_global_tensor(gguf, "output_norm.weight");
    if (final_norm) {
        const float *norm_w = (const float *)ss_gguf_tensor_data(gguf, final_norm);
        if (norm_w) {
            if (sched->is_gemma) {
                ss_gemma_rmsnorm(s->hidden, s->hidden, norm_w, s->hidden_size,
                                 sched->rms_norm_eps, final_norm->type);
            } else {
                ss_rmsnorm(s->hidden, s->hidden, norm_w, s->hidden_size,
                           sched->rms_norm_eps, final_norm->type);
            }
        }
    }


    // ── Output head (LM head) ──
    const ss_tensor_info_t *lm_head = find_global_tensor(gguf, "output.weight");
    if (!lm_head) {
        // Some models tie embeddings — use token_embd as lm_head
        lm_head = embed;
    }

    if (lm_head) {
        const void *head_data = ss_gguf_tensor_data(gguf, lm_head);
        if (head_data) {
            ss_matmul_dispatch(sched, s->logits, head_data, s->hidden, s->vocab_size, s->hidden_size, lm_head->type);
        }
    }

    // Gemma 1/2 mathematically requires scaling logit outputs down by sqrt(hidden_size)
    // because the token embeddings are inherently scaled UP at the input layer.
    // Failing to do so explodes the dots and destroys softmax entropy!
    if (sched->is_gemma) {
        float inv_sq = 1.0f / sqrtf((float)s->hidden_size);
        for (int32_t i = 0; i < s->vocab_size; i++) {
            s->logits[i] *= inv_sq;
        }
    }

    if (gguf->final_logit_softcapping > 0.0f) {
        ss_softcap(s->logits, s->vocab_size, gguf->final_logit_softcapping);
    }

    return s->logits;
}

float *ss_scheduler_prefill(ss_scheduler_t *sched, const int32_t *tokens, int32_t num_tokens, int32_t start_pos) {
    if (!sched || sched->cancelled || num_tokens <= 0) return NULL;

    ss_gguf_file_t *gguf = sched->gguf;
    ss_layer_state_t *s = &sched->state;

    // Allocate batch buffers for hidden states
    float *batch_hidden = (float *)malloc((size_t)num_tokens * s->hidden_size * sizeof(float));
    if (!batch_hidden) return NULL;

    // ── Embedding lookup for all tokens ──
    const ss_tensor_info_t *embed = find_global_tensor(gguf, "token_embd.weight");
    if (!embed) { free(batch_hidden); return NULL; }
    const void *embed_data = ss_gguf_tensor_data(gguf, embed);
    if (!embed_data) { free(batch_hidden); return NULL; }

    for (int32_t i = 0; i < num_tokens; i++) {
        int32_t token_id = tokens[i];
        float *hidden_i = batch_hidden + (size_t)i * s->hidden_size;

        if (embed->type == GGML_TYPE_F32) {
            const float *emb = (const float *)embed_data + (size_t)token_id * s->hidden_size;
            memcpy(hidden_i, emb, s->hidden_size * sizeof(float));
        } else {
            size_t type_sz = ss_ggml_type_size(embed->type);
            uint32_t block_sz = ss_ggml_block_size(embed->type);
            size_t row_bytes = ((size_t)s->hidden_size / block_sz) * type_sz;
            const uint8_t *row = (const uint8_t *)embed_data + (size_t)token_id * row_bytes;
            ss_dequantize(row, hidden_i, s->hidden_size, embed->type);
        }

        if (sched->is_gemma) {
            float sq = sqrtf((float)s->hidden_size);
            for (int32_t j = 0; j < s->hidden_size; j++) {
                hidden_i[j] *= sq;
            }
        }
    }

    // ── Process all layers (THE INVERTED STREAMING LOOP) ──
    for (int32_t l = 0; l < sched->num_layers; l++) {
        if (sched->cancelled) { free(batch_hidden); return NULL; }

        // Prefetch next 2 layers asynchronously
        if (sched->use_prefetch) {
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                if (l + 1 < sched->num_layers) ss_gguf_prefetch_layer(gguf, (uint32_t)(l + 1));
                if (l + 2 < sched->num_layers) ss_gguf_prefetch_layer(gguf, (uint32_t)(l + 2));
            });
        }

        // TurboQuant removes LZ4 un-compression step natively.
        // Process this layer purely for prompt embedding ingestion
        for (int32_t i = 0; i < num_tokens; i++) {
            // Copy batch item into the normal scheduler hidden state
            memcpy(s->hidden, batch_hidden + (size_t)i * s->hidden_size, s->hidden_size * sizeof(float));
            
            // Process layer (caches weights internally or uses already-loaded weights!)
            process_layer(sched, l, start_pos + i);
            
            // Extract the mutated hidden state back into the batch
            memcpy(batch_hidden + (size_t)i * s->hidden_size, s->hidden, s->hidden_size * sizeof(float));
        }

        // Evict this layer's weights from page cache AFTER all tokens have passed
        ss_gguf_evict_layer(gguf, (uint32_t)l);

        // Progress callback per layer during prefill
        if (sched->progress_cb) {
            char phase[64];
            snprintf(phase, sizeof(phase), "layer %d/%d", l + 1, sched->num_layers);
            sched->progress_cb((uint32_t)(l + 1), (uint32_t)sched->num_layers,
                               phase, sched->progress_user_data);
        }
    }

    // ── Final RMS Norm (Only needed for the LAST token since we only need the next prediction) ──
    const ss_tensor_info_t *final_norm = find_global_tensor(gguf, "output_norm.weight");
    if (final_norm) {
        const float *norm_w = (const float *)ss_gguf_tensor_data(gguf, final_norm);
        if (norm_w) {
            // Apply RMS to the LAST token in the batch
            float *last_hidden = batch_hidden + (size_t)(num_tokens - 1) * s->hidden_size;
            
            if (sched->is_gemma) {
                ss_gemma_rmsnorm(last_hidden, last_hidden, norm_w, s->hidden_size, sched->rms_norm_eps, final_norm->type);
            } else {
                ss_rmsnorm(last_hidden, last_hidden, norm_w, s->hidden_size, sched->rms_norm_eps, final_norm->type);
            }
            
            // Store it into the scheduler's hidden state for LM head
            memcpy(s->hidden, last_hidden, s->hidden_size * sizeof(float));
        }
    }

    free(batch_hidden);

    // ── Output head (LM head) for the LAST token ──
    const ss_tensor_info_t *lm_head = find_global_tensor(gguf, "output.weight");
    if (!lm_head) {
        lm_head = embed; // tie embeddings
    }

    if (lm_head) {
        const void *head_data = ss_gguf_tensor_data(gguf, lm_head);
        if (head_data) {
            ss_matmul_dispatch(sched, s->logits, head_data, s->hidden, s->vocab_size, s->hidden_size, lm_head->type);
        }
    }

    if (sched->is_gemma) {
        float inv_sq = 1.0f / sqrtf((float)s->hidden_size);
        for (int32_t i = 0; i < s->vocab_size; i++) {
            s->logits[i] *= inv_sq;
        }
    }

    if (gguf->final_logit_softcapping > 0.0f) {
        ss_softcap(s->logits, s->vocab_size, gguf->final_logit_softcapping);
    }

    return s->logits;
}

float *ss_scheduler_embed(ss_scheduler_t *sched, const int32_t *tokens, int32_t num_tokens) {
    if (!sched || sched->cancelled || num_tokens <= 0) return NULL;

    ss_gguf_file_t *gguf = sched->gguf;
    ss_layer_state_t *s = &sched->state;

    // Allocate batch buffers for hidden states
    float *batch_hidden = (float *)malloc((size_t)num_tokens * s->hidden_size * sizeof(float));
    if (!batch_hidden) return NULL;

    // ── Embedding lookup for all tokens ──
    const ss_tensor_info_t *embed = find_global_tensor(gguf, "token_embd.weight");
    if (!embed) { free(batch_hidden); return NULL; }
    const void *embed_data = ss_gguf_tensor_data(gguf, embed);
    if (!embed_data) { free(batch_hidden); return NULL; }

    for (int32_t i = 0; i < num_tokens; i++) {
        int32_t token_id = tokens[i];
        float *hidden_i = batch_hidden + (size_t)i * s->hidden_size;

        if (embed->type == GGML_TYPE_F32) {
            const float *emb = (const float *)embed_data + (size_t)token_id * s->hidden_size;
            memcpy(hidden_i, emb, s->hidden_size * sizeof(float));
        } else {
            size_t type_sz = ss_ggml_type_size(embed->type);
            uint32_t block_sz = ss_ggml_block_size(embed->type);
            size_t row_bytes = ((size_t)s->hidden_size / block_sz) * type_sz;
            const uint8_t *row = (const uint8_t *)embed_data + (size_t)token_id * row_bytes;
            ss_dequantize(row, hidden_i, s->hidden_size, embed->type);
        }

        if (sched->is_gemma) {
            float sq = sqrtf((float)s->hidden_size);
            for (int32_t j = 0; j < s->hidden_size; j++) {
                hidden_i[j] *= sq;
            }
        }
    }

    // ── Process all layers (Streaming sequentially across entire sequence) ──
    for (int32_t l = 0; l < sched->num_layers; l++) {
        if (sched->cancelled) { free(batch_hidden); return NULL; }

        if (sched->use_prefetch) {
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                if (l + 1 < sched->num_layers) ss_gguf_prefetch_layer(gguf, (uint32_t)(l + 1));
                if (l + 2 < sched->num_layers) ss_gguf_prefetch_layer(gguf, (uint32_t)(l + 2));
            });
        }
        // TurboQuant removes LZ4 un-compression step natively.
        for (int32_t i = 0; i < num_tokens; i++) {
            memcpy(s->hidden, batch_hidden + (size_t)i * s->hidden_size, s->hidden_size * sizeof(float));
            process_layer(sched, l, i);
            memcpy(batch_hidden + (size_t)i * s->hidden_size, s->hidden, s->hidden_size * sizeof(float));
        }

        ss_gguf_evict_layer(gguf, (uint32_t)l);

        if (sched->progress_cb) {
            char phase[64];
            snprintf(phase, sizeof(phase), "layer %d/%d", l + 1, sched->num_layers);
            sched->progress_cb((uint32_t)(l + 1), (uint32_t)sched->num_layers,
                               phase, sched->progress_user_data);
        }
    }

    // ── Final RMS Norm (Applied universally to ALL tokens for Embeddings) ──
    const ss_tensor_info_t *final_norm = find_global_tensor(gguf, "output_norm.weight");
    if (final_norm) {
        const float *norm_w = (const float *)ss_gguf_tensor_data(gguf, final_norm);
        if (norm_w) {
            for (int32_t i = 0; i < num_tokens; i++) {
                float *hidden_i = batch_hidden + (size_t)i * s->hidden_size;
                if (sched->is_gemma) {
                    ss_gemma_rmsnorm(hidden_i, hidden_i, norm_w, s->hidden_size, sched->rms_norm_eps, final_norm->type);
                } else {
                    ss_rmsnorm(hidden_i, hidden_i, norm_w, s->hidden_size, sched->rms_norm_eps, final_norm->type);
                }
            }
        }
    }

    // ── Calculate Mean Pooling Embedding Vector ──
    float *embedding = (float *)malloc(s->hidden_size * sizeof(float));
    if (!embedding) { free(batch_hidden); return NULL; }
    memset(embedding, 0, s->hidden_size * sizeof(float));

    for (int32_t i = 0; i < num_tokens; i++) {
        float *hidden_i = batch_hidden + (size_t)i * s->hidden_size;
        for (int32_t j = 0; j < s->hidden_size; j++) {
            embedding[j] += hidden_i[j];
        }
    }

    // Average scaling and L2 Normalization sequence
    float inv_tokens = 1.0f / (float)num_tokens;
    float l2_norm_sq = 0.0f;
    for (int32_t j = 0; j < s->hidden_size; j++) {
        embedding[j] *= inv_tokens;
        l2_norm_sq += embedding[j] * embedding[j];
    }
    
    // Normalize mapping bounding into a 1-length unit sphere dynamically
    float l2_scale = 1.0f / (sqrtf(l2_norm_sq) + 1e-12f);
    for (int32_t j = 0; j < s->hidden_size; j++) {
        embedding[j] *= l2_scale;
    }

    free(batch_hidden);
    return embedding;
}

void ss_scheduler_reset(ss_scheduler_t *sched) {
    if (!sched) return;

    // Clear TurboQuant KV caches!
    size_t kv_total = (size_t)sched->kv_cache.num_layers *
                      sched->kv_cache.max_seq_len *
                      sched->kv_cache.kv_dim;
    memset(sched->kv_cache.q_key_cache, 0, kv_total * sizeof(int8_t));
    memset(sched->kv_cache.q_val_cache, 0, kv_total * sizeof(int8_t));
    
    size_t scale_total = (size_t)sched->kv_cache.num_layers * sched->kv_cache.max_seq_len;
    memset(sched->kv_cache.k_scales, 0, scale_total * sizeof(float));
    memset(sched->kv_cache.v_scales, 0, scale_total * sizeof(float));
    
    sched->kv_cache.seq_len = 0;

    // Reset state
    memset(sched->state.hidden, 0, sched->state.hidden_size * sizeof(float));
    sched->cancelled = false;
}

void ss_scheduler_free(ss_scheduler_t *sched) {
    if (!sched) return;

    kv_cache_free(&sched->kv_cache);

    ss_layer_state_t *s = &sched->state;
    free(s->hidden);
    if (s->attn_out) free(s->attn_out);
    if (s->ffn_buf) free(s->ffn_buf);
    if (s->ffn_gate) free(s->ffn_gate);
    if (s->ffn_up_buf) free(s->ffn_up_buf);
    if (s->q_buf) free(s->q_buf);
    free(s->residual);
    free(s->k_buf);
    free(s->v_buf);
    free(s->qkv_buf);
    free(s->head_out);
    free(s->logits);
    if (s->scores_buf) free(s->scores_buf);

    free(sched);
}
