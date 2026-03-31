/**
 * SlipStream — Tensor Operations
 *
 * Core math operations for transformer inference,
 * dispatching to NEON or Metal backends.
 */

#ifndef SS_TENSOR_H
#define SS_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ─── Tensor Struct ───────────────────────────────────────────────────────────

typedef struct {
    float   *data;
    int32_t  dims[4];
    int32_t  n_dims;
    size_t   size;      // Total number of elements
} ss_tensor_t;

// ─── Lifecycle ───────────────────────────────────────────────────────────────

ss_tensor_t *ss_tensor_alloc(int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3);
ss_tensor_t *ss_tensor_alloc_1d(int32_t size);
ss_tensor_t *ss_tensor_alloc_2d(int32_t rows, int32_t cols);
void          ss_tensor_free(ss_tensor_t *t);
ss_tensor_t *ss_tensor_clone(const ss_tensor_t *src);

// ─── Core Operations ────────────────────────────────────────────────────────

/**
 * General matrix multiply: C = A @ B^T
 * A: [M x K], B: [N x K], C: [M x N]
 */
void ss_matmul(float *out, const float *a, const float *b,
               int32_t M, int32_t K, int32_t N);

/**
 * Matrix-vector multiply: out = mat @ vec
 * mat: [rows x cols], vec: [cols], out: [rows]
 */
void ss_matvec(float *out, const float *mat, const float *vec,
               int32_t rows, int32_t cols);

/**
 * RMS Normalization: out = x * (1/rms(x)) * weight
 */
void ss_rmsnorm(float *out, const float *x, const void *weight,
                int32_t size, float eps, int32_t wtype);

/**
 * Gemma offset RMS Normalization: out = (x / rms(x)) * (weight + 1.0f)
 */
void ss_gemma_rmsnorm(float *out, const float *x, const void *weight,
                      int32_t size, float eps, int32_t wtype);

/**
 * Softmax in-place: x[i] = exp(x[i]) / sum(exp(x))
 */
void ss_softmax(float *x, int32_t size);

/**
 * SiLU activation in-place: x = x * sigmoid(x)
 */
void ss_silu(float *x, int32_t size);

/**
 * GeGLU activation in-place (Gemma/Gemma 2)
 */
void ss_geglu(float *x, int32_t size);

/**
 * Tanh logit soft-capping (Gemma 2)
 */
void ss_softcap(float *x, int32_t size, float cap);

/**
 * Element-wise multiply: out = a * b
 */
void ss_elementwise_mul(float *out, const float *a, const float *b, int32_t size);

/**
 * Element-wise add: out = a + b
 */
void ss_elementwise_add(float *out, const float *a, const float *b, int32_t size);

/**
 * Apply RoPE (Rotary Position Embeddings)
 * q: [num_heads, head_dim], k: [num_kv_heads, head_dim]
 */
void ss_rope(float *q, float *k,
             int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
             int32_t rope_dim, int32_t pos, float theta, bool is_neox);

/**
 * Scaled dot-product attention for a single head.
 * q: [head_dim], keys: [seq_len, head_dim], values: [seq_len, head_dim]
 * out: [head_dim]
 */
void ss_attention(float *out,
                  const float *q,
                  const float *keys,
                  const float *values,
                  int32_t head_dim,
                  int32_t seq_len);

// ─── Sampling ────────────────────────────────────────────────────────────────

/**
 * Argmax: return index of largest element.
 */
int32_t ss_argmax(const float *x, int32_t size);

/**
 * Top-k + top-p sampling with temperature.
 * Returns sampled token index.
 */
int32_t ss_sample(const float *logits, int32_t vocab_size,
                  float temperature, float top_p, int32_t top_k,
                  uint64_t *rng_state);

#endif // SS_TENSOR_H
