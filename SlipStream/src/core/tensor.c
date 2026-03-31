/**
 * SlipStream — Tensor Operations Implementation
 *
 * Scalar reference implementations of all tensor ops.
 * NEON-accelerated versions are in backends/cpu_neon.c and override these.
 */

#include "core/tensor.h"
#include "backends/cpu_neon.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
#include <float.h>

// ─── Lifecycle ───────────────────────────────────────────────────────────────

ss_tensor_t *ss_tensor_alloc(int32_t d0, int32_t d1, int32_t d2, int32_t d3) {
    ss_tensor_t *t = (ss_tensor_t *)calloc(1, sizeof(ss_tensor_t));
    if (!t) return NULL;

    t->dims[0] = d0;
    t->dims[1] = d1 > 0 ? d1 : 1;
    t->dims[2] = d2 > 0 ? d2 : 1;
    t->dims[3] = d3 > 0 ? d3 : 1;
    t->n_dims = (d3 > 0) ? 4 : (d2 > 0) ? 3 : (d1 > 0) ? 2 : 1;

    t->size = (size_t)t->dims[0] * t->dims[1] * t->dims[2] * t->dims[3];

    // Aligned allocation for SIMD
    t->data = (float *)calloc(t->size, sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }

    return t;
}

ss_tensor_t *ss_tensor_alloc_1d(int32_t size) {
    return ss_tensor_alloc(size, 0, 0, 0);
}

ss_tensor_t *ss_tensor_alloc_2d(int32_t rows, int32_t cols) {
    return ss_tensor_alloc(rows, cols, 0, 0);
}

void ss_tensor_free(ss_tensor_t *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

ss_tensor_t *ss_tensor_clone(const ss_tensor_t *src) {
    if (!src) return NULL;
    ss_tensor_t *dst = ss_tensor_alloc(src->dims[0], src->dims[1],
                                        src->dims[2], src->dims[3]);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->size * sizeof(float));
    return dst;
}

// ─── Matrix Multiply ────────────────────────────────────────────────────────

// C = A @ B^T  where A=[M,K], B=[N,K], C=[M,N]
void ss_matmul(float *out, const float *a, const float *b,
               int32_t M, int32_t K, int32_t N) {
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            const float *ai = a + i * K;
            const float *bj = b + j * K;
            for (int32_t k = 0; k < K; k++) {
                sum += ai[k] * bj[k];
            }
            out[i * N + j] = sum;
        }
    }
}

void ss_matvec(float *out, const float *mat, const float *vec,
               int32_t rows, int32_t cols) {
    ss_neon_matvec(out, mat, vec, rows, cols);
}

// ─── RMS Normalization ──────────────────────────────────────────────────────

static inline float ss_f16_to_f32_local(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h & 0x7C00) >> 10;
    uint32_t m = (h & 0x03FF);
    if (e == 0) {
        if (m == 0) return *(float*)&s;
        while ((m & 0x0400) == 0) {
            m <<= 1;
            e--;
        }
        e++;
        m &= ~0x0400;
    } else if (e == 31) {
        if (m == 0) {
            uint32_t inf = s | 0x7F800000;
            return *(float*)&inf;
        } else {
            uint32_t nan = s | 0x7F800000 | (m << 13);
            return *(float*)&nan;
        }
    }
    e = e + (127 - 15);
    uint32_t out = s | (e << 23) | (m << 13);
    return *(float*)&out;
}

void ss_rmsnorm(float *out, const float *x, const void *weight, int32_t size, float eps, int32_t wtype) {
    if (wtype == 0 /* GGML_TYPE_F32 */) {
        ss_neon_rmsnorm(out, x, (const float *)weight, size, eps);
        return;
    }

    // F16 Fallback
    float ss = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / (float)size + eps);

    for (int32_t i = 0; i < size; i++) {
        float w = ss_f16_to_f32_local(((const uint16_t *)weight)[i]);
        out[i] = x[i] * ss * w;
    }
}

void ss_gemma_rmsnorm(float *out, const float *x, const void *weight, int32_t size, float eps, int32_t wtype) {
    if (wtype == 0 /* GGML_TYPE_F32 */) {
        ss_neon_gemma_rmsnorm(out, x, (const float *)weight, size, eps);
        return;
    }

    // F16 Fallback (Gemma adds 1.0f to weights)
    float ss = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / (float)size + eps);

    for (int32_t i = 0; i < size; i++) {
        float w = ss_f16_to_f32_local(((const uint16_t *)weight)[i]);
        out[i] = x[i] * ss * (w + 1.0f);
    }
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

void ss_softmax(float *x, int32_t size) {
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int32_t i = 0; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // exp and sum
    float sum = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    float inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// ─── SiLU & GeGLU ────────────────────────────────────────────────────────────

void ss_silu(float *x, int32_t size) {
    ss_neon_silu(x, size);
}

void ss_geglu(float *x, int32_t size) {
    // x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    for (int32_t i = 0; i < size; i++) {
        float v = x[i];
        float cube = v * v * v;
        float inner = sqrt_2_over_pi * (v + 0.044715f * cube);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// ─── Softcapping ─────────────────────────────────────────────────────────────

void ss_softcap(float *x, int32_t size, float cap) {
    if (cap <= 0.0f) return;
    float inv_cap = 1.0f / cap;
    for (int32_t i = 0; i < size; i++) {
        x[i] = cap * tanhf(x[i] * inv_cap);
    }
}

// ─── Element-wise ────────────────────────────────────────────────────────────

void ss_elementwise_mul(float *out, const float *a, const float *b, int32_t size) {
    ss_neon_elementwise_mul(out, a, b, size);
}

void ss_elementwise_add(float *out, const float *a, const float *b, int32_t size) {
#ifdef __APPLE__
    vDSP_vadd(a, 1, b, 1, out, 1, size);
#else
    for (int32_t i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
#endif
}

// ─── RoPE ────────────────────────────────────────────────────────────────────

void ss_rope(float *q, float *k,
             int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
             int32_t rope_dim, int32_t pos, float theta, bool is_neox) {
    if (is_neox) {
        int32_t half_d = rope_dim / 2;
        for (int32_t i = 0; i < half_d; i++) {
            float freq = 1.0f / powf(theta, (float)(i * 2) / (float)rope_dim);
            float angle = (float)pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            // Apply to Q heads
            for (int32_t h = 0; h < num_heads; h++) {
                float *qh = q + h * head_dim;
                float q0 = qh[i];
                float q1 = qh[i + half_d];
                qh[i]          = q0 * cos_val - q1 * sin_val;
                qh[i + half_d] = q0 * sin_val + q1 * cos_val;
            }

            // Apply to K heads
            for (int32_t h = 0; h < num_kv_heads; h++) {
                float *kh = k + h * head_dim;
                float k0 = kh[i];
                float k1 = kh[i + half_d];
                kh[i]          = k0 * cos_val - k1 * sin_val;
                kh[i + half_d] = k0 * sin_val + k1 * cos_val;
            }
        }
    } else {
        for (int32_t i = 0; i < rope_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)rope_dim);
            float angle = (float)pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            // Apply to Q heads
            for (int32_t h = 0; h < num_heads; h++) {
                float *qh = q + h * head_dim;
                float q0 = qh[i];
                float q1 = qh[i + 1];
                qh[i]     = q0 * cos_val - q1 * sin_val;
                qh[i + 1] = q0 * sin_val + q1 * cos_val;
            }

            // Apply to K heads
            for (int32_t h = 0; h < num_kv_heads; h++) {
                float *kh = k + h * head_dim;
                float k0 = kh[i];
                float k1 = kh[i + 1];
                kh[i]     = k0 * cos_val - k1 * sin_val;
                kh[i + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

void ss_attention(float *out,
                  const float *q,
                  const float *keys,
                  const float *values,
                  int32_t head_dim,
                  int32_t seq_len) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Allocate attention scores
    float *scores = (float *)malloc(seq_len * sizeof(float));
    if (!scores) return;

    // Q @ K^T
    for (int32_t t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int32_t d = 0; d < head_dim; d++) {
            dot += q[d] * keys[t * head_dim + d];
        }
        scores[t] = dot * scale;
    }

    // Softmax
    ss_softmax(scores, seq_len);

    // Weighted sum of values
    memset(out, 0, head_dim * sizeof(float));
    for (int32_t t = 0; t < seq_len; t++) {
        float w = scores[t];
        const float *v = values + t * head_dim;
        for (int32_t d = 0; d < head_dim; d++) {
            out[d] += w * v[d];
        }
    }

    free(scores);
}

// ─── Sampling ────────────────────────────────────────────────────────────────

int32_t ss_argmax(const float *x, int32_t size) {
    int32_t max_idx = 0;
    float max_val = x[0];
    for (int32_t i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Simple xorshift64 RNG
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float rand_float(uint64_t *state) {
    return (float)(xorshift64(state) >> 11) / (float)(1ULL << 53);
}

// Struct for sorting tokens by probability
typedef struct {
    int32_t id;
    float prob;
} ss_token_prob_t;

static int compare_probs(const void *a, const void *b) {
    float pa = ((ss_token_prob_t *)a)->prob;
    float pb = ((ss_token_prob_t *)b)->prob;
    
    // Protect against NaN corrupting qsort strict weak ordering
    if (isnan(pa)) pa = 0.0f;
    if (isnan(pb)) pb = 0.0f;
    
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

int32_t ss_sample(const float *logits, int32_t vocab_size,
                  float temperature, float top_p, int32_t top_k,
                  uint64_t *rng_state) {
    // Greedy
    if (temperature <= 0.0f) {
        return ss_argmax(logits, vocab_size);
    }

    // Apply temperature
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    if (!probs) return ss_argmax(logits, vocab_size);

    for (int32_t i = 0; i < vocab_size; i++) {
        if (isnan(logits[i])) {
            probs[i] = -FLT_MAX; // NaN prevention before softmax
        } else {
            probs[i] = logits[i] / temperature;
        }
    }
    ss_softmax(probs, vocab_size);

    // Prepare array for sorting if we need top_k or top_p
    ss_token_prob_t *sorted = (ss_token_prob_t *)malloc(vocab_size * sizeof(ss_token_prob_t));
    if (sorted) {
        for (int32_t i = 0; i < vocab_size; i++) {
            sorted[i].id = i;
            sorted[i].prob = isnan(probs[i]) ? 0.0f : probs[i]; // Secondary NaN safeguard
        }
        // Fast O(N log N) sort descending
        qsort(sorted, vocab_size, sizeof(ss_token_prob_t), compare_probs);
    }

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size && sorted) {
        float threshold = sorted[top_k - 1].prob;
        for (int32_t i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
    }

    // Top-p (nucleus) filtering
    if (top_p > 0.0f && top_p < 1.0f && sorted) {
        float cumsum = 0.0f;
        float cutoff = 0.0f;
        
        for (int32_t i = 0; i < vocab_size; i++) {
            // Only count non-zeroed probabilities (from top-k)
            float p_val = probs[sorted[i].id];
            if (p_val > 0.0f) {
                cumsum += p_val;
            }
            if (cumsum >= top_p) {
                cutoff = p_val;
                break;
            }
        }

        for (int32_t i = 0; i < vocab_size; i++) {
            if (probs[i] < cutoff) probs[i] = 0.0f;
        }
    }

    if (sorted) {
        free(sorted);
    }

    // Re-normalize
    float sum = 0.0f;
    for (int32_t i = 0; i < vocab_size; i++) sum += probs[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int32_t i = 0; i < vocab_size; i++) probs[i] *= inv;
    }

    // Sample from distribution
    float r = rand_float(rng_state);
    float cdf = 0.0f;
    int32_t sampled = vocab_size - 1;
    for (int32_t i = 0; i < vocab_size; i++) {
        cdf += probs[i];
        if (r < cdf) {
            sampled = i;
            break;
        }
    }

    free(probs);
    return sampled;
}
