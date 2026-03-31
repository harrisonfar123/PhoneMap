/**
 * SlipStream — ARM NEON Compute Backend Implementation
 *
 * SIMD-accelerated versions of core tensor operations.
 * Falls back to scalar when NEON is not available.
 */

#include "backends/cpu_neon.h"

#include <math.h>
#include <string.h>

// ─── NEON Dot Product ────────────────────────────────────────────────────────

float ss_neon_dot(const float *a, const float *b, int32_t n) {
#ifdef SS_HAS_NEON
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int32_t i = 0;

    // Process 16 elements per iteration (4 NEON lanes × 4 unrolled)
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        sum_vec = vfmaq_f32(sum_vec, a0, b0);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        sum_vec = vfmaq_f32(sum_vec, a1, b1);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        sum_vec = vfmaq_f32(sum_vec, a2, b2);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        sum_vec = vfmaq_f32(sum_vec, a3, b3);
    }

    // Process remaining 4-element blocks
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        sum_vec = vfmaq_f32(sum_vec, av, bv);
    }

    // Horizontal sum
    float sum = vaddvq_f32(sum_vec);

    // Scalar remainder
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
#else
    // Scalar fallback
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// ─── NEON Matrix-Vector Multiply ────────────────────────────────────────────

void ss_neon_matvec(float *out, const float *mat, const float *vec,
                    int32_t rows, int32_t cols) {
    for (int32_t i = 0; i < rows; i++) {
        out[i] = ss_neon_dot(mat + i * cols, vec, cols);
    }
}

// ─── NEON RMS Normalization ─────────────────────────────────────────────────

void ss_neon_rmsnorm(float *out, const float *x, const float *weight,
                     int32_t size, float eps) {
#ifdef SS_HAS_NEON
    // Compute sum of squares
    float32x4_t ss_vec = vdupq_n_f32(0.0f);
    int32_t i = 0;

    for (; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        ss_vec = vfmaq_f32(ss_vec, xv, xv);
    }

    float ss = vaddvq_f32(ss_vec);
    for (; i < size; i++) {
        ss += x[i] * x[i];
    }

    float scale = 1.0f / sqrtf(ss / (float)size + eps);

    // Normalize and apply weight
    float32x4_t scale_vec = vdupq_n_f32(scale);
    i = 0;

    for (; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(weight + i);
        float32x4_t result = vmulq_f32(vmulq_f32(xv, scale_vec), wv);
        vst1q_f32(out + i, result);
    }

    for (; i < size; i++) {
        out[i] = x[i] * scale * weight[i];
    }
#else
    // Scalar fallback
    float ss = 0.0f;
    for (int32_t i = 0; i < size; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)size + eps);
    for (int32_t i = 0; i < size; i++) out[i] = x[i] * scale * weight[i];
#endif
}

// ─── NEON Gemma RMS Normalization ────────────────────────────────────────────

void ss_neon_gemma_rmsnorm(float *out, const float *x, const float *weight, int32_t size, float eps) {
#ifdef SS_HAS_NEON
    float32x4_t ss_vec = vdupq_n_f32(0.0f);
    int32_t i = 0;

    for (; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        ss_vec = vfmaq_f32(ss_vec, xv, xv);
    }

    float ss = vaddvq_f32(ss_vec);
    for (; i < size; i++) {
        ss += x[i] * x[i];
    }

    float scale = 1.0f / sqrtf(ss / (float)size + eps);

    // Normalize and apply weight + 1.0f
    float32x4_t scale_vec = vdupq_n_f32(scale);
    float32x4_t one_vec = vdupq_n_f32(1.0f);
    i = 0;

    for (; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(weight + i);
        wv = vaddq_f32(wv, one_vec);
        float32x4_t result = vmulq_f32(vmulq_f32(xv, scale_vec), wv);
        vst1q_f32(out + i, result);
    }

    for (; i < size; i++) {
        out[i] = x[i] * scale * (weight[i] + 1.0f);
    }
#else
    // Scalar fallback
    float ss = 0.0f;
    for (int32_t i = 0; i < size; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)size + eps);
    for (int32_t i = 0; i < size; i++) out[i] = x[i] * scale * (weight[i] + 1.0f);
#endif
}

// ─── NEON Element-wise Multiply ─────────────────────────────────────────────

void ss_neon_elementwise_mul(float *out, const float *a, const float *b, int32_t n) {
#ifdef SS_HAS_NEON
    int32_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(av, bv));
    }
    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
#else
    for (int32_t i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

// ─── NEON SiLU ──────────────────────────────────────────────────────────────

void ss_neon_silu(float *x, int32_t n) {
    // SiLU doesn't vectorize well with NEON (no native exp),
    // so we use scalar with loop unrolling
    int32_t i = 0;
    for (; i + 3 < n; i += 4) {
        x[i]     = x[i]     / (1.0f + expf(-x[i]));
        x[i + 1] = x[i + 1] / (1.0f + expf(-x[i + 1]));
        x[i + 2] = x[i + 2] / (1.0f + expf(-x[i + 2]));
        x[i + 3] = x[i + 3] / (1.0f + expf(-x[i + 3]));
    }
    for (; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}
