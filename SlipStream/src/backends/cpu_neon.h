/**
 * SlipStream — ARM NEON Compute Backend
 *
 * SIMD-accelerated tensor operations for ARM processors
 * (all modern iPhones, iPads, and Android devices).
 */

#ifndef SS_CPU_NEON_H
#define SS_CPU_NEON_H

#include <stdint.h>

#ifdef SS_HAS_NEON
#include <arm_neon.h>
#endif

/**
 * NEON-accelerated dot product of two float vectors.
 */
float ss_neon_dot(const float *a, const float *b, int32_t n);

/**
 * NEON-accelerated matrix-vector multiply: out = mat @ vec
 */
void ss_neon_matvec(float *out, const float *mat, const float *vec,
                    int32_t rows, int32_t cols);

/**
 * NEON-accelerated RMS normalization.
 */
void ss_neon_rmsnorm(float *out, const float *x, const float *weight,
                     int32_t size, float eps);

/**
 * NEON-accelerated Gemma offset RMS normalization.
 */
void ss_neon_gemma_rmsnorm(float *out, const float *x, const float *weight,
                           int32_t size, float eps);

/**
 * NEON-accelerated element-wise multiply.
 */
void ss_neon_elementwise_mul(float *out, const float *a, const float *b, int32_t n);

/**
 * NEON-accelerated SiLU activation.
 */
void ss_neon_silu(float *x, int32_t n);

#endif // SS_CPU_NEON_H
