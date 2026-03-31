/**
 * SlipStream — Metal GPU Compute Backend
 *
 * Uses Apple Metal for GPU-accelerated matrix multiplication on iOS/macOS.
 */

#ifndef SS_METAL_BACKEND_H
#define SS_METAL_BACKEND_H

#include <stdint.h>
#include <stdbool.h>

#ifdef SS_HAS_METAL

typedef struct ss_metal_ctx ss_metal_ctx_t;

/**
 * Initialize Metal compute context.
 * Returns NULL if Metal is not available.
 */
ss_metal_ctx_t *ss_metal_init(void);

/**
 * GPU matrix-vector multiply: out = mat @ vec
 * Uploads data, dispatches compute, reads back result.
 */
bool ss_metal_matvec(ss_metal_ctx_t *ctx,
                     float *out,
                     const float *mat,
                     const float *vec,
                     int32_t rows,
                     int32_t cols);

/**
 * GPU matrix-vector multiply for Q4_0 quantized matrices.
 */
bool ss_metal_matvec_q4_0(ss_metal_ctx_t *ctx,
                          float *out,
                          const void *mat_q4_0,
                          const float *vec,
                          int32_t rows,
                          int32_t cols);

/**
 * GPU matrix-vector multiply for Q4_K quantized matrices.
 */
bool ss_metal_matvec_q4_K(ss_metal_ctx_t *ctx,
                          float *out,
                          const void *mat_q4_K,
                          const float *vec,
                          int32_t rows,
                          int32_t cols);

/**
 * Check if input dimensions are worth GPU dispatch.
 * Small operations should stay on CPU due to transfer overhead.
 */
bool ss_metal_should_use_gpu(int32_t rows, int32_t cols);

/**
 * Begin a Command Buffer parallel batch. Subsequent matrix multiplications
 * will be enqueued into the unified buffer, deferring XPC synchronization.
 */
void ss_metal_begin_batch(ss_metal_ctx_t *ctx);

/**
 * End a Command Buffer batch, synchronously flushing all parallel pipelines 
 * onto the GPU compute grids and waiting for their unified completion.
 */
void ss_metal_end_batch(ss_metal_ctx_t *ctx);

/**
 * Free Metal context.
 */
void ss_metal_free(ss_metal_ctx_t *ctx);

#endif // SS_HAS_METAL

#endif // SS_METAL_BACKEND_H
