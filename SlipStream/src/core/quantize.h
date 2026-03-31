/**
 * SlipStream — Quantization / Dequantization
 */

#ifndef SS_QUANTIZE_H
#define SS_QUANTIZE_H

#include "core/gguf_loader.h"
#include <stdint.h>

// ─── Dequantization Block Structures ─────────────────────────────────────────

// Q4_0: 32 x 4-bit values + f16 scale
typedef struct {
    uint16_t scale;     // f16 scale factor
    uint8_t  qs[16];    // 32 x 4-bit quantized values (packed)
} ss_block_q4_0_t;

// Q4_1: 32 x 4-bit values + f16 scale + f16 min (asymmetric)
typedef struct {
    uint16_t scale;
    uint16_t min;
    uint8_t  qs[16];
} ss_block_q4_1_t;

// Q5_0: 32 x 5-bit values + f16 scale (symmetric)
// 5th bit stored separately in qh
typedef struct {
    uint16_t scale;       // f16 scale factor
    uint8_t  qh[4];       // 32 high bits (5th bit per value)
    uint8_t  qs[16];      // 32 x 4-bit low nibbles (packed)
} ss_block_q5_0_t;

// Q5_1: 32 x 5-bit values + f16 scale + f16 min (asymmetric)
typedef struct {
    uint16_t d;           // f16 scale
    uint16_t m;           // f16 min
    uint8_t  qh[4];       // 32 high bits (5th bit per value)
    uint8_t  qs[16];      // 32 x 4-bit low nibbles (packed)
} ss_block_q5_1_t;

// Q8_0: 32 x 8-bit values + f16 scale
typedef struct {
    uint16_t scale;
    int8_t   qs[32];
} ss_block_q8_0_t;

// Q8_1: 32 x 8-bit values + f32 scale + f32 sum (intermediate quant)
typedef struct {
    float   d;            // scale
    float   s;            // sum (not used for dequant)
    int8_t  qs[32];
} ss_block_q8_1_t;

// ─── K-Quant Block Structures (super-blocks of 256 values) ──────────────────

#define SS_QK_K 256
#define K_SCALE_SIZE 12

// Q2_K: 256 x 2-bit values with 4-bit sub-block scales/mins
typedef struct {
    uint8_t  scales[SS_QK_K/16]; // 16 bytes: 4-bit scale + 4-bit min per 16 values
    uint8_t  qs[SS_QK_K/4];     // 64 bytes: 2-bit quants packed 4 per byte
    uint16_t d;                  // f16 super-block scale
    uint16_t dmin;               // f16 super-block min
} ss_block_q2_K_t;               // 84 bytes

// Q3_K: 256 x 3-bit values with packed scales
typedef struct {
    uint8_t  hmask[SS_QK_K/8];  // 32 bytes: high bits (3rd bit)
    uint8_t  qs[SS_QK_K/4];    // 64 bytes: low 2 bits packed
    uint8_t  scales[12];       // 12 bytes: packed 6-bit scales
    uint16_t d;                // 2 bytes: f16 scale
} ss_block_q3_K_t;              // 110 bytes

// Q4_K: 256 x 4-bit values with 6-bit sub-block scales/mins
typedef struct {
    uint16_t d;                     // f16 super-block scale
    uint16_t dmin;                  // f16 super-block min
    uint8_t  scales[K_SCALE_SIZE];  // 12 bytes: packed 6-bit scales/mins
    uint8_t  qs[SS_QK_K/2];        // 128 bytes: 4-bit quants
} ss_block_q4_K_t;                  // 144 bytes

// Q5_K: 256 x 5-bit values with 6-bit sub-block scales/mins
typedef struct {
    uint16_t d;                     // f16 super-block scale
    uint16_t dmin;                  // f16 super-block min
    uint8_t  scales[K_SCALE_SIZE];  // 12 bytes: packed 6-bit scales/mins
    uint8_t  qh[SS_QK_K/8];        // 32 bytes: high bits (5th bit)
    uint8_t  qs[SS_QK_K/2];        // 128 bytes: low 4-bit quants
} ss_block_q5_K_t;                  // 176 bytes

// Q6_K: 256 x 6-bit values with 8-bit sub-block scales
typedef struct {
    uint8_t  ql[SS_QK_K/2];         // 128 bytes: low 4 bits
    uint8_t  qh[SS_QK_K/4];         // 64 bytes: high 2 bits
    int8_t   scales[SS_QK_K/16];    // 16 bytes: 8-bit scales
    uint16_t d;                     // 2 bytes: f16 scale
} ss_block_q6_K_t;                  // 210 bytes

// ─── API ─────────────────────────────────────────────────────────────────────

/**
 * Dequantize a buffer of quantized data to float32.
 */
void ss_dequantize(const void *src, float *dst, int32_t n, ggml_type_t type);

/**
 * Dequantize and multiply in one pass (fused dequant + matvec).
 */
void ss_dequant_matvec(float *out, const void *weight, const float *x,
                       int32_t rows, int32_t cols, ggml_type_t type);

/**
 * Convert f16 to f32.
 */
float ss_f16_to_f32(uint16_t h);

#endif // SS_QUANTIZE_H
