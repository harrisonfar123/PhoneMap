/**
 * SlipStream — Quantization / Dequantization
 *
 * Full dequantization support for all GGML quantization types.
 * Ported from llama.cpp (ggml-quants.c).
 */

#include "core/quantize.h"
#include "core/gguf_loader.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>

// ─── F16 → F32 Conversion ──────────────────────────────────────────────────

float ss_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, sizeof(f));
            return f;
        }
        // Subnormal
        exponent = 1;
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
        exponent += 127 - 15;
        uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    } else if (exponent == 31) {
        uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    }

    exponent += 127 - 15;
    uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ─── Q4_0 Dequantization ────────────────────────────────────────────────────

static void dequant_q4_0(const void *src, float *dst, int32_t n) {
    const ss_block_q4_0_t *blocks = (const ss_block_q4_0_t *)src;
    int32_t num_blocks = n / 32;

    for (int32_t b = 0; b < num_blocks; b++) {
        float scale = ss_f16_to_f32(blocks[b].scale);
        for (int32_t i = 0; i < 16; i++) {
            uint8_t byte = blocks[b].qs[i];
            dst[b * 32 + i]      = scale * ((int32_t)(byte & 0xF) - 8);
            dst[b * 32 + i + 16] = scale * ((int32_t)(byte >> 4) - 8);
        }
    }
}

// ─── Q4_1 Dequantization ────────────────────────────────────────────────────

static void dequant_q4_1(const void *src, float *dst, int32_t n) {
    const ss_block_q4_1_t *blocks = (const ss_block_q4_1_t *)src;
    int32_t num_blocks = n / 32;

    for (int32_t b = 0; b < num_blocks; b++) {
        float scale = ss_f16_to_f32(blocks[b].scale);
        float min   = ss_f16_to_f32(blocks[b].min);
        for (int32_t i = 0; i < 16; i++) {
            uint8_t byte = blocks[b].qs[i];
            dst[b * 32 + i]      = scale * (byte & 0xF) + min;
            dst[b * 32 + i + 16] = scale * (byte >> 4) + min;
        }
    }
}

// ─── Q5_0 Dequantization ────────────────────────────────────────────────────

static void dequant_q5_0(const void *src, float *dst, int32_t n) {
    const ss_block_q5_0_t *blocks = (const ss_block_q5_0_t *)src;
    int32_t num_blocks = n / 32;

    for (int32_t b = 0; b < num_blocks; b++) {
        float scale = ss_f16_to_f32(blocks[b].scale);
        uint32_t qh;
        memcpy(&qh, blocks[b].qh, sizeof(qh));

        for (int32_t i = 0; i < 16; i++) {
            uint8_t byte = blocks[b].qs[i];
            const uint8_t xh_0 = ((qh >> (i +  0)) & 1) << 4;
            const uint8_t xh_1 = ((qh >> (i + 16)) & 1) << 4;
            const int32_t x0 = (int32_t)(byte & 0xF) | xh_0;
            const int32_t x1 = (int32_t)(byte >>  4) | xh_1;
            dst[b * 32 + i]      = (x0 - 16) * scale;
            dst[b * 32 + i + 16] = (x1 - 16) * scale;
        }
    }
}

// ─── Q5_1 Dequantization ────────────────────────────────────────────────────

static void dequant_q5_1(const void *src, float *dst, int32_t n) {
    const ss_block_q5_1_t *blocks = (const ss_block_q5_1_t *)src;
    int32_t num_blocks = n / 32;

    for (int32_t b = 0; b < num_blocks; b++) {
        float d = ss_f16_to_f32(blocks[b].d);
        float m = ss_f16_to_f32(blocks[b].m);
        uint32_t qh;
        memcpy(&qh, blocks[b].qh, sizeof(qh));

        for (int32_t j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int x0 = (blocks[b].qs[j] & 0x0F) | xh_0;
            const int x1 = (blocks[b].qs[j] >>   4) | xh_1;
            dst[b * 32 + j]      = x0 * d + m;
            dst[b * 32 + j + 16] = x1 * d + m;
        }
    }
}

// ─── Q8_0 Dequantization ────────────────────────────────────────────────────

static void dequant_q8_0(const void *src, float *dst, int32_t n) {
    const ss_block_q8_0_t *blocks = (const ss_block_q8_0_t *)src;
    int32_t num_blocks = n / 32;

    for (int32_t b = 0; b < num_blocks; b++) {
        float scale = ss_f16_to_f32(blocks[b].scale);
        for (int32_t i = 0; i < 32; i++) {
            dst[b * 32 + i] = scale * blocks[b].qs[i];
        }
    }
}

// ─── Q8_1 Dequantization ────────────────────────────────────────────────────

static void dequant_q8_1(const void *src, float *dst, int32_t n) {
    const ss_block_q8_1_t *blocks = (const ss_block_q8_1_t *)src;
    int32_t num_blocks = n / 32;

    for (int32_t b = 0; b < num_blocks; b++) {
        float d = blocks[b].d;
        for (int32_t i = 0; i < 32; i++) {
            dst[b * 32 + i] = d * blocks[b].qs[i];
        }
    }
}

// ─── F16 Dequantization ─────────────────────────────────────────────────────

static void dequant_f16(const void *src, float *dst, int32_t n) {
    const uint16_t *f16 = (const uint16_t *)src;
    for (int32_t i = 0; i < n; i++) {
        dst[i] = ss_f16_to_f32(f16[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// K-Quant Dequantization (super-blocks of 256 values)
// ═══════════════════════════════════════════════════════════════════════════

// ─── Q2_K Dequantization ────────────────────────────────────────────────────

static void dequant_q2_K(const void *src, float *dst, int32_t n) {
    const ss_block_q2_K_t *x = (const ss_block_q2_K_t *)src;
    int32_t num_blocks = n / SS_QK_K;

    for (int32_t i = 0; i < num_blocks; i++) {
        const float d   = ss_f16_to_f32(x[i].d);
        const float min = ss_f16_to_f32(x[i].dmin);
        const uint8_t *q = x[i].qs;

        int is = 0;
        float dl, ml;
        float *y = dst + (size_t)i * SS_QK_K;

        for (int32_t n_idx = 0; n_idx < SS_QK_K; n_idx += 128) {
            int shift = 0;
            for (int32_t j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int32_t l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                }

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int32_t l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                }

                shift += 2;
            }
            q += 32;
        }
    }
}

// ─── Q3_K Dequantization ────────────────────────────────────────────────────

static void dequant_q3_K(const void *src, float *dst, int32_t n) {
    const ss_block_q3_K_t *x = (const ss_block_q3_K_t *)src;
    int32_t num_blocks = n / SS_QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;

    for (int32_t i = 0; i < num_blocks; i++) {
        const float d_all = ss_f16_to_f32(x[i].d);
        const uint8_t *q  = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        float *y = dst + (size_t)i * SS_QK_K;

        for (int32_t n_idx = 0; n_idx < SS_QK_K; n_idx += 128) {
            int shift = 0;
            for (int32_t j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int32_t l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int32_t l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// ─── Q4_K Dequantization ────────────────────────────────────────────────────

// Helper: extract 6-bit scale and min from packed K_SCALE array
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void dequant_q4_K(const void *src, float *dst, int32_t n) {
    const ss_block_q4_K_t *x = (const ss_block_q4_K_t *)src;
    int32_t num_blocks = n / SS_QK_K;

    for (int32_t i = 0; i < num_blocks; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = ss_f16_to_f32(x[i].d);
        const float min = ss_f16_to_f32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int32_t j = 0; j < SS_QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;

            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int32_t l = 0; l < 32; ++l) {
                dst[i * SS_QK_K + j + l]      = d1 * (q[l] & 0xF) - m1;
            }
            for (int32_t l = 0; l < 32; ++l) {
                dst[i * SS_QK_K + j + l + 32] = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

// ─── Q5_K Dequantization ────────────────────────────────────────────────────

static void dequant_q5_K(const void *src, float *dst, int32_t n) {
    const ss_block_q5_K_t *x = (const ss_block_q5_K_t *)src;
    int32_t num_blocks = n / SS_QK_K;

    for (int32_t i = 0; i < num_blocks; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d   = ss_f16_to_f32(x[i].d);
        const float min = ss_f16_to_f32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        float *y = dst + (size_t)i * SS_QK_K;

        for (int32_t j = 0; j < SS_QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;

            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int32_t l = 0; l < 32; ++l) {
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            }
            for (int32_t l = 0; l < 32; ++l) {
                *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ─── Q6_K Dequantization ────────────────────────────────────────────────────

static void dequant_q6_K(const void *src, float *dst, int32_t n) {
    const ss_block_q6_K_t *x = (const ss_block_q6_K_t *)src;
    int32_t num_blocks = n / SS_QK_K;

    for (int32_t i = 0; i < num_blocks; i++) {
        const float d = ss_f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        float *y = dst + (size_t)i * SS_QK_K;

        for (int32_t n_idx = 0; n_idx < SS_QK_K; n_idx += 128) {
            for (int32_t l = 0; l < 32; ++l) {
                int32_t is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch
// ═══════════════════════════════════════════════════════════════════════════

void ss_dequantize(const void *src, float *dst, int32_t n, ggml_type_t type) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            break;
        case GGML_TYPE_F16:
            dequant_f16(src, dst, n);
            break;
        case GGML_TYPE_Q4_0:
            dequant_q4_0(src, dst, n);
            break;
        case GGML_TYPE_Q4_1:
            dequant_q4_1(src, dst, n);
            break;
        case GGML_TYPE_Q5_0:
            dequant_q5_0(src, dst, n);
            break;
        case GGML_TYPE_Q5_1:
            dequant_q5_1(src, dst, n);
            break;
        case GGML_TYPE_Q8_0:
            dequant_q8_0(src, dst, n);
            break;
        case GGML_TYPE_Q8_1:
            dequant_q8_1(src, dst, n);
            break;
        case GGML_TYPE_Q2_K:
            dequant_q2_K(src, dst, n);
            break;
        case GGML_TYPE_Q3_K:
            dequant_q3_K(src, dst, n);
            break;
        case GGML_TYPE_Q4_K:
            dequant_q4_K(src, dst, n);
            break;
        case GGML_TYPE_Q5_K:
            dequant_q5_K(src, dst, n);
            break;
        case GGML_TYPE_Q6_K:
            dequant_q6_K(src, dst, n);
            break;
        default:
            // Unsupported type — zero fill and log once
            {
                static int warn_count = 0;
                if (warn_count < 3) {
                    printf("[WARN] Unsupported quant type=%d, n=%d — zero filling\n", (int)type, n);
                    warn_count++;
                }
            }
            memset(dst, 0, n * sizeof(float));
            break;
    }
}

// ─── Fused Dequant + MatVec ─────────────────────────────────────────────────

void ss_dequant_matvec(float *out, const void *weight, const float *x,
                       int32_t rows, int32_t cols, ggml_type_t type) {
    size_t type_size = ss_ggml_type_size(type);
    uint32_t block_size = ss_ggml_block_size(type);

    if (type_size == 0 || block_size == 0) return;

    size_t row_bytes = ((size_t)cols / block_size) * type_size;

    // Match thread count to hardware cores for optimal throughput (cache result)
    static int32_t cached_ncpu = 0;
    if (cached_ncpu == 0) {
        int32_t hw_ncpu = 0;
        size_t len = sizeof(hw_ncpu);
        if (sysctlbyname("hw.activecpu", &hw_ncpu, &len, NULL, 0) == 0 && hw_ncpu > 0) {
            cached_ncpu = hw_ncpu;
        } else {
            cached_ncpu = 6;
        }
    }
    int32_t ncpu = cached_ncpu;
    if (ncpu > rows) ncpu = rows;

    int32_t rows_per_chunk = (rows + ncpu - 1) / ncpu;

    // Pre-allocate all thread buffers at once (single malloc for all threads)
    float *all_bufs = (float *)malloc((size_t)ncpu * cols * sizeof(float));
    if (!all_bufs) return;

    dispatch_apply(ncpu, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk) {
        int32_t start_r = (int32_t)chunk * rows_per_chunk;
        int32_t end_r = start_r + rows_per_chunk;
        if (end_r > rows) end_r = rows;

        // Each thread gets its own slice of the pre-allocated buffer
        float *tmp = all_bufs + chunk * cols;

        for (int32_t r = start_r; r < end_r; r++) {
            const uint8_t *row_ptr = (const uint8_t *)weight + (size_t)r * row_bytes;
            ss_dequantize(row_ptr, tmp, cols, type);

            float sum = 0.0f;
            vDSP_dotpr(tmp, 1, x, 1, &sum, cols);
            out[r] = sum;
        }
    });

    free(all_bufs);
}
