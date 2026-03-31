/**
 * SlipStream — Quantization Unit Tests
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Import quantize functions
extern float ss_f16_to_f32(uint16_t h);
extern void ss_dequantize(const void *src, float *dst, int32_t n, int type);

// ─── Test Helpers ────────────────────────────────────────────────────────────

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [TEST] %-40s ", #name); \
    if (test_##name()) { tests_passed++; printf("✅ PASS\n"); } \
    else { printf("❌ FAIL\n"); } \
} while(0)

static int approx_eq(float a, float b, float tol) {
    return fabsf(a - b) < tol;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

static int test_f16_to_f32_zero(void) {
    return approx_eq(ss_f16_to_f32(0x0000), 0.0f, 1e-10f);
}

static int test_f16_to_f32_one(void) {
    // IEEE 754 half-precision: 1.0 = 0 01111 0000000000 = 0x3C00
    return approx_eq(ss_f16_to_f32(0x3C00), 1.0f, 1e-5f);
}

static int test_f16_to_f32_negative(void) {
    // -1.0 = 1 01111 0000000000 = 0xBC00
    return approx_eq(ss_f16_to_f32(0xBC00), -1.0f, 1e-5f);
}

static int test_f16_to_f32_half(void) {
    // 0.5 = 0 01110 0000000000 = 0x3800
    return approx_eq(ss_f16_to_f32(0x3800), 0.5f, 1e-5f);
}

static int test_dequant_f32_passthrough(void) {
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[4] = {0};

    ss_dequantize(src, dst, 4, 0);  // GGML_TYPE_F32 = 0

    return approx_eq(dst[0], 1.0f, 1e-5f) &&
           approx_eq(dst[1], 2.0f, 1e-5f) &&
           approx_eq(dst[2], 3.0f, 1e-5f) &&
           approx_eq(dst[3], 4.0f, 1e-5f);
}

static int test_dequant_f16(void) {
    uint16_t src[4] = {
        0x3C00,  // 1.0
        0x4000,  // 2.0
        0x4200,  // 3.0
        0x4400,  // 4.0
    };
    float dst[4] = {0};

    ss_dequantize(src, dst, 4, 1);  // GGML_TYPE_F16 = 1

    return approx_eq(dst[0], 1.0f, 1e-3f) &&
           approx_eq(dst[1], 2.0f, 1e-3f) &&
           approx_eq(dst[2], 3.0f, 1e-3f) &&
           approx_eq(dst[3], 4.0f, 1e-3f);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(void) {
    printf("\n🌊 SlipStream — Quantization Tests\n\n");

    TEST(f16_to_f32_zero);
    TEST(f16_to_f32_one);
    TEST(f16_to_f32_negative);
    TEST(f16_to_f32_half);
    TEST(dequant_f32_passthrough);
    TEST(dequant_f16);

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
