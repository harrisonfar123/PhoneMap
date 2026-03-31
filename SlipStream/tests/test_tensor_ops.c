/**
 * SlipStream — Tensor Operations Unit Tests
 */

#include "slipstream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Import tensor ops directly
extern void ss_matmul(float *out, const float *a, const float *b,
                      int32_t M, int32_t K, int32_t N);
extern void ss_matvec(float *out, const float *mat, const float *vec,
                      int32_t rows, int32_t cols);
extern void ss_rmsnorm(float *out, const float *x, const float *weight,
                       int32_t size, float eps);
extern void ss_softmax(float *x, int32_t size);
extern void ss_silu(float *x, int32_t size);
extern int32_t ss_argmax(const float *x, int32_t size);
extern void ss_elementwise_mul(float *out, const float *a, const float *b, int32_t size);
extern void ss_elementwise_add(float *out, const float *a, const float *b, int32_t size);

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

static int test_matmul_identity(void) {
    // C = A @ B^T  (B is stored row-major, used as transposed)
    // Identity @ B^T where B = Identity → C = Identity
    float a[4] = {1, 0, 0, 1};   // Identity [2x2]
    float b[4] = {1, 0, 0, 1};   // Identity [2x2] (B^T = B for identity)
    float c[4] = {0};

    ss_matmul(c, a, b, 2, 2, 2);

    // Identity @ Identity^T = Identity
    return approx_eq(c[0], 1.0f, 1e-5f) &&
           approx_eq(c[1], 0.0f, 1e-5f) &&
           approx_eq(c[2], 0.0f, 1e-5f) &&
           approx_eq(c[3], 1.0f, 1e-5f);
}

static int test_matmul_basic(void) {
    // [1,2] @ [3,4]^T = [1*3+2*4] = [11]
    //         [5,6]     [1*5+2*6] = [17]
    float a[2] = {1, 2};
    float b[4] = {3, 4, 5, 6};
    float c[2] = {0};

    ss_matmul(c, a, b, 1, 2, 2);

    return approx_eq(c[0], 11.0f, 1e-5f) &&
           approx_eq(c[1], 17.0f, 1e-5f);
}

static int test_matvec(void) {
    // [1,2,3] @ [1] = [1+4+9] = [14]
    // [4,5,6]   [2]   [4+10+18]= [32]
    //           [3]
    float mat[6] = {1, 2, 3, 4, 5, 6};
    float vec[3] = {1, 2, 3};
    float out[2] = {0};

    ss_matvec(out, mat, vec, 2, 3);

    return approx_eq(out[0], 14.0f, 1e-5f) &&
           approx_eq(out[1], 32.0f, 1e-5f);
}

static int test_softmax_uniform(void) {
    float x[4] = {1, 1, 1, 1};
    ss_softmax(x, 4);
    return approx_eq(x[0], 0.25f, 1e-5f) &&
           approx_eq(x[1], 0.25f, 1e-5f) &&
           approx_eq(x[2], 0.25f, 1e-5f) &&
           approx_eq(x[3], 0.25f, 1e-5f);
}

static int test_softmax_sums_to_one(void) {
    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ss_softmax(x, 5);
    float sum = 0;
    for (int i = 0; i < 5; i++) sum += x[i];
    return approx_eq(sum, 1.0f, 1e-5f);
}

static int test_silu(void) {
    float x[1] = {0.0f};
    ss_silu(x, 1);
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    return approx_eq(x[0], 0.0f, 1e-5f);
}

static int test_silu_positive(void) {
    float x[1] = {2.0f};
    float expected = 2.0f / (1.0f + expf(-2.0f));
    ss_silu(x, 1);
    return approx_eq(x[0], expected, 1e-5f);
}

static int test_argmax(void) {
    float x[5] = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
    return ss_argmax(x, 5) == 3;
}

static int test_rmsnorm(void) {
    float x[4] = {1, 2, 3, 4};
    float w[4] = {1, 1, 1, 1};
    float out[4] = {0};

    ss_rmsnorm(out, x, w, 4, 1e-5f);

    // RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5)
    float rms = sqrtf(30.0f / 4.0f);
    float scale = 1.0f / rms;

    return approx_eq(out[0], 1.0f * scale, 1e-4f) &&
           approx_eq(out[1], 2.0f * scale, 1e-4f);
}

static int test_elementwise_mul(void) {
    float a[3] = {2, 3, 4};
    float b[3] = {5, 6, 7};
    float out[3] = {0};

    ss_elementwise_mul(out, a, b, 3);
    return approx_eq(out[0], 10.0f, 1e-5f) &&
           approx_eq(out[1], 18.0f, 1e-5f) &&
           approx_eq(out[2], 28.0f, 1e-5f);
}

static int test_elementwise_add(void) {
    float a[3] = {1, 2, 3};
    float b[3] = {4, 5, 6};
    float out[3] = {0};

    ss_elementwise_add(out, a, b, 3);
    return approx_eq(out[0], 5.0f, 1e-5f) &&
           approx_eq(out[1], 7.0f, 1e-5f) &&
           approx_eq(out[2], 9.0f, 1e-5f);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(void) {
    printf("\n🌊 SlipStream — Tensor Operations Tests\n\n");

    TEST(matmul_identity);
    TEST(matmul_basic);
    TEST(matvec);
    TEST(softmax_uniform);
    TEST(softmax_sums_to_one);
    TEST(silu);
    TEST(silu_positive);
    TEST(argmax);
    TEST(rmsnorm);
    TEST(elementwise_mul);
    TEST(elementwise_add);

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
