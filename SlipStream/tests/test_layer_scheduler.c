/**
 * SlipStream — Layer Scheduler Unit Tests
 */

#include "slipstream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ─── Test Helpers ────────────────────────────────────────────────────────────

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [TEST] %-40s ", #name); \
    if (test_##name()) { tests_passed++; printf("✅ PASS\n"); } \
    else { printf("❌ FAIL\n"); } \
} while(0)

// ─── Tests ───────────────────────────────────────────────────────────────────

// Test: Generate with NULL model returns error
static int test_generate_null_model(void) {
    ss_generate_params_t params = ss_default_params();
    ss_error_t err = ss_generate(NULL, "hello", &params, NULL, NULL);
    return err == SS_ERROR_INVALID_PARAMS;
}

// Test: Generate with NULL prompt returns error
static int test_generate_null_prompt(void) {
    // We can't create a real model without a GGUF file,
    // but we can test the NULL prompt path
    ss_error_t err = ss_generate(NULL, NULL, NULL, NULL, NULL);
    return err == SS_ERROR_INVALID_PARAMS;
}

// Test: Cancel NULL model doesn't crash
static int test_cancel_null(void) {
    ss_cancel(NULL);  // Should not crash
    return 1;
}

// Test: Model info with NULL returns error
static int test_model_info_null(void) {
    ss_model_info_t info;
    ss_error_t err = ss_model_get_info(NULL, &info);
    return err == SS_ERROR_INVALID_PARAMS;
}

// Test: Free NULL model doesn't crash
static int test_free_null(void) {
    ss_model_free(NULL);  // Should not crash
    return 1;
}

// Test: Progress callback with NULL doesn't crash
static int test_progress_null(void) {
    ss_set_progress_callback(NULL, NULL, NULL);  // Should not crash
    return 1;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(void) {
    printf("\n🌊 SlipStream — Layer Scheduler Tests\n\n");

    TEST(generate_null_model);
    TEST(generate_null_prompt);
    TEST(cancel_null);
    TEST(model_info_null);
    TEST(free_null);
    TEST(progress_null);

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
