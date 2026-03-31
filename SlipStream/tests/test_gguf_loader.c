/**
 * SlipStream — GGUF Loader Unit Tests
 */

#include "slipstream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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

// Test: Loading a non-existent file returns NULL
static int test_load_nonexistent(void) {
    ss_model_t *model = ss_model_load("/nonexistent/path.gguf", NULL);
    if (model != NULL) {
        ss_model_free(model);
        return 0;
    }
    return ss_last_error() == SS_ERROR_INVALID_MODEL;
}

// Test: NULL path returns error
static int test_load_null_path(void) {
    ss_model_t *model = ss_model_load(NULL, NULL);
    return model == NULL && ss_last_error() == SS_ERROR_INVALID_PARAMS;
}

// Test: Version string is not empty
static int test_version(void) {
    const char *ver = ss_version();
    return ver != NULL && strlen(ver) > 0;
}

// Test: Error strings are meaningful
static int test_error_strings(void) {
    return strcmp(ss_error_string(SS_OK), "OK") == 0 &&
           strcmp(ss_error_string(SS_ERROR_FILE_NOT_FOUND), "File not found") == 0 &&
           strcmp(ss_error_string(SS_ERROR_OUT_OF_MEMORY), "Out of memory") == 0;
}

// Test: Default config has sensible values
static int test_default_config(void) {
    ss_model_config_t cfg = ss_default_config();
    return cfg.backend == SS_BACKEND_AUTO &&
           cfg.use_mmap == true &&
           cfg.prefetch == true;
}

// Test: Default params have sensible values
static int test_default_params(void) {
    ss_generate_params_t p = ss_default_params();
    return p.max_tokens > 0 &&
           p.temperature > 0.0f &&
           p.top_p > 0.0f &&
           p.top_k > 0;
}

// Test: Memory estimation with invalid path returns error
static int test_estimate_memory_invalid(void) {
    uint64_t mem = 0;
    ss_error_t err = ss_estimate_memory("/nonexistent.gguf", 256, &mem);
    return err == SS_ERROR_INVALID_MODEL;
}

// Test: Memory estimation with NULL params returns error
static int test_estimate_memory_null(void) {
    ss_error_t err = ss_estimate_memory(NULL, 256, NULL);
    return err == SS_ERROR_INVALID_PARAMS;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(void) {
    printf("\n🌊 SlipStream — GGUF Loader Tests\n\n");

    TEST(load_nonexistent);
    TEST(load_null_path);
    TEST(version);
    TEST(error_strings);
    TEST(default_config);
    TEST(default_params);
    TEST(estimate_memory_invalid);
    TEST(estimate_memory_null);

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
