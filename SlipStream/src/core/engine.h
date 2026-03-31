/**
 * SlipStream — Main Engine
 *
 * Internal header for the engine state.
 */

#ifndef SS_ENGINE_H
#define SS_ENGINE_H

#include "slipstream.h"
#include "core/gguf_loader.h"
#include "core/layer_scheduler.h"
#include "core/tokenizer.h"
#include "utils/thread_pool.h"

#ifdef SS_HAS_METAL
#include "backends/metal_backend.h"
#endif

struct ss_model {
    ss_gguf_file_t    *gguf;
    ss_scheduler_t    *scheduler;
    ss_tokenizer_t    *tokenizer;
    ss_thread_pool_t  *thread_pool;
    ss_model_config_t  config;
    bool               cancelled;
    uint32_t           throttle_us;  // Per-token delay in microseconds (0 = no throttle)

#ifdef SS_HAS_METAL
    ss_metal_ctx_t    *metal;
#endif

    // Progress
    ss_progress_callback_t progress_cb;
    void                  *progress_user_data;
};

#endif // SS_ENGINE_H
