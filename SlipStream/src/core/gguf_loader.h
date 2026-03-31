/**
 * SlipStream — GGUF Model File Loader
 *
 * Parses GGUF v3 format files and builds a layer-to-offset index
 * for memory-mapped weight streaming.
 */

#ifndef SS_GGUF_LOADER_H
#define SS_GGUF_LOADER_H

#include "slipstream.h"
#include <stdio.h>

// ─── GGUF Constants ──────────────────────────────────────────────────────────

#define GGUF_MAGIC          0x46554747  // "GGUF"
#define GGUF_VERSION         3
#define GGUF_MAX_DIMS        4
#define GGUF_MAX_NAME_LEN    256

// ─── GGUF Data Types ─────────────────────────────────────────────────────────

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

// ─── GGUF Tensor Types ──────────────────────────────────────────────────────

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
} ggml_type_t;

// ─── GGUF Tensor Info ────────────────────────────────────────────────────────

typedef struct {
    char         name[GGUF_MAX_NAME_LEN];
    uint32_t     n_dims;
    uint64_t     dims[GGUF_MAX_DIMS];
    ggml_type_t  type;
    uint64_t     offset;         // Byte offset within the data section
    uint64_t     size_bytes;     // Total size of this tensor's data
    int32_t      layer_index;    // Which transformer layer (-1 = shared/embedding)
} ss_tensor_info_t;

// ─── Layer Index ─────────────────────────────────────────────────────────────

typedef struct {
    uint32_t          num_tensors;      // Number of tensors in this layer
    ss_tensor_info_t *tensors;          // Array of tensor infos
    uint64_t          total_size;       // Total bytes for all tensors in this layer
    uint64_t          data_offset_min;  // Earliest byte offset in GGUF data
    uint64_t          data_offset_max;  // Latest byte offset + size in GGUF data
} ss_layer_info_t;

// ─── GGUF File Handle ────────────────────────────────────────────────────────

typedef struct {
    // File
    int             fd;              // File descriptor
    uint64_t        file_size;       // Total file size
    uint8_t        *mmap_base;       // Memory-mapped file base
    uint64_t        data_offset;     // Offset where tensor data starts

    // Header info
    uint32_t        version;
    uint64_t        n_tensors;
    uint64_t        n_kv;

    // Tensor catalog
    ss_tensor_info_t *tensors;       // All tensors
    uint64_t          tensor_count;

    // Layer index (for streaming)
    ss_layer_info_t  *layers;        // Indexed by layer number
    uint32_t          num_layers;

    // Model metadata extracted from KV pairs
    char              arch[64];
    uint32_t          hidden_size;
    uint32_t          num_heads;
    uint32_t          num_kv_heads;
    uint32_t          vocab_size;
    char            **vocab;
    float            *vocab_scores;
    char            **merges;          // BPE merge rules from tokenizer.ggml.merges
    uint32_t          n_merges;        // Number of merge rules
    uint32_t          context_length;
    uint32_t          intermediate_size;
    uint32_t          head_dim;
    uint32_t          rope_dim;
    float             rope_theta;
    float             rms_norm_eps;
    float             attn_logit_softcapping;
    float             final_logit_softcapping;
} ss_gguf_file_t;

// ─── API ─────────────────────────────────────────────────────────────────────

/**
 * Open and parse a GGUF file, building the layer index.
 */
ss_gguf_file_t *ss_gguf_open(const char *path);

/**
 * Get a pointer to a specific tensor's data via mmap.
 * The data is only paged in when accessed (lazy loading).
 */
const void *ss_gguf_tensor_data(const ss_gguf_file_t *file,
                                 const ss_tensor_info_t *tensor);

/**
 * Advise the OS to page in a layer's data (prefetch).
 */
void ss_gguf_prefetch_layer(const ss_gguf_file_t *file, uint32_t layer_idx);

/**
 * Advise the OS to evict a layer's data from page cache.
 */
void ss_gguf_evict_layer(const ss_gguf_file_t *file, uint32_t layer_idx);

/**
 * Close the GGUF file and free all resources.
 */
void ss_gguf_close(ss_gguf_file_t *file);

/**
 * Get the size in bytes of a ggml type element.
 */
size_t ss_ggml_type_size(ggml_type_t type);

/**
 * Get the block size for a quantized type (number of elements per block).
 */
uint32_t ss_ggml_block_size(ggml_type_t type);

#endif // SS_GGUF_LOADER_H
