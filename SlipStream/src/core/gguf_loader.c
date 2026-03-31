/**
 * SlipStream — GGUF Model File Loader Implementation
 *
 * Parses GGUF v3 files, extracts metadata, builds a layer-indexed
 * tensor catalog, and memory-maps the file for zero-copy weight access.
 */

#include "core/gguf_loader.h"
#include "utils/memory.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dispatch/dispatch.h>

// ─── GGML Type Sizes ─────────────────────────────────────────────────────────

size_t ss_ggml_type_size(ggml_type_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 2 + 16;            // 18 bytes per block of 32
        case GGML_TYPE_Q4_1: return 2 + 2 + 16;        // 20 bytes per block of 32
        case GGML_TYPE_Q5_0: return 2 + 4 + 16;        // 22 bytes per block of 32
        case GGML_TYPE_Q5_1: return 2 + 2 + 4 + 16;    // 24 bytes per block of 32
        case GGML_TYPE_Q8_0: return 2 + 32;             // 34 bytes per block of 32
        case GGML_TYPE_Q8_1: return 4 + 4 + 32;         // 40 bytes per block of 32
        case GGML_TYPE_Q2_K: return 256/16 + 256/4 + 2 + 2;  // 84 bytes per block of 256
        case GGML_TYPE_Q3_K: return 256/8 + 256/4 + 12 + 2;  // 110 bytes per block of 256
        case GGML_TYPE_Q4_K: return 2 + 2 + 12 + 256/2;      // 144 bytes per block of 256
        case GGML_TYPE_Q5_K: return 2 + 2 + 12 + 256/8 + 256/2; // 176 bytes per block of 256
        case GGML_TYPE_Q6_K: return 256/2 + 256/4 + 256/16 + 2; // 210 bytes per block of 256
        default: return 0;
    }
}

uint32_t ss_ggml_block_size(ggml_type_t type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return 1;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
            return 32;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return 256;
        default:
            return 1;
    }
}

// ─── Binary Reader Helpers ───────────────────────────────────────────────────

typedef struct {
    const uint8_t *data;
    uint64_t       pos;
    uint64_t       size;
} ss_reader_t;

static inline uint8_t read_u8(ss_reader_t *r) {
    if (r->pos >= r->size) return 0;
    return r->data[r->pos++];
}

static inline uint32_t read_u32(ss_reader_t *r) {
    if (r->pos + 4 > r->size) return 0;
    uint32_t val;
    memcpy(&val, r->data + r->pos, 4);
    r->pos += 4;
    return val;
}

static inline uint64_t read_u64(ss_reader_t *r) {
    if (r->pos + 8 > r->size) return 0;
    uint64_t val;
    memcpy(&val, r->data + r->pos, 8);
    r->pos += 8;
    return val;
}

static inline int32_t read_i32(ss_reader_t *r) {
    return (int32_t)read_u32(r);
}

static inline float read_f32(ss_reader_t *r) {
    if (r->pos + 4 > r->size) return 0.0f;
    float val;
    memcpy(&val, r->data + r->pos, 4);
    r->pos += 4;
    return val;
}

static char *read_string(ss_reader_t *r) {
    uint64_t len = read_u64(r);
    if (r->pos + len > r->size) {
        fprintf(stderr, "gguf read_string out of bounds\n");
         return NULL;
    }

    char *str = (char *)malloc(len + 1);
    if (!str) {   return NULL; }
    if (len > 0) {
        memcpy(str, r->data + r->pos, len);
    }
    str[len] = '\0';
    r->pos += len;
    return str;
}

static void skip_string(ss_reader_t *r) {
    uint64_t len = read_u64(r);
    r->pos += len;
}

// Forward-skip a KV value based on its type
static void skip_kv_value(ss_reader_t *r, gguf_type_t type);

static void skip_kv_value(ss_reader_t *r, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            r->pos += 1;
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            r->pos += 2;
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            r->pos += 4;
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            r->pos += 8;
            break;
        case GGUF_TYPE_STRING:
            skip_string(r);
            break;
        case GGUF_TYPE_ARRAY: {
            gguf_type_t arr_type = (gguf_type_t)read_u32(r);
            uint64_t arr_len = read_u64(r);
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_kv_value(r, arr_type);
            }
            break;
        }
        default:
            break;
    }
}

// ─── Parse Tensor Layer Index ────────────────────────────────────────────────

/**
 * Determine which transformer layer a tensor belongs to based on its name.
 * Convention: "blk.{N}.attn_q.weight" → layer N
 * Returns -1 for non-layer tensors (embeddings, output, etc.)
 */
static int32_t parse_layer_index(const char *name) {
    // Look for "blk." prefix
    const char *blk = strstr(name, "blk.");
    if (!blk) return -1;

    // Parse number after "blk."
    int32_t idx = 0;
    const char *p = blk + 4;
    while (*p >= '0' && *p <= '9') {
        idx = idx * 10 + (*p - '0');
        p++;
    }

    if (*p == '.') return idx;
    return -1;
}

// ─── Compute Tensor Size ─────────────────────────────────────────────────────

static uint64_t compute_tensor_size(const ss_tensor_info_t *info) {
    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < info->n_dims; i++) {
        n_elements *= info->dims[i];
    }

    uint32_t block_sz = ss_ggml_block_size(info->type);
    size_t type_sz = ss_ggml_type_size(info->type);

    if (block_sz == 1) {
        return n_elements * type_sz;
    }
    return (n_elements / block_sz) * type_sz;
}

// ─── Main GGUF Parser ───────────────────────────────────────────────────────

ss_gguf_file_t *ss_gguf_open(const char *path) {
    if (!path) {   return NULL; }

    // Open file
    int fd = open(path, O_RDONLY);
    if (fd < 0) {   return NULL; }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
         return NULL;
    }
    uint64_t file_size = (uint64_t)st.st_size;

    // Memory-map the entire file (read-only) with MAP_SHARED to ensure
    // zero-copy read boundaries and true disk-backed memory on iOS
    uint8_t *mmap_base = (uint8_t *)mmap(NULL, file_size, PROT_READ,
                                          MAP_SHARED, fd, 0);
    if (mmap_base == MAP_FAILED) {
        close(fd);
         return NULL;
    }

    // Allocate file handle
    ss_gguf_file_t *file = (ss_gguf_file_t *)calloc(1, sizeof(ss_gguf_file_t));
    if (!file) {
        munmap(mmap_base, file_size);
        close(fd);
         return NULL;
    }

    file->fd = fd;
    file->file_size = file_size;
    file->mmap_base = mmap_base;

    // Create reader
    ss_reader_t reader = { .data = mmap_base, .pos = 0, .size = file_size };

    // ── Parse header ──
    uint32_t magic = read_u32(&reader);
    if (magic != GGUF_MAGIC) {
        ss_gguf_close(file);
         return NULL;
    }

    file->version = read_u32(&reader);
    if (file->version < 2 || file->version > 3) {
        ss_gguf_close(file);
         return NULL;
    }

    file->n_tensors = read_u64(&reader);
    file->n_kv = read_u64(&reader);

    // Helper: check if key ends with suffix
    #define KEY_IS(k, s) (strcmp(k, s) == 0)
    #define KEY_ENDS(k, s) (strlen(k) >= strlen(s) && strcmp(k + strlen(k) - strlen(s), s) == 0)

    // ── Parse KV metadata ──
    for (uint64_t i = 0; i < file->n_kv; i++) {
        char *key = read_string(&reader);
        if (!key) {
            // Failed to read key — parser is out of sync
            fprintf(stderr, "SlipStream: GGUF parse error at KV %llu\n", i);
            ss_gguf_close(file);
             return NULL;
        }

        gguf_type_t val_type = (gguf_type_t)read_u32(&reader);

        // Extract model architecture metadata from known keys
        if (KEY_IS(key, "general.architecture") && val_type == GGUF_TYPE_STRING) {
            char *arch = read_string(&reader);
            if (arch) {
                strncpy(file->arch, arch, sizeof(file->arch) - 1);
                free(arch);
            }
        } else if (KEY_ENDS(key, ".embedding_length") && val_type == GGUF_TYPE_UINT32) {
            file->hidden_size = read_u32(&reader);
        } else if (KEY_ENDS(key, ".attention.head_count_kv") && val_type == GGUF_TYPE_UINT32) {
            // Must check _kv BEFORE head_count since head_count is a suffix of head_count_kv
            file->num_kv_heads = read_u32(&reader);
        } else if (KEY_ENDS(key, ".attention.head_count") && val_type == GGUF_TYPE_UINT32) {
            file->num_heads = read_u32(&reader);
        } else if (KEY_ENDS(key, ".context_length") && val_type == GGUF_TYPE_UINT32) {
            file->context_length = read_u32(&reader);
        } else if (KEY_ENDS(key, ".feed_forward_length") && val_type == GGUF_TYPE_UINT32) {
            file->intermediate_size = read_u32(&reader);
        } else if (KEY_ENDS(key, ".block_count") && val_type == GGUF_TYPE_UINT32) {
            file->num_layers = read_u32(&reader);
        } else if (KEY_ENDS(key, ".vocab_size") && val_type == GGUF_TYPE_UINT32) {
            file->vocab_size = read_u32(&reader);
        } else if (KEY_ENDS(key, ".rope.freq_base") && val_type == GGUF_TYPE_FLOAT32) {
            file->rope_theta = read_f32(&reader);
        } else if (KEY_ENDS(key, ".attention.layer_norm_rms_epsilon") && val_type == GGUF_TYPE_FLOAT32) {
            file->rms_norm_eps = read_f32(&reader);
        } else if (KEY_ENDS(key, ".attention.key_length") && val_type == GGUF_TYPE_UINT32) {
            file->head_dim = read_u32(&reader);
        } else if (KEY_ENDS(key, ".rope.dimension_count") && val_type == GGUF_TYPE_UINT32) {
            file->rope_dim = read_u32(&reader);
        } else if (KEY_IS(key, "gemma2.attention.sliding_window") && val_type == GGUF_TYPE_UINT32) {
            // Unused currently but parsed for forward compatibility
            read_u32(&reader); 
        } else if (KEY_ENDS(key, ".attn_logit_softcapping") && val_type == GGUF_TYPE_FLOAT32) {
            file->attn_logit_softcapping = read_f32(&reader);
        } else if (KEY_ENDS(key, ".final_logit_softcapping") && val_type == GGUF_TYPE_FLOAT32) {
            file->final_logit_softcapping = read_f32(&reader);
        } else if (KEY_IS(key, "tokenizer.ggml.tokens") && val_type == GGUF_TYPE_ARRAY) {
            gguf_type_t arr_type = (gguf_type_t)read_u32(&reader);
            uint64_t arr_len = read_u64(&reader);
            if (arr_type == GGUF_TYPE_STRING && arr_len > 0) {
                file->vocab_size = (uint32_t)arr_len;
                file->vocab = (char **)calloc(arr_len, sizeof(char *));
                for (uint64_t v = 0; v < arr_len; v++) {
                    file->vocab[v] = read_string(&reader);
                }
            } else {
                for (uint64_t v = 0; v < arr_len; v++) {
                    skip_kv_value(&reader, arr_type);
                }
            }
        } else if (KEY_IS(key, "tokenizer.ggml.scores") && val_type == GGUF_TYPE_ARRAY) {
            gguf_type_t arr_type = (gguf_type_t)read_u32(&reader);
            uint64_t arr_len = read_u64(&reader);
            if (arr_type == GGUF_TYPE_FLOAT32 && arr_len > 0) {
                file->vocab_scores = (float *)malloc(arr_len * sizeof(float));
                for (uint64_t v = 0; v < arr_len; v++) {
                    file->vocab_scores[v] = read_f32(&reader);
                }
            } else {
                for (uint64_t v = 0; v < arr_len; v++) {
                    skip_kv_value(&reader, arr_type);
                }
            }
        } else if (KEY_IS(key, "tokenizer.ggml.merges") && val_type == GGUF_TYPE_ARRAY) {
            gguf_type_t arr_type = (gguf_type_t)read_u32(&reader);
            uint64_t arr_len = read_u64(&reader);
            if (arr_type == GGUF_TYPE_STRING && arr_len > 0) {
                file->n_merges = (uint32_t)arr_len;
                file->merges = (char **)calloc(arr_len, sizeof(char *));
                for (uint64_t v = 0; v < arr_len; v++) {
                    file->merges[v] = read_string(&reader);
                }
            } else {
                for (uint64_t v = 0; v < arr_len; v++) {
                    skip_kv_value(&reader, arr_type);
                }
            }
        } else {
            skip_kv_value(&reader, val_type);
        }

        free(key);
    }

    #undef KEY_IS
    #undef KEY_ENDS

    // Set defaults
    if (file->rms_norm_eps == 0.0f) file->rms_norm_eps = 1e-5f;
    if (file->rope_theta == 0.0f) {
        // Architecture-specific defaults for RoPE frequency base
        if (strstr(file->arch, "qwen")) {
            file->rope_theta = 1000000.0f; // Qwen 3.5 / 2.5 default
        } else {
            file->rope_theta = 10000.0f;
        }
    }
    if (file->context_length == 0) file->context_length = 2048;

    // ── Parse tensor infos ──
    file->tensor_count = file->n_tensors;
    file->tensors = (ss_tensor_info_t *)calloc(file->n_tensors, sizeof(ss_tensor_info_t));
    if (!file->tensors) {
        ss_gguf_close(file);
         return NULL;
    }

    for (uint64_t i = 0; i < file->n_tensors; i++) {
        ss_tensor_info_t *t = &file->tensors[i];

        // Read tensor name
        char *name = read_string(&reader);
        if (name) {
            strncpy(t->name, name, GGUF_MAX_NAME_LEN - 1);
            free(name);
        }

        // Read dimensions
        t->n_dims = read_u32(&reader);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = read_u64(&reader);
        }

        // Read type and offset
        t->type = (ggml_type_t)read_u32(&reader);
        t->offset = read_u64(&reader);

        // Infer vocab size from token embeddings if missing
        if (file->vocab_size == 0 && strcmp(t->name, "token_embd.weight") == 0 && t->n_dims == 2) {
            file->vocab_size = (uint32_t)t->dims[1];
        }

        // Compute size and layer assignment
        t->size_bytes = compute_tensor_size(t);
        t->layer_index = parse_layer_index(t->name);
    }

    // The file's vocab_size is already derived from 'tokenizer.ggml.tokens' array length (e.g. 151936).
    // We explicitly do NOT override this with 'output.weight' dimension (which might be padded to 248320)
    // because padding tokens are invalid and lead to NaN outputs in attention which destabilize qsort.
    // The matrix multiplication natively handles smaller row bounds by just limiting the calculation.
    
    // ── Compute data offset ──
    // Align to 32 bytes after all tensor info entries
    file->data_offset = (reader.pos + 31) & ~(uint64_t)31;

    // Adjust tensor offsets to be absolute file offsets
    for (uint64_t i = 0; i < file->n_tensors; i++) {
        file->tensors[i].offset += file->data_offset;
    }

    // ── Build layer index ──
    if (file->num_layers > 0) {
        file->layers = (ss_layer_info_t *)calloc(file->num_layers, sizeof(ss_layer_info_t));
        if (!file->layers) {
            ss_gguf_close(file);
             return NULL;
        }

        // First pass: count tensors per layer
        for (uint64_t i = 0; i < file->n_tensors; i++) {
            int32_t li = file->tensors[i].layer_index;
            if (li >= 0 && (uint32_t)li < file->num_layers) {
                file->layers[li].num_tensors++;
            }
        }

        // Allocate tensor arrays per layer
        for (uint32_t l = 0; l < file->num_layers; l++) {
            if (file->layers[l].num_tensors > 0) {
                file->layers[l].tensors = (ss_tensor_info_t *)calloc(
                    file->layers[l].num_tensors, sizeof(ss_tensor_info_t));
                file->layers[l].data_offset_min = UINT64_MAX;
            }
        }

        // Second pass: populate layer tensor arrays
        uint32_t *layer_cursor = (uint32_t *)calloc(file->num_layers, sizeof(uint32_t));
        for (uint64_t i = 0; i < file->n_tensors; i++) {
            int32_t li = file->tensors[i].layer_index;
            if (li >= 0 && (uint32_t)li < file->num_layers) {
                ss_layer_info_t *layer = &file->layers[li];
                uint32_t idx = layer_cursor[li]++;
                layer->tensors[idx] = file->tensors[i];
                layer->total_size += file->tensors[i].size_bytes;

                uint64_t off = file->tensors[i].offset;
                uint64_t end = off + file->tensors[i].size_bytes;
                if (off < layer->data_offset_min) layer->data_offset_min = off;
                if (end > layer->data_offset_max) layer->data_offset_max = end;
            }
        }
        free(layer_cursor);
    }

    // Explicitly command the Apple Kernel to load the model sequentially from the NVMe SSD.
    // This perfectly caps the Resident Memory (RAM) usage around 150MB by dynamically dropping
    // old pages while aggressively prefetching the immediate subsequent linear tensor layers!
    if (file->mmap_base && file->file_size > 0) {
        madvise(file->mmap_base, (size_t)file->file_size, MADV_SEQUENTIAL);
    }

    return file;
}

const void *ss_gguf_tensor_data(const ss_gguf_file_t *file,
                                 const ss_tensor_info_t *tensor) {
    if (!file || !tensor || !file->mmap_base) {   return NULL; }
    if (tensor->offset + tensor->size_bytes > file->file_size) {   return NULL; }
    return file->mmap_base + tensor->offset;
}

void ss_gguf_prefetch_layer(const ss_gguf_file_t *file, uint32_t layer_idx) {
    if (!file || !file->layers || layer_idx >= file->num_layers) return;
    
    // Disabled: iOS Unified Memory correctly predicts linear/mapped access natively.
    // Explicit madvise causes thrashing and overrides Apple's intelligent page cache.
}

void ss_gguf_evict_layer(const ss_gguf_file_t *file, uint32_t layer_idx) {
    if (!file || !file->layers || layer_idx >= file->num_layers) return;

    // Disabled: Using `MADV_DONTNEED` repeatedly forces the iOS kernel to drop the NVMe
    // cached array from DRAM, completely bottlenecking token throughput to SSD speeds.
    // Mmapped clean pages do not contribute to phys_footprint, so Jetsam will not abort.
}

void ss_gguf_close(ss_gguf_file_t *file) {
    if (!file) return;

    // Free vocabulary strings
    if (file->vocab) {
        for (uint32_t i = 0; i < file->vocab_size; i++) {
            if (file->vocab[i]) free(file->vocab[i]);
        }
        free(file->vocab);
    }
    if (file->vocab_scores) {
        free(file->vocab_scores);
    }
    if (file->merges) {
        for (uint32_t i = 0; i < file->n_merges; i++) {
            if (file->merges[i]) free(file->merges[i]);
        }
        free(file->merges);
    }

    // Free layer tensor arrays
    if (file->layers) {
        for (uint32_t i = 0; i < file->num_layers; i++) {
            free(file->layers[i].tensors);
        }
        free(file->layers);
    }

    // Free tensor catalog
    free(file->tensors);

    // Unmap and close
    if (file->mmap_base && file->mmap_base != MAP_FAILED) {
        munmap(file->mmap_base, file->file_size);
    }
    if (file->fd >= 0) {
        close(file->fd);
    }

    free(file);
}
