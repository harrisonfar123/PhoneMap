/**
 * SlipStream — Metal GPU Compute Backend Implementation
 *
 * Provides GPU-accelerated matrix-vector multiply using Metal compute shaders.
 * Only compiled when SS_HAS_METAL is defined (Apple platforms).
 */

#ifdef SS_HAS_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backends/metal_backend.h"
#include <unistd.h>

// ─── Metal Compute Shader Source ─────────────────────────────────────────────

static NSString *const kMetalShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Matrix-vector multiply kernel
// Each thread computes one output element (one dot product)
kernel void matvec(
    device const float *mat    [[buffer(0)]],
    device const float *vec    [[buffer(1)]],
    device float       *out    [[buffer(2)]],
    constant uint      &cols   [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]])
{
    float sum = 0.0f;
    uint row_offset = gid * cols;

    // Process 4 elements at a time
    uint i = 0;
    for (; i + 3 < cols; i += 4) {
        sum += mat[row_offset + i]     * vec[i];
        sum += mat[row_offset + i + 1] * vec[i + 1];
        sum += mat[row_offset + i + 2] * vec[i + 2];
        sum += mat[row_offset + i + 3] * vec[i + 3];
    }
    for (; i < cols; i++) {
        sum += mat[row_offset + i] * vec[i];
    }

    out[gid] = sum;
}

// Q4_0 block structure
struct block_q4_0 {
    half d;
    uchar qs[16];
};

// Matrix-vector multiply for Q4_0 quantized weights
kernel void matvec_q4_0(
    device const block_q4_0 *mat    [[buffer(0)]],
    device const float      *vec    [[buffer(1)]],
    device float            *out    [[buffer(2)]],
    constant uint           &cols   [[buffer(3)]],
    uint                     gid    [[thread_position_in_grid]])
{
    float sum = 0.0f;
    uint block_size = 32;
    uint num_blocks = cols / block_size;
    uint row_offset = gid * num_blocks;

    // Multiply using natively accelerated float4 and 16-byte vector loads!
    for (uint b = 0; b < num_blocks; b++) {
        device const block_q4_0 *block = &mat[row_offset + b];
        float d = (float)block->d;
        device const float *v = vec + (b * block_size);
        
        for (uint i = 0; i < 16; i += 4) {
            // Read 4 quant bytes simultaneously (representing 8 weights)
            uchar4 q = *(device const uchar4*)(&block->qs[i]);
            
            // Decompress the two halves natively using vector casting
            float4 v0 = float4(q & (uchar4)0x0F) - 8.0f;
            float4 v1 = float4(q >> 4) - 8.0f;
            
            // 128-bit aligned float loads natively bypass VRAM memory bottlenecks
            float4 vec_0 = *(device const float4*)(v + i);
            float4 vec_1 = *(device const float4*)(v + i + 16);
            
            // Accelerated dual-dot product sum compilation!
            sum += dot(v0 * d, vec_0) + dot(v1 * d, vec_1);
        }
    }
    
    out[gid] = sum;
}

// Q4_K block structure
struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

inline void get_scale_min_k4(int j, device const uchar *q, thread uchar &d, thread uchar &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

kernel void matvec_q4_K(
    device const block_q4_K *mat    [[buffer(0)]],
    device const float      *vec    [[buffer(1)]],
    device float            *out    [[buffer(2)]],
    constant uint           &cols   [[buffer(3)]],
    uint                     gid    [[thread_position_in_grid]])
{
    float sum = 0.0f;
    uint num_blocks = cols / 256;
    uint row_offset = gid * num_blocks;

    for (uint b = 0; b < num_blocks; b++) {
        device const block_q4_K *block = &mat[row_offset + b];
        float d = (float)block->d;
        float min = (float)block->dmin;
        uint vec_offset = b * 256;
        
        int is = 0;
        uchar sc, m;
        for (uint j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, block->scales, sc, m);
            float d1 = d * sc;
            float m1 = min * m;

            get_scale_min_k4(is + 1, block->scales, sc, m);
            float d2 = d * sc;
            float m2 = min * m;

            for (uint l = 0; l < 32; l += 4) {
                uchar4 q_val = *(device const uchar4*)(&block->qs[j/2 + l]);
                
                float4 q1 = float4(q_val & (uchar4)0xF);
                float4 q2 = float4(q_val >> 4);
                
                float4 v1 = *(device const float4*)(vec + vec_offset + j + l);
                float4 v2 = *(device const float4*)(vec + vec_offset + j + l + 32);
                
                sum += dot(d1 * q1 - m1, v1) + dot(d2 * q2 - m2, v2);
            }
            is += 2;
        }
    }
    out[gid] = sum;
}
)";

// ─── Metal Context ───────────────────────────────────────────────────────────

#define MAX_TENSOR_BUFS 1024

struct ss_metal_ctx {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLComputePipelineState> matvec_pipeline;
    id<MTLComputePipelineState> matvec_q4_0_pipeline;
    id<MTLComputePipelineState> matvec_q4_K_pipeline;

    // Vector and output buffers
    id<MTLBuffer> vec_buf;
    id<MTLBuffer> out_buf;
    size_t vec_buf_size;
    size_t out_buf_size;

    // Zero-Copy Dynamic Tensor Registry (Bypasses Metal 3GB Max Limit)
    void         *tensor_bufs[MAX_TENSOR_BUFS];
    uintptr_t     tensor_addrs[MAX_TENSOR_BUFS];
    int           num_tensor_bufs;

    // Parallel XPC Batching Architecture
    id<MTLCommandBuffer>         active_command_buffer;
    id<MTLComputeCommandEncoder> active_command_encoder;
    float                       *batch_cpu_outs[16];
    void                        *batch_gpu_outs[16];
    size_t                       batch_out_sizes[16];
    int                          batch_count;
};

// ─── Zero-Copy Tensor Wrapper ──────────────────────────────────────────────

static id<MTLBuffer> get_zerocopy_mat_buf(ss_metal_ctx_t *ctx, const void *ptr, size_t size, NSUInteger *out_offset) {
    if (!ctx || !ptr || size == 0) return nil;
    
    // Page alignment required by newBufferWithBytesNoCopy (e.g. 16KB on M-Series / A-Series)
    size_t page_size = (size_t)getpagesize();
    uint64_t aligned_addr = ((uint64_t)ptr) & ~((uint64_t)page_size - 1);
    uint64_t diff = ((uint64_t)ptr) - aligned_addr;
    uint64_t aligned_size = (size + diff + page_size - 1) & ~((uint64_t)page_size - 1);
    
    *out_offset = (NSUInteger)diff;
    
    // Check if we've already securely bridged this target tensor address
    for (int i = 0; i < ctx->num_tensor_bufs; i++) {
        if (ctx->tensor_addrs[i] == (uintptr_t)ptr) {
            return (__bridge id<MTLBuffer>)(ctx->tensor_bufs[i]);
        }
    }
    
    // Otherwise, attempt a localized Zero-Copy Apple Silicon buffer wrapper on the fly!
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void *)aligned_addr 
                                                       length:(NSUInteger)aligned_size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];

    if (buf == nil) {
        // iOS Sandbox Sandbox blocks `newBufferWithBytesNoCopy` for App-Data page cache models!
        // We MUST permanently clone the tensor into a native GPU-safe hardware buffer exactly once.
        buf = [ctx->device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (buf != nil) {
            memcpy(buf.contents, ptr, size);
            *out_offset = 0; // We isolated purely the data structure, no page bleeding
        } else {
            return nil; // Unrecoverable OOM constraint, let it elegantly bounce to CPU Neon fallback
        }
    }
                                                  
    // CACHE the generated tensor (whether zero-copy or native copy) dynamically 
    // This absolutely guarantees we never `memcpy` the tensor matrix twice!
    if (ctx->num_tensor_bufs < MAX_TENSOR_BUFS && buf != nil) {
        ctx->tensor_addrs[ctx->num_tensor_bufs] = (uintptr_t)ptr;
        ctx->tensor_bufs[ctx->num_tensor_bufs] = (__bridge_retained void *)buf;
        ctx->num_tensor_bufs++;
    }
    
    return buf;
}

// ─── Initialize ──────────────────────────────────────────────────────────────

ss_metal_ctx_t *ss_metal_init(void) {
#if TARGET_OS_SIMULATOR
    // The iOS Simulator introduces enormous virtualization driver overhead on Metal GPU
    // command buffers, making inference hundreds of times slower than the host CPU.
    // Falling back natively to vDSP and CPU NEON disables this massive throttle.
    fprintf(stderr, "\n[SlipStream WARNING] iOS Simulator detected! Disabling standard Metal GPU backend.\n"
                    "Simulator ML tensor operations face severe virtualization translation barriers (often 10-100x slower).\n"
                    "Executing fallback CPU path (expect ~0.2 to 1.5 tok/s natively).\n"
                    "Deploy to a physical device for maximum fluid generation!\n\n");
    return NULL;
#endif

    ss_metal_ctx_t *ctx = (ss_metal_ctx_t *)calloc(1, sizeof(ss_metal_ctx_t));
    if (!ctx) return NULL;

    @autoreleasepool {
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            free(ctx);
            return NULL;
        }

        ctx->queue = [ctx->device newCommandQueue];

        // Compile shader
        NSError *error = nil;
        id<MTLLibrary> library = [ctx->device newLibraryWithSource:kMetalShaderSource
                                                           options:nil
                                                             error:&error];
        if (!library) {
            NSLog(@"SlipStream Metal: Failed to compile shader: %@", error);
            free(ctx);
            return NULL;
        }

        id<MTLFunction> matvecFunc = [library newFunctionWithName:@"matvec"];
        ctx->matvec_pipeline = [ctx->device newComputePipelineStateWithFunction:matvecFunc
                                                                         error:&error];
        if (!ctx->matvec_pipeline) {
            NSLog(@"SlipStream Metal: Failed to create pipeline: %@", error);
            free(ctx);
            return NULL;
        }

        id<MTLFunction> matvecQ4Func = [library newFunctionWithName:@"matvec_q4_0"];
        ctx->matvec_q4_0_pipeline = [ctx->device newComputePipelineStateWithFunction:matvecQ4Func
                                                                              error:&error];
        if (!ctx->matvec_q4_0_pipeline) {
            NSLog(@"SlipStream Metal: Failed to create Q4_0 pipeline: %@", error);
            free(ctx);
            return NULL;
        }

        id<MTLFunction> matvecQ4KFunc = [library newFunctionWithName:@"matvec_q4_K"];
        ctx->matvec_q4_K_pipeline = [ctx->device newComputePipelineStateWithFunction:matvecQ4KFunc
                                                                               error:&error];
        if (!ctx->matvec_q4_K_pipeline) {
            NSLog(@"SlipStream Metal: Failed to create Q4_K pipeline: %@", error);
            free(ctx);
            return NULL;
        }
    }

    return ctx;
}

// ─── Parallel Batching ───────────────────────────────────────────────────────

void ss_metal_begin_batch(ss_metal_ctx_t *ctx) {
    if (!ctx || ctx->active_command_buffer) return;
    ctx->active_command_buffer = [ctx->queue commandBuffer];
    ctx->active_command_encoder = [ctx->active_command_buffer computeCommandEncoder];
    ctx->batch_count = 0;
}

void ss_metal_end_batch(ss_metal_ctx_t *ctx) {
    if (!ctx || !ctx->active_command_buffer) return;
    
    @autoreleasepool {
        [ctx->active_command_encoder endEncoding];
        [ctx->active_command_buffer commit];
        [ctx->active_command_buffer waitUntilCompleted];
        
        // Natively sync ALL batch outputs back to CPU now that the GPU is completely finished!
        for (int i = 0; i < ctx->batch_count; i++) {
            id<MTLBuffer> gpu_out = (__bridge_transfer id<MTLBuffer>)ctx->batch_gpu_outs[i];
            memcpy(ctx->batch_cpu_outs[i], [gpu_out contents], ctx->batch_out_sizes[i]);
        }
        
        ctx->batch_count = 0;
        ctx->active_command_encoder = nil;
        ctx->active_command_buffer = nil;
    }
}

// ─── Should Use GPU ──────────────────────────────────────────────────────────

bool ss_metal_should_use_gpu(int32_t rows, int32_t cols) {
    // Because all tensors are now permanently cached in zero-copy MTLBuffers with no transfer overhead,
    // we should forcefully keep EVERYTHING on the GPU to avoid catastrophic CPU thread desynchronization!
    return (int64_t)rows * cols > 1024;  // Evaluate anything larger than tiny 1D arrays on the GPU
}

// ─── Matrix-Vector Multiply ─────────────────────────────────────────────────

bool ss_metal_matvec(ss_metal_ctx_t *ctx,
                     float *out,
                     const float *mat,
                     const float *vec,
                     int32_t rows,
                     int32_t cols) {
    if (!ctx || !out || !mat || !vec) return false;

    @autoreleasepool {
        size_t mat_size = (size_t)rows * cols * sizeof(float);
        size_t vec_size = (size_t)cols * sizeof(float);
        size_t out_size = (size_t)rows * sizeof(float);

        // 1. Vector and Output buffers (Always copied since they are small/dynamic)
        if (!ctx->vec_buf || ctx->vec_buf_size < vec_size) {
            ctx->vec_buf = [ctx->device newBufferWithLength:vec_size
                                                    options:MTLResourceStorageModeShared];
            ctx->vec_buf_size = vec_size;
        }
        if (!ctx->out_buf || ctx->out_buf_size < out_size) {
            ctx->out_buf = [ctx->device newBufferWithLength:out_size
                                                    options:MTLResourceStorageModeShared];
            ctx->out_buf_size = out_size;
        }
        memcpy(ctx->vec_buf.contents, vec, vec_size);

        // 2. Matrix Buffer (Zero-Copy UMA Cache)
        NSUInteger global_offset = 0;
        id<MTLBuffer> mat_buf = get_zerocopy_mat_buf(ctx, mat, mat_size, &global_offset);
        if (!mat_buf) return false;

        // Create command buffer and encoder
        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:ctx->matvec_pipeline];
        [encoder setBuffer:mat_buf       offset:global_offset atIndex:0];
        [encoder setBuffer:ctx->vec_buf  offset:0 atIndex:1];
        [encoder setBuffer:ctx->out_buf  offset:0 atIndex:2];
        
        uint32_t cols_u = (uint32_t)cols;
        [encoder setBytes:&cols_u length:sizeof(uint32_t) atIndex:3];

        // Dispatch: one thread per output row
        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        
        // Find a threadgroup size that perfectly divides rows, up to 256
        NSUInteger threadGroupSize = 256;
        while (rows % threadGroupSize != 0 && threadGroupSize > 1) {
            threadGroupSize /= 2;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            // Execution was aborted, likely because the app entered the background 
            // without permission to submit GPU commands. Return false to invoke CPU fallback!
            return false;
        }

        // Read back result
        memcpy(out, ctx->out_buf.contents, out_size);
    }

    return true;
}

// ─── Q4_0 Matrix-Vector Multiply ────────────────────────────────────────────

bool ss_metal_matvec_q4_0(ss_metal_ctx_t *ctx,
                          float *out,
                          const void *mat_q4_0,
                          const float *vec,
                          int32_t rows,
                          int32_t cols) {
    if (!ctx || !out || !mat_q4_0 || !vec) return false;

    @autoreleasepool {
        // q4_0 block size is 32 elements encoded into 18 bytes
        size_t mat_size = (size_t)rows * (cols / 32) * 18;
        size_t vec_size = (size_t)cols * sizeof(float);
        size_t out_size = (size_t)rows * sizeof(float);

        // 1. Vector Buffer (Always copied since it's small/dynamic)
        if (!ctx->vec_buf || ctx->vec_buf_size < vec_size) {
            ctx->vec_buf = [ctx->device newBufferWithLength:vec_size
                                                    options:MTLResourceStorageModeShared];
            ctx->vec_buf_size = vec_size;
        }
        memcpy(ctx->vec_buf.contents, vec, vec_size);

        // 2. Output Buffer & Batch Routing
        id<MTLCommandBuffer> cmdBuf;
        id<MTLComputeCommandEncoder> encoder;
        id<MTLBuffer> target_out_buf;
        bool is_batched = (ctx->active_command_buffer != nil);

        if (is_batched) {
            cmdBuf = ctx->active_command_buffer;
            encoder = ctx->active_command_encoder;
            target_out_buf = [ctx->device newBufferWithLength:out_size options:MTLResourceStorageModeShared];
            
            if (ctx->batch_count < 16) {
                ctx->batch_cpu_outs[ctx->batch_count] = out;
                ctx->batch_gpu_outs[ctx->batch_count] = (__bridge_retained void *)target_out_buf;
                ctx->batch_out_sizes[ctx->batch_count] = out_size;
                ctx->batch_count++;
            }
        } else {
            if (!ctx->out_buf || ctx->out_buf_size < out_size) {
                ctx->out_buf = [ctx->device newBufferWithLength:out_size
                                                        options:MTLResourceStorageModeShared];
                ctx->out_buf_size = out_size;
            }
            target_out_buf = ctx->out_buf;
            cmdBuf = [ctx->queue commandBuffer];
            encoder = [cmdBuf computeCommandEncoder];
        }

        // 3. Matrix Buffer (Zero-Copy UMA Cache)
        NSUInteger global_offset = 0;
        id<MTLBuffer> mat_buf = get_zerocopy_mat_buf(ctx, mat_q4_0, mat_size, &global_offset);
        if (!mat_buf) return false;

        [encoder setComputePipelineState:ctx->matvec_q4_0_pipeline];
        [encoder setBuffer:mat_buf        offset:global_offset atIndex:0];
        [encoder setBuffer:ctx->vec_buf   offset:0 atIndex:1];
        [encoder setBuffer:target_out_buf offset:0 atIndex:2];
        
        uint32_t cols_u = (uint32_t)cols;
        [encoder setBytes:&cols_u length:sizeof(uint32_t) atIndex:3];

        // Dispatch: one thread per output row
        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        
        NSUInteger threadGroupSize = 256;
        while (rows % threadGroupSize != 0 && threadGroupSize > 1) {
            threadGroupSize /= 2;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadgroupSize];

        if (!is_batched) {
            [encoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (cmdBuf.status == MTLCommandBufferStatusError) {
                return false;
            }

            // Read back result immediately
            memcpy(out, target_out_buf.contents, out_size);
        }
    }

    return true;
}

// ─── Free ────────────────────────────────────────────────────────────────────

bool ss_metal_matvec_q4_K(ss_metal_ctx_t *ctx,
                          float *out,
                          const void *mat_q4_K,
                          const float *vec,
                          int32_t rows,
                          int32_t cols) {
    if (!ctx || !out || !mat_q4_K || !vec) return false;

    @autoreleasepool {
        // q4_K block size is 256 elements encoded into 144 bytes
        size_t mat_size = (size_t)rows * (cols / 256) * 144;
        size_t vec_size = (size_t)cols * sizeof(float);
        size_t out_size = (size_t)rows * sizeof(float);

        // 1. Vector Buffer (Always copied since it's small/dynamic)
        if (!ctx->vec_buf || ctx->vec_buf_size < vec_size) {
            ctx->vec_buf = [ctx->device newBufferWithLength:vec_size
                                                    options:MTLResourceStorageModeShared];
            ctx->vec_buf_size = vec_size;
        }
        memcpy(ctx->vec_buf.contents, vec, vec_size);

        // 2. Output Buffer & Batch Routing
        id<MTLCommandBuffer> cmdBuf;
        id<MTLComputeCommandEncoder> encoder;
        id<MTLBuffer> target_out_buf;
        bool is_batched = (ctx->active_command_buffer != nil);

        if (is_batched) {
            cmdBuf = ctx->active_command_buffer;
            encoder = ctx->active_command_encoder;
            target_out_buf = [ctx->device newBufferWithLength:out_size options:MTLResourceStorageModeShared];
            
            if (ctx->batch_count < 16) {
                ctx->batch_cpu_outs[ctx->batch_count] = out;
                ctx->batch_gpu_outs[ctx->batch_count] = (__bridge_retained void *)target_out_buf;
                ctx->batch_out_sizes[ctx->batch_count] = out_size;
                ctx->batch_count++;
            }
        } else {
            if (!ctx->out_buf || ctx->out_buf_size < out_size) {
                ctx->out_buf = [ctx->device newBufferWithLength:out_size
                                                        options:MTLResourceStorageModeShared];
                ctx->out_buf_size = out_size;
            }
            target_out_buf = ctx->out_buf;
            cmdBuf = [ctx->queue commandBuffer];
            encoder = [cmdBuf computeCommandEncoder];
        }

        // 3. Matrix Buffer (Zero-Copy UMA Cache)
        NSUInteger global_offset = 0;
        id<MTLBuffer> mat_buf = get_zerocopy_mat_buf(ctx, mat_q4_K, mat_size, &global_offset);
        if (!mat_buf) return false;

        [encoder setComputePipelineState:ctx->matvec_q4_K_pipeline];
        [encoder setBuffer:mat_buf        offset:global_offset atIndex:0];
        [encoder setBuffer:ctx->vec_buf   offset:0 atIndex:1];
        [encoder setBuffer:target_out_buf offset:0 atIndex:2];
        
        uint32_t cols_u = (uint32_t)cols;
        [encoder setBytes:&cols_u length:sizeof(uint32_t) atIndex:3];

        // Dispatch: one thread per output row
        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        
        // Find a threadgroup size that perfectly divides rows, up to 256
        NSUInteger threadGroupSize = 256;
        while (rows % threadGroupSize != 0 && threadGroupSize > 1) {
            threadGroupSize /= 2;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadgroupSize];

        // Handle Execution
        if (!is_batched) {
            [encoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (cmdBuf.status == MTLCommandBufferStatusError) {
                return false;
            }

            // Read back result immediately
            memcpy(out, target_out_buf.contents, out_size);
        }
    }

    return true;
}

// ─── Free ────────────────────────────────────────────────────────────────────

void ss_metal_free(ss_metal_ctx_t *ctx) {
    if (!ctx) return;
    // ARC handles Metal object cleanup
    free(ctx);
}

#endif // SS_HAS_METAL
