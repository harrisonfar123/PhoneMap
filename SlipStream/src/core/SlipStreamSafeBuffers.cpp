#include "SlipStreamSafeBuffers.hpp"
#include <cstdlib>
#include <iostream>

namespace slipstream {
namespace safety {

void handle_bounds_violation(
    const char* file, int line, const char* func,
    const char* index_expr, size_t index, size_t size, 
    const char* context
) {
    // In production, this would be an os_log fault and potential safe crash logic.
    std::cerr << "bounds violation in " << file << ":" << line << " (" << func << ")\n";
    std::cerr << "Index out of bounds accessing " << index_expr << "=" << index 
              << " against size=" << size << " Context: " << context << "\n";
    abort();
}

SafeLogitBuffer::~SafeLogitBuffer() {
    if (data_) {
        free(data_);
        data_ = nullptr;
    }
}

BufferStatus SafeLogitBuffer::allocate(uint32_t vocab_size) {
    if (vocab_size == 0 || vocab_size > MAX_VOCAB_SIZE) {
        return BufferStatus::SizeTooLarge;
    }
    
    size_t buffer_size = 0;
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_mul_overflow(static_cast<size_t>(vocab_size), sizeof(float), &buffer_size)) {
        return BufferStatus::SizeTooLarge;
    }
#else
    buffer_size = static_cast<size_t>(vocab_size) * sizeof(float);
#endif

    // Add padding for alignment
    buffer_size = (buffer_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGNMENT, buffer_size) != 0) {
        return BufferStatus::AllocationFailed;
    }
    
    setup_guard_canaries(ptr, buffer_size, vocab_size);
    
    data_ = static_cast<float*>(ptr);
    capacity_ = buffer_size;
    vocab_size_ = vocab_size;
    
    return BufferStatus::Success;
}

void SafeLogitBuffer::setup_guard_canaries(void* ptr, size_t buffer_size, uint32_t vocab_size) {
    // Write canary values at end of buffer
    float* end_canary = static_cast<float*>(
        static_cast<void*>(static_cast<char*>(ptr) + buffer_size - ALIGNMENT)
    );
    for (size_t i = 0; i < ALIGNMENT / sizeof(float); i++) {
        // Pun float bits with DEADBEEF
        uint32_t canary = 0xDEADBEEF;
        std::memcpy(&end_canary[i], &canary, sizeof(float));
    }
}

} // namespace safety
} // namespace slipstream
