#ifndef SLIPSTREAM_SAFE_BUFFERS_HPP
#define SLIPSTREAM_SAFE_BUFFERS_HPP

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace slipstream {
namespace safety {

enum class BufferStatus {
    Success = 0,
    SizeTooLarge,
    AllocationFailed
};

[[noreturn]] void handle_bounds_violation(
    const char* file, int line, const char* func,
    const char* index_expr, size_t index, size_t size, 
    const char* context
);

#define SLIPSTREAM_CHECK_BOUNDS(index, size, context) \
    do { \
        if ((index) >= (size)) { \
            slipstream::safety::handle_bounds_violation( \
                __FILE__, __LINE__, __func__, \
                #index, (index), (size), (context) \
            ); \
        } \
    } while(0)

class SafeLogitBuffer {
public:
    static constexpr uint32_t MAX_VOCAB_SIZE = 500000;
    static constexpr size_t ALIGNMENT = 64;
    
    SafeLogitBuffer() : data_(nullptr), capacity_(0), vocab_size_(0) {}
    ~SafeLogitBuffer();
    
    BufferStatus allocate(uint32_t vocab_size);
    
    float get(uint32_t index) const {
        SLIPSTREAM_CHECK_BOUNDS(index, vocab_size_, "logit buffer read");
        return data_[index];
    }
    
    void set(uint32_t index, float value) {
        SLIPSTREAM_CHECK_BOUNDS(index, vocab_size_, "logit buffer write");
        data_[index] = value;
    }
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    uint32_t vocab_size() const { return vocab_size_; }

private:
    void setup_guard_canaries(void* ptr, size_t buffer_size, uint32_t vocab_size);
    
    float* data_;
    size_t capacity_;
    uint32_t vocab_size_;
};

} // namespace safety
} // namespace slipstream

#endif // SLIPSTREAM_SAFE_BUFFERS_HPP
