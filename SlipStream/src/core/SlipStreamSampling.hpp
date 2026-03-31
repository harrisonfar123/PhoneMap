#ifndef SLIPSTREAM_SAMPLING_HPP
#define SLIPSTREAM_SAMPLING_HPP

#include <cstdint>

namespace slipstream {
namespace safety {

class SafeTopKSampling {
public:
    static constexpr uint32_t MAX_K = 1000;
    
    static void apply(float* logits, uint32_t vocab_size, uint32_t k);
};

} // namespace safety
} // namespace slipstream

#endif // SLIPSTREAM_SAMPLING_HPP
