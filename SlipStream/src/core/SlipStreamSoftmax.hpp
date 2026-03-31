#ifndef SLIPSTREAM_SOFTMAX_HPP
#define SLIPSTREAM_SOFTMAX_HPP

#include <cstdint>

namespace slipstream {
namespace safety {

class SafeSoftmax {
public:
    static void compute(const float* input, float* output, uint32_t vocab_size);
};

} // namespace safety
} // namespace slipstream

#endif // SLIPSTREAM_SOFTMAX_HPP
