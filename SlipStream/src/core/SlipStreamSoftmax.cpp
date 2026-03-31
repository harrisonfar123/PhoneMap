#include "SlipStreamSoftmax.hpp"
#include <cmath>
#include <algorithm>

namespace slipstream {
namespace safety {

void SafeSoftmax::compute(const float* input, float* output, uint32_t vocab_size) {
    if (vocab_size == 0) return;

    // Step 1: Find maximum value for numerical stability
    float max_val = input[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Step 2: Compute exp(x - max) and sum
    double sum = 0.0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        float shifted = input[i] - max_val;
        // Clamp to prevent underflow/overflow in exp
        shifted = std::max(shifted, -80.0f);  // exp(-80) ~ 1e-35
        shifted = std::min(shifted, 80.0f);   // exp(80) ~ 5e34
        
        output[i] = std::exp(shifted);
        sum += output[i];
    }
    
    // Step 3: Normalize with overflow protection
    if (sum == 0.0 || !std::isfinite(sum)) {
        // Fallback: uniform distribution
        float uniform_prob = 1.0f / vocab_size;
        for (uint32_t i = 0; i < vocab_size; i++) {
            output[i] = uniform_prob;
        }
        return;
    }
    
    // Step 4: Normalize
    float inv_sum = static_cast<float>(1.0 / sum);
    for (uint32_t i = 0; i < vocab_size; i++) {
        output[i] *= inv_sum;
    }
}

} // namespace safety
} // namespace slipstream
