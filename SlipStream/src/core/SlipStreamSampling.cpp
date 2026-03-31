#include "SlipStreamSampling.hpp"
#include "SlipStreamSafeBuffers.hpp"
#include <vector>
#include <algorithm>
#include <limits>

namespace slipstream {
namespace safety {

struct TokenProb {
    uint32_t id;
    float probability;
    float original_logit;
    
    bool operator>(const TokenProb& other) const {
        return probability > other.probability;
    }
};

void SafeTopKSampling::apply(float* logits, uint32_t vocab_size, uint32_t k) {
    if (vocab_size == 0) return;
    
    k = std::min(k, std::min(vocab_size, MAX_K));
    
    if (k == vocab_size) return; // No filtering needed
    
    // Find k-th largest value using nth_element
    std::vector<TokenProb> tokens;
    tokens.reserve(vocab_size);
    
    for (uint32_t i = 0; i < vocab_size; i++) {
        SLIPSTREAM_CHECK_BOUNDS(i, vocab_size, "top-k token collection");
        tokens.push_back({i, logits[i], logits[i]});
    }
    
    std::nth_element(
        tokens.begin(),
        tokens.begin() + k,
        tokens.end(),
        std::greater<TokenProb>()
    );
    
    float kth_value = tokens[k].probability;
    
    // Zero out values below threshold
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (logits[i] < kth_value) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

} // namespace safety
} // namespace slipstream
