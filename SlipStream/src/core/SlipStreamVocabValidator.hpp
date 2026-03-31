#ifndef SLIPSTREAM_VOCAB_VALIDATOR_HPP
#define SLIPSTREAM_VOCAB_VALIDATOR_HPP

#include <string>
#include <optional>
#include <cstdint>

namespace slipstream {
namespace safety {

struct ValidationResult {
    bool passed;
    std::string message;
};

// Dummy interfaces for standardizing the GGUF metadata expectations
struct GGUFMetadata {
    std::optional<uint32_t> get_u32(const std::string& key) const;
    std::optional<size_t> get_array_len(const std::string& key) const;
};

struct ModelConfig {
    uint32_t vocab_size;
};

struct Tokenizer {
    uint32_t vocab_size() const;
    std::optional<uint32_t> bos_token_id() const;
    std::optional<uint32_t> eos_token_id() const;
};

class VocabSafetyValidator {
public:
    ValidationResult validate_all_layers(
        const GGUFMetadata& metadata,
        const ModelConfig& config,
        const Tokenizer& tokenizer
    );
    
    uint32_t validated_vocab_size() const { return validated_vocab_size_; }

private:
    uint32_t compute_fingerprint(uint32_t vocab_size, const Tokenizer& tokenizer) const;
    
    uint32_t validated_vocab_size_ = 0;
    uint32_t stored_fingerprint_ = 0;
};

} // namespace safety
} // namespace slipstream

#endif // SLIPSTREAM_VOCAB_VALIDATOR_HPP
