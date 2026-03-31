#include "SlipStreamVocabValidator.hpp"

namespace slipstream {
namespace safety {

ValidationResult VocabSafetyValidator::validate_all_layers(
    const GGUFMetadata& metadata,
    const ModelConfig& config,
    const Tokenizer& tokenizer
) {
    // Layer 1: GGUF metadata
    auto gguf_vocab = metadata.get_u32("llama.vocab_size");
    if (!gguf_vocab) return {false, "Cannot determine vocab size from GGUF"};
    
    // Check tokenizer.ggml.tokens array length
    auto tokens_len = metadata.get_array_len("tokenizer.ggml.tokens");
    if (tokens_len && *tokens_len != *gguf_vocab) {
        return {false, "Vocab size mismatch: llama.vocab_size != tokenizer.ggml.tokens.len"};
    }
    
    // Layer 2: Model config
    if (config.vocab_size != *gguf_vocab) {
        return {false, "Model config vocab_size doesn't match GGUF metadata"};
    }
    
    // Layer 3: Tokenizer
    if (tokenizer.vocab_size() != *gguf_vocab) {
        return {false, "Tokenizer vocab size mismatch"};
    }
    
    // Validate special token IDs are within bounds
    auto bos = tokenizer.bos_token_id();
    if (bos && *bos >= *gguf_vocab) {
        return {false, "BOS token ID out of bounds"};
    }
    
    auto eos = tokenizer.eos_token_id();
    if (eos && *eos >= *gguf_vocab) {
        return {false, "EOS token ID out of bounds"};
    }
    
    // Layer 4: Store fingerprint for runtime checks
    stored_fingerprint_ = compute_fingerprint(*gguf_vocab, tokenizer);
    validated_vocab_size_ = *gguf_vocab;
    
    return {true, "All validation layers passed"};
}

uint32_t VocabSafetyValidator::compute_fingerprint(uint32_t vocab_size, const Tokenizer& tokenizer) const {
    uint32_t fp = vocab_size;
    fp ^= tokenizer.bos_token_id().value_or(0) << 1;
    fp ^= tokenizer.eos_token_id().value_or(0) << 2;
    return fp;
}

// Dummy impls meant to satisfy linker for now, in a real impl these call down to gguf layer.
std::optional<uint32_t> GGUFMetadata::get_u32(const std::string& key) const { return 32000; }
std::optional<size_t> GGUFMetadata::get_array_len(const std::string& key) const { return 32000; }
uint32_t Tokenizer::vocab_size() const { return 32000; }
std::optional<uint32_t> Tokenizer::bos_token_id() const { return 1; }
std::optional<uint32_t> Tokenizer::eos_token_id() const { return 2; }

} // namespace safety
} // namespace slipstream
