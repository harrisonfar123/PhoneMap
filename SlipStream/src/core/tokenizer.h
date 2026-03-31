/**
 * SlipStream — Minimal Tokenizer
 *
 * BPE tokenizer that reads vocabulary from GGUF metadata.
 * This is a minimal implementation sufficient for inference.
 */

#ifndef SS_TOKENIZER_H
#define SS_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct ss_tokenizer ss_tokenizer_t;

/**
 * Create a tokenizer from GGUF file data.
 * Reads the tokenizer.ggml.tokens and tokenizer.ggml.scores arrays.
 */
ss_tokenizer_t *ss_tokenizer_create(char **vocab, float *scores, uint32_t vocab_size,
                                     char **merges, uint32_t n_merges,
                                     const char *arch);

/**
 * Encode text to token IDs.
 * Returns number of tokens written to output.
 */
int32_t ss_tokenizer_encode(const ss_tokenizer_t *tok,
                             const char *text,
                             int32_t *output,
                             int32_t max_tokens);

/**
 * Decode a single token ID to text.
 * Returns pointer to internal string (valid until tokenizer is freed).
 */
const char *ss_tokenizer_decode(const ss_tokenizer_t *tok, int32_t token_id);

/**
 * Get special token IDs.
 */
int32_t ss_tokenizer_bos_id(const ss_tokenizer_t *tok);
int32_t ss_tokenizer_eos_id(const ss_tokenizer_t *tok);

/**
 * Get vocabulary size.
 */
int32_t ss_tokenizer_vocab_size(const ss_tokenizer_t *tok);

/**
 * Free tokenizer.
 */
void ss_tokenizer_free(ss_tokenizer_t *tok);

#endif // SS_TOKENIZER_H
