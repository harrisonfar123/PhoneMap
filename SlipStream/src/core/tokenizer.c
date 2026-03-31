/**
 * SlipStream — Minimal Tokenizer Implementation
 *
 * Reads vocabulary from GGUF metadata and provides basic BPE encoding.
 * Supports both SentencePiece (score-based) and Tiktoken/GPT-2 (rank-based)
 * BPE merge strategies, following llama.cpp's unicode.cpp approach.
 */

#include "core/tokenizer.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

static int utf8_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// ─── Tokenizer State ────────────────────────────────────────────────────────

struct ss_tokenizer {
    char    **vocab;        // Token strings
    float    *scores;       // BPE merge scores (SentencePiece)
    int32_t   vocab_size;
    int32_t   bos_id;
    int32_t   eos_id;
    char      arch[64];
    
    // BPE merge rules (Tiktoken/GPT-2 models)
    char    **merges;       // Merge rule strings ("token1 token2")
    int32_t   n_merges;     // Number of merge rules
    bool      is_bpe;       // true = use merge ranks, false = use scores
};

// ─── GPT-2 byte_to_unicode mapping ──────────────────────────────────────────
// Matches llama.cpp unicode.cpp unicode_byte_to_utf8_map() exactly.
// Printable ASCII + certain Latin-1 bytes map to themselves.
// All other bytes map to U+0100..U+0143 sequentially.

static bool gpt2_byte_is_raw(uint8_t b) {
    if (b >= 0x21 && b <= 0x7E) return true;  // '!' to '~'
    if (b >= 0xA1 && b <= 0xAC) return true;  // '¡' to '¬'
    if (b >= 0xAE && b <= 0xFF) return true;  // '®' to 'ÿ'
    return false;
}

// Build the GPT-2 byte->codepoint mapping table (called once)
static void gpt2_build_byte_map(uint32_t map[256]) {
    int n = 0;
    // First pass: mark raw bytes
    for (int b = 0; b < 256; b++) {
        if (gpt2_byte_is_raw((uint8_t)b)) {
            map[b] = (uint32_t)b;
        } else {
            map[b] = 0xFFFFFFFF; // placeholder
        }
    }
    // Second pass: assign 256+n to non-raw bytes
    for (int b = 0; b < 256; b++) {
        if (map[b] == 0xFFFFFFFF) {
            map[b] = 256 + n;
            n++;
        }
    }
}

// Encode a single codepoint as UTF-8 into buf, return bytes written
static int cpt_to_utf8(uint32_t cpt, char *buf) {
    if (cpt <= 0x7F) {
        buf[0] = (char)cpt;
        return 1;
    }
    if (cpt <= 0x7FF) {
        buf[0] = (char)(0xC0 | ((cpt >> 6) & 0x1F));
        buf[1] = (char)(0x80 | (cpt & 0x3F));
        return 2;
    }
    if (cpt <= 0xFFFF) {
        buf[0] = (char)(0xE0 | ((cpt >> 12) & 0x0F));
        buf[1] = (char)(0x80 | ((cpt >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cpt & 0x3F));
        return 3;
    }
    buf[0] = (char)(0xF0 | ((cpt >> 18) & 0x07));
    buf[1] = (char)(0x80 | ((cpt >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cpt >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cpt & 0x3F));
    return 4;
}

// ─── GGUF Tokenizer Data Reader ──────────────────────────────────────────────

ss_tokenizer_t *ss_tokenizer_create(char **vocab, float *scores, uint32_t vocab_size,
                                     char **merges, uint32_t n_merges,
                                     const char *arch) {
    ss_tokenizer_t *tok = (ss_tokenizer_t *)calloc(1, sizeof(ss_tokenizer_t));
    if (!tok) return NULL;

    // Store architecture
    if (arch) {
        strncpy(tok->arch, arch, sizeof(tok->arch) - 1);
    }
    
    // Determine if this is a BPE (Tiktoken) model
    tok->is_bpe = false;
    if (strncmp(tok->arch, "qwen2", 5) == 0 || strncmp(tok->arch, "qwen35", 6) == 0 || strncmp(tok->arch, "llama3", 6) == 0) {
        tok->is_bpe = true;
    }
    
    // Defaults — these would be read from GGUF KV metadata
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->vocab_size = vocab_size;

    if (vocab_size > 0 && vocab) {
        tok->vocab = (char **)calloc(vocab_size, sizeof(char *));
        tok->scores = (float *)calloc(vocab_size, sizeof(float));
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (vocab[i]) {
                tok->vocab[i] = strdup(vocab[i]);
            }
            if (scores) {
                tok->scores[i] = scores[i];
            } else {
                // If no scores exist, fallback to string-length based score
                tok->scores[i] = (float)strlen(vocab[i] ? vocab[i] : "");
            }
        }
    }
    
    // Store BPE merge rules
    tok->n_merges = (int32_t)n_merges;
    if (n_merges > 0 && merges) {
        tok->merges = (char **)calloc(n_merges, sizeof(char *));
        for (uint32_t i = 0; i < n_merges; i++) {
            if (merges[i]) {
                tok->merges[i] = strdup(merges[i]);
            }
        }
        tok->is_bpe = true; // If we have merges, this is definitely a BPE model
    }

    return tok;
}

// ─── Encode ──────────────────────────────────────────────────────────────────

int32_t ss_tokenizer_encode(const ss_tokenizer_t *tok,
                             const char *text,
                             int32_t *output,
                             int32_t max_tokens) {
    if (!tok || !text || !output || max_tokens <= 0) return 0;

    int32_t count = 0;

    if (!tok->vocab || tok->vocab_size == 0) {
        // Fallback byte-encoding
        const char *p = text;
        while (*p && count < max_tokens) {
            output[count++] = (int32_t)(unsigned char)(*p) + 3;
            p++;
        }
        return count;
    }

    // ── Prepare text for tokenization ──
    // For SentencePiece: convert spaces to U+2581 (▁)
    // For BPE/Tiktoken: convert each byte through GPT-2 byte_to_unicode mapping
    
    char sp_text[8192] = {0};
    int sp_idx = 0;
    
    if (tok->is_bpe) {
        // GPT-2 / Tiktoken byte mapping
        uint32_t byte_map[256];
        gpt2_build_byte_map(byte_map);
        
        for (int i = 0; text[i] != '\0' && sp_idx < 8000; i++) {
            uint8_t b = (uint8_t)text[i];
            uint32_t cpt = byte_map[b];
            sp_idx += cpt_to_utf8(cpt, sp_text + sp_idx);
        }
    } else {
        // SentencePiece: prefix with ▁ and convert spaces to ▁
        sp_text[sp_idx++] = (char)0xE2;
        sp_text[sp_idx++] = (char)0x96;
        sp_text[sp_idx++] = (char)0x81;
        
        for (int i = 0; text[i] != '\0' && sp_idx < 8000; i++) {
            if (text[i] == ' ') {
                sp_text[sp_idx++] = (char)0xE2;
                sp_text[sp_idx++] = (char)0x96;
                sp_text[sp_idx++] = (char)0x81;
            } else {
                sp_text[sp_idx++] = text[i];
            }
        }
    }
    
    // ── Initial tokenization: split into single characters/tokens ──
    int32_t tokens[4096];
    int n_tokens = 0;
    
    // SentencePiece models start with BOS
    if (!tok->is_bpe) {
        tokens[n_tokens++] = tok->bos_id;
    }

    const char *p = sp_text;
    while (*p && n_tokens < 4000) {
        // Dynamically match multi-character special tokens (like <|im_start|> or [INST])
        int best_special_id = -1;
        int best_special_len = -1;
        
        // Handle SentencePiece artificial ▁ prefixes mathematically before Special Tokens
        int sp_offset = 0;
        if (p[0] == (char)0xE2 && p[1] == (char)0x96 && p[2] == (char)0x81 && 
            (p[3] == '<' || p[3] == '[')) {
            sp_offset = 3;
        } else if (p[0] == '<' || p[0] == '[') {
            sp_offset = 0;
        } else {
            sp_offset = -1;
        }
        
        if (sp_offset >= 0) {
            const char *token_start = p + sp_offset;
            for (int v = 0; v < tok->vocab_size; v++) {
                if (!tok->vocab[v]) continue;
                int vlen = (int)strlen(tok->vocab[v]);
                if (vlen > 2 && tok->vocab[v][0] == token_start[0]) {
                    if ((token_start[0] == '<' && tok->vocab[v][vlen-1] == '>') || 
                        (token_start[0] == '[' && tok->vocab[v][vlen-1] == ']')) {
                        if (strncmp(token_start, tok->vocab[v], vlen) == 0) {
                            if (vlen > best_special_len) {
                                best_special_len = vlen;
                                best_special_id = v;
                            }
                        }
                    }
                }
            }
        }
        
        if (best_special_id >= 0) {
            tokens[n_tokens++] = best_special_id;
            // Advance cursor skipping both the ▁ prefix explicitly if it natively existed and the token entirely!
            p += sp_offset + best_special_len;
            continue;
        }

        // Get UTF-8 character length
        int clen = utf8_len((unsigned char)*p);
        char char_buf[8] = {0};
        for (int i = 0; i < clen && p[i]; i++) {
            char_buf[i] = p[i];
        }
        
        // Look up this character in the vocabulary
        int best_id = -1;
        for (int i = 0; i < tok->vocab_size; i++) {
            if (!tok->vocab[i]) continue;
            if (strcmp(tok->vocab[i], char_buf) == 0) {
                best_id = i;
                break;
            }
        }
        
        if (best_id >= 0) {
            tokens[n_tokens++] = best_id;
        } else {
            // Byte fallback for unknown bytes
            for (int i = 0; i < clen && p[i]; i++) {
                tokens[n_tokens++] = (int32_t)(unsigned char)(p[i]) + 3;
            }
        }
        p += clen;
    }
    
    // ── BPE Merge Loop ──
    if (tok->is_bpe && tok->merges && tok->n_merges > 0) {
        // Rank-based BPE: iterate through merges in order (lowest rank = highest priority)
        // Each merge rule is "token1 token2" — we merge the first occurrence found
        
        for (int32_t rank = 0; rank < tok->n_merges; rank++) {
            if (n_tokens <= 1) break;
            
            const char *merge_rule = tok->merges[rank];
            if (!merge_rule) continue;
            
            // Parse merge rule: "token1 token2" split on first space
            const char *space = strchr(merge_rule, ' ');
            if (!space) continue;
            
            int len1 = (int)(space - merge_rule);
            const char *part2 = space + 1;
            int len2 = (int)strlen(part2);
            
            // Build the merged string to find its vocab ID
            char merged[512] = {0};
            memcpy(merged, merge_rule, len1);
            memcpy(merged + len1, part2, len2);
            merged[len1 + len2] = '\0';
            
            // Find the merged token's ID in vocab
            int merged_id = -1;
            for (int v = 0; v < tok->vocab_size; v++) {
                if (!tok->vocab[v]) continue;
                if (strcmp(tok->vocab[v], merged) == 0) {
                    merged_id = v;
                    break;
                }
            }
            if (merged_id == -1) continue;
            
            // Scan for adjacent pairs matching this merge rule
            bool found = true;
            while (found) {
                found = false;
                for (int i = 0; i < n_tokens - 1; i++) {
                    // Get string representations of adjacent tokens
                    const char *str1 = (tokens[i] >= 0 && tokens[i] < tok->vocab_size && tok->vocab[tokens[i]]) 
                                       ? tok->vocab[tokens[i]] : NULL;
                    const char *str2 = (tokens[i+1] >= 0 && tokens[i+1] < tok->vocab_size && tok->vocab[tokens[i+1]]) 
                                       ? tok->vocab[tokens[i+1]] : NULL;
                    
                    if (!str1 || !str2) continue;
                    
                    // Check if this pair matches the merge rule
                    if ((int)strlen(str1) == len1 && (int)strlen(str2) == len2 &&
                        strncmp(str1, merge_rule, len1) == 0 &&
                        strcmp(str2, part2) == 0) {
                        // Merge!
                        tokens[i] = merged_id;
                        for (int j = i + 1; j < n_tokens - 1; j++) {
                            tokens[j] = tokens[j + 1];
                        }
                        n_tokens--;
                        found = true;
                        break; // Restart scan from beginning for this rank
                    }
                }
            }
        }
    } else {
        // Score-based BPE (SentencePiece): merge highest-scoring pairs first
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; i++) {
                char pair_buf[256];
                
                const char *str1 = tok->vocab[tokens[i]] ? tok->vocab[tokens[i]] : "";
                const char *str2 = tok->vocab[tokens[i+1]] ? tok->vocab[tokens[i+1]] : "";
                
                char fallback_str1[2] = {0};
                char fallback_str2[2] = {0};
                
                if (tokens[i] >= 3 && tokens[i] < 259) {
                    fallback_str1[0] = (char)(tokens[i] - 3);
                    str1 = fallback_str1;
                }
                if (tokens[i+1] >= 3 && tokens[i+1] < 259) {
                    fallback_str2[0] = (char)(tokens[i+1] - 3);
                    str2 = fallback_str2;
                }
                
                snprintf(pair_buf, sizeof(pair_buf), "%s%s", str1, str2);
                
                int id = -1;
                for (int v = 0; v < tok->vocab_size; v++) {
                    if (!tok->vocab[v]) continue;
                    if (strcmp(tok->vocab[v], pair_buf) == 0) {
                        id = v;
                        break;
                    }
                }
                
                if (id != -1 && tok->scores[id] > best_score) {
                    best_score = tok->scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break;
            }

            tokens[best_idx] = best_id;
            for (int i = best_idx + 1; i < n_tokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            n_tokens--;
        }
    }
    
    // Copy to output
    for (int i = 0; i < n_tokens && count < max_tokens; i++) {
        output[count++] = tokens[i];
    }
    
    return count;
}

// ─── Decode ──────────────────────────────────────────────────────────────────

static int ss_hex_char_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    return -1;
}

const char *ss_tokenizer_decode(const ss_tokenizer_t *tok, int32_t token_id) {
    if (!tok) return "";

    // Special tokens — return empty so they don't print
    if (token_id == tok->bos_id) return "";
    if (token_id == tok->eos_id) return "";
    // <unk> token
    if (token_id == 0) return "";

    // Vocabulary lookup
    if (tok->vocab && token_id >= 0 && token_id < tok->vocab_size) {
        static __thread char clean_buf[512];
        const char *raw = tok->vocab[token_id];
        if (!raw) return "";

        // Handle SentencePiece byte fallback tokens: <0xHH>
        if (raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' &&
            raw[5] == '>' && raw[6] == '\0') {
            int hi = ss_hex_char_to_int(raw[3]);
            int lo = ss_hex_char_to_int(raw[4]);
            if (hi >= 0 && lo >= 0) {
                static __thread char hex_byte_buf[2];
                hex_byte_buf[0] = (char)((hi << 4) | lo);
                hex_byte_buf[1] = '\0';
                return hex_byte_buf;
            }
        }

        if (tok->is_bpe) {
            // GPT-2 / Tiktoken decode: reverse the byte_to_unicode mapping
            // Codepoints in U+0100..U+0143 map back to the original non-printable bytes
            uint32_t byte_map[256];
            gpt2_build_byte_map(byte_map);
            
            int out_idx = 0;
            int in_idx = 0;
            while (raw[in_idx] && out_idx < 500) {
                // Decode UTF-8 codepoint
                unsigned char c = (unsigned char)raw[in_idx];
                uint32_t cpt;
                int clen;
                if ((c & 0x80) == 0) {
                    cpt = c;
                    clen = 1;
                } else if ((c & 0xE0) == 0xC0) {
                    cpt = ((c & 0x1F) << 6) | (raw[in_idx+1] & 0x3F);
                    clen = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    cpt = ((c & 0x0F) << 12) | ((raw[in_idx+1] & 0x3F) << 6) | (raw[in_idx+2] & 0x3F);
                    clen = 3;
                } else {
                    cpt = ((c & 0x07) << 18) | ((raw[in_idx+1] & 0x3F) << 12) | ((raw[in_idx+2] & 0x3F) << 6) | (raw[in_idx+3] & 0x3F);
                    clen = 4;
                }
                
                // Reverse map: find which byte maps to this codepoint
                bool mapped = false;
                for (int b = 0; b < 256; b++) {
                    if (byte_map[b] == cpt) {
                        clean_buf[out_idx++] = (char)(uint8_t)b;
                        mapped = true;
                        break;
                    }
                }
                if (!mapped) {
                    // Not in the byte map — write the raw UTF-8 bytes
                    for (int j = 0; j < clen && out_idx < 500; j++) {
                        clean_buf[out_idx++] = raw[in_idx + j];
                    }
                }
                in_idx += clen;
            }
            clean_buf[out_idx] = '\0';
            return clean_buf;
        } else {
            // SentencePiece decode: convert U+2581 (▁) back to space
            int out_idx = 0;
            int in_idx = 0;
            while (raw[in_idx] && out_idx < 500) {
                if ((unsigned char)raw[in_idx] == 0xE2 &&
                    (unsigned char)raw[in_idx+1] == 0x96 &&
                    (unsigned char)raw[in_idx+2] == 0x81) {
                    clean_buf[out_idx++] = ' ';
                    in_idx += 3;
                } else {
                    clean_buf[out_idx++] = raw[in_idx++];
                }
            }
            clean_buf[out_idx] = '\0';
            return clean_buf;
        }
    }

    // Byte fallback (legacy range)
    static __thread char byte_buf[2];
    if (token_id >= 3 && token_id < 259) {
        byte_buf[0] = (char)(token_id - 3);
        byte_buf[1] = '\0';
        return byte_buf;
    }

    return "";
}

// ─── Special Tokens ──────────────────────────────────────────────────────────

int32_t ss_tokenizer_bos_id(const ss_tokenizer_t *tok) {
    return tok ? tok->bos_id : 1;
}

int32_t ss_tokenizer_eos_id(const ss_tokenizer_t *tok) {
    return tok ? tok->eos_id : 2;
}

int32_t ss_tokenizer_vocab_size(const ss_tokenizer_t *tok) {
    return tok ? tok->vocab_size : 0;
}

// ─── Free ────────────────────────────────────────────────────────────────────

void ss_tokenizer_free(ss_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->vocab) {
        for (int32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->scores);
    
    if (tok->merges) {
        for (int32_t i = 0; i < tok->n_merges; i++) {
            free(tok->merges[i]);
        }
        free(tok->merges);
    }
    
    free(tok);
}
