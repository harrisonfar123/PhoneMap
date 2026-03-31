#include <stdio.h>
#include <stdlib.h>
#include "src/core/gguf_loader.h"

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    ss_gguf_file_t *f = ss_gguf_open(argv[1]);
    if (!f) {
        printf("Failed to load\n");
        return 1;
    }

    printf("Vocab Size: %u\n", f->vocab_size);
    
    // find tokenizer.ggml.tokens in metadata
    // wait I cant easily extract arr_len without patching gguf_loader to print it.
    
    ss_gguf_close(f);
    return 0;
}
