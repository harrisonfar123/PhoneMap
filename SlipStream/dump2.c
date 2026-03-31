#include "src/core/gguf_loader.h"
#include <stdio.h>
#include <string.h>

int main() {
    ss_gguf_file_t *gguf = ss_gguf_open("/Users/harrisonfarrell/Library/Developer/CoreSimulator/Devices/F63131EA-7456-41B8-B5F6-103559E63876/data/Containers/Shared/AppGroup/E61A30A0-4B7C-450C-9D2B-291B0CB92F26/File Provider Storage/Qwen3.5-4B-Q4_K_M.gguf");
    if (!gguf) { printf("Failed\n"); return 1; }
    
    for(int i=0; i < gguf->tensor_count; i++) {
        if (strncmp(gguf->tensors[i].name, "blk.0.", 6) == 0) {
            printf("%s: type=%d, n_dims=%d (dims[0]=%llu, dims[1]=%llu)\n", gguf->tensors[i].name, gguf->tensors[i].type, gguf->tensors[i].n_dims, gguf->tensors[i].dims[0], gguf->tensors[i].dims[1]);
        }
    }
    ss_gguf_close(gguf);
    return 0;
}
