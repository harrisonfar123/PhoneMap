#include "src/core/gguf_loader.h"
#include <stdio.h>
#include <string.h>

int main() {
    ss_gguf_file_t *gguf = ss_gguf_open("/Users/harrisonfarrell/Library/Developer/CoreSimulator/Devices/F63131EA-7456-41B8-B5F6-103559E63876/data/Containers/Shared/AppGroup/E61A30A0-4B7C-450C-9D2B-291B0CB92F26/File Provider Storage/Qwen3.5-4B-Q4_K_M.gguf");
    if (!gguf) { printf("Failed\n"); return 1; }
    
    printf("RoPE: %u\n", gguf->rope_dim);
    ss_gguf_close(gguf);
    return 0;
}
