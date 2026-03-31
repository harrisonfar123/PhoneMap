#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int main() {
    FILE *f = fopen("/Users/harrisonfarrell/Library/Developer/CoreSimulator/Devices/F63131EA-7456-41B8-B5F6-103559E63876/data/Containers/Shared/AppGroup/E61A30A0-4B7C-450C-9D2B-291B0CB92F26/File Provider Storage/Qwen3.5-4B-Q4_K_M.gguf", "rb");
    if (!f) {
        printf("Could not open model.\n");
        return 0;
    }
    printf("Model opened.\n");
    fclose(f);
    return 0;
}
