#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint16_t scale;     // f16 scale factor
    uint8_t  qs[16];    // 32 x 4-bit quantized values (packed)
} ss_block_q4_0_t;

int main() {
    printf("Size: %zu\n", sizeof(ss_block_q4_0_t));
    return 0;
}
