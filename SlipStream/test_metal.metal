#include <metal_stdlib>
using namespace metal;

struct block_q4_0 {
    half d;
    uchar qs[16];
};

kernel void test_sz(device uint *out [[buffer(0)]]) {
    out[0] = sizeof(block_q4_0);
}
