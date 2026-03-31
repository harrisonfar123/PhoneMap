#include <metal_stdlib>
using namespace metal;

// Safe memory access helpers
template<typename T>
inline bool isValidIndex(device const T* buffer, uint index, uint bufferLength) {
    return index < (bufferLength / sizeof(T));
}

template<typename T>
inline T safeLoad(device const T* buffer, uint index, uint bufferLength, T defaultValue = T(0)) {
    if (isValidIndex(buffer, index, bufferLength)) {
        return buffer[index];
    }
    return defaultValue;
}

template<typename T>
inline void safeStore(device T* buffer, uint index, uint bufferLength, T value) {
    if (isValidIndex(buffer, index, bufferLength)) {
        buffer[index] = value;
    }
}

// Example kernel with bounds checking
kernel void safeMatrixMultiply(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint3& dimensions [[buffer(3)]],
    constant uint3& bufferLengths [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint M = dimensions.x;
    const uint N = dimensions.y;
    const uint K = dimensions.z;
    const uint row = gid.y;
    const uint col = gid.x;
    
    // Bounds check for output position
    if (row >= M || col >= N) {
        return;
    }
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        uint aIndex = row * K + k;
        uint bIndex = k * N + col;
        
        // Safe loads with bounds checking
        float a = safeLoad(matrixA, aIndex, bufferLengths.x, 0.0f);
        float b = safeLoad(matrixB, bIndex, bufferLengths.y, 0.0f);
        sum += a * b;
    }
    
    safeStore(result, row * N + col, bufferLengths.z, sum);
}
