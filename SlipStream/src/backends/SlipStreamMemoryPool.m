#import "SlipStreamMemoryPool.h"
#import <pthread.h>

#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

@implementation SlipStreamMemoryPool {
    id<MTLDevice> _device;
    size_t _maxLayerSize;
    size_t _maxActivationSize;
    
    id<MTLBuffer> _buffers[2];
    BOOL _bufferInUse[2];
    
    pthread_mutex_t _mutex;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
            maxLayerWeightSize:(size_t)maxLayerSize
             maxActivationSize:(size_t)maxActivationSize {
    self = [super init];
    if (self) {
        _device = device;
        size_t pageSize = getpagesize();
        _maxLayerSize = ALIGN_UP(maxLayerSize, pageSize);
        _maxActivationSize = ALIGN_UP(maxActivationSize, pageSize);
        
        pthread_mutex_init(&_mutex, NULL);
        
        // Double buffering: Pre-allocate 2 weight buffers
        MTLResourceOptions options = MTLResourceStorageModeShared;
        _buffers[0] = [_device newBufferWithLength:_maxLayerSize options:options];
        _buffers[1] = [_device newBufferWithLength:_maxLayerSize options:options];
        _bufferInUse[0] = NO;
        _bufferInUse[1] = NO;
    }
    return self;
}

- (void)dealloc {
    [self drainAllPools];
    pthread_mutex_destroy(&_mutex);
}

- (nullable id<MTLBuffer>)acquireWeightBufferForSize:(size_t)size {
    pthread_mutex_lock(&_mutex);
    id<MTLBuffer> allocatedBuffer = nil;
    
    for (int i = 0; i < 2; i++) {
        if (!_bufferInUse[i] && _buffers[i]) {
            if (_buffers[i].length >= size) {
                _bufferInUse[i] = YES;
                allocatedBuffer = _buffers[i];
                break;
            }
        }
    }
    pthread_mutex_unlock(&_mutex);
    
    // If double buffers are in use or too small, fall back (though ideal state is using the pre-allocated pool)
    if (!allocatedBuffer) {
        size_t pageSize = getpagesize();
        size_t alignedSize = ALIGN_UP(size, pageSize);
        allocatedBuffer = [_device newBufferWithLength:alignedSize options:MTLResourceStorageModeShared];
    }
    
    return allocatedBuffer;
}

- (void)releaseWeightBuffer:(id<MTLBuffer>)buffer {
    if (!buffer) return;
    
    pthread_mutex_lock(&_mutex);
    BOOL returnedToPool = NO;
    for (int i = 0; i < 2; i++) {
        if (_buffers[i] == buffer) {
            _bufferInUse[i] = NO;
            returnedToPool = YES;
            break;
        }
    }
    pthread_mutex_unlock(&_mutex);
    
    if (!returnedToPool) {
        // Was dynamically allocated, let ARC deallocate
    }
}

- (void)drainAllPools {
    pthread_mutex_lock(&_mutex);
    for (int i = 0; i < 2; i++) {
        if (!_bufferInUse[i]) {
            _buffers[i] = nil;
        }
    }
    pthread_mutex_unlock(&_mutex);
}

@end
