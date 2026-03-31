#import "SlipStreamTensor.h"
#import <stdatomic.h>

@interface SlipStreamTensor ()
@property (nonatomic, strong, readwrite) id<MTLBuffer> buffer;
@property (nonatomic, strong) NSArray<NSNumber *> *shape;
@property (nonatomic, strong) NSArray<NSNumber *> *strides;
@property (nonatomic, assign) NSUInteger offset;
@property (nonatomic, assign) SlipStreamDataType dtype;
@end

@implementation SlipStreamTensor {
    atomic_int _refCount;
    BOOL _isValid;
}

+ (instancetype)tensorWithBuffer:(id<MTLBuffer>)buffer
                            shape:(NSArray<NSNumber *> *)shape
                          strides:(NSArray<NSNumber *> *)strides
                           offset:(NSUInteger)offset
                            dtype:(SlipStreamDataType)dtype {
    return [[self alloc] initWithBuffer:buffer shape:shape strides:strides offset:offset dtype:dtype];
}

- (instancetype)initWithBuffer:(id<MTLBuffer>)buffer
                         shape:(NSArray<NSNumber *> *)shape
                       strides:(NSArray<NSNumber *> *)strides
                        offset:(NSUInteger)offset
                         dtype:(SlipStreamDataType)dtype {
    self = [super init];
    if (self) {
        _buffer = buffer;
        _shape = shape;
        _strides = strides;
        _offset = offset;
        _dtype = dtype;
        
        atomic_init(&_refCount, 1);
        _isValid = YES;
    }
    return self;
}

- (SlipStreamTensor *)retainTensor {
    if (!self.isValid) return nil;
    atomic_fetch_add(&_refCount, 1);
    return self;
}

- (void)releaseTensor {
    int count = atomic_fetch_sub(&_refCount, 1);
    if (count == 1) { // That was the last reference
        _isValid = NO;
        self.buffer = nil;
        // In a real memory pool setup, we might return to SlipStreamMemoryPool here
    }
}

- (void)releaseAfterCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    if (!self.isValid) return;
    
    // Retain explicitly for the closure block to own the object effectively
    [self retainTensor];
    
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        [self releaseTensor];
    }];
}

- (BOOL)isValid {
    return _isValid && atomic_load(&_refCount) > 0;
}

@end
