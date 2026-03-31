#import <Metal/Metal.h>
#import <mach/mach.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SlipStreamMemoryPool : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device
            maxLayerWeightSize:(size_t)maxLayerSize
             maxActivationSize:(size_t)maxActivationSize;

- (nullable id<MTLBuffer>)acquireWeightBufferForSize:(size_t)size;
- (void)releaseWeightBuffer:(id<MTLBuffer>)buffer;
- (void)drainAllPools;  // Emergency cleanup for memory warnings

@end

NS_ASSUME_NONNULL_END
