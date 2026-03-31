#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, SlipStreamDataType) {
    SlipStreamDataTypeFloat32 = 0,
    SlipStreamDataTypeFloat16 = 1,
    SlipStreamDataTypeInt32 = 2
};

@interface SlipStreamTensor : NSObject

+ (instancetype)tensorWithBuffer:(id<MTLBuffer>)buffer
                            shape:(NSArray<NSNumber *> *)shape
                          strides:(NSArray<NSNumber *> *)strides
                           offset:(NSUInteger)offset
                            dtype:(SlipStreamDataType)dtype;

- (SlipStreamTensor *)retainTensor;
- (void)releaseTensor;
- (void)releaseAfterCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (BOOL)isValid;  // Use-after-free protection

@property (nonatomic, readonly) id<MTLBuffer> buffer;

@end

NS_ASSUME_NONNULL_END
