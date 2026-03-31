#import <Foundation/Foundation.h>
#import <sys/mman.h>
#import <sys/stat.h>

NS_ASSUME_NONNULL_BEGIN

@interface SlipStreamMappedRegion : NSObject
@property (nonatomic, readonly) void *baseAddress;
@property (nonatomic, readonly) size_t length;
@property (nonatomic, readonly) off_t fileOffset;
@end

@interface SlipStreamGGUFMapper : NSObject

- (nullable instancetype)initWithFilePath:(NSString *)filePath error:(NSError **)error;

- (nullable SlipStreamMappedRegion *)mapTensorAtOffset:(off_t)offset
                                                length:(size_t)length
                                                 error:(NSError **)error;

- (void)unmapRegion:(SlipStreamMappedRegion *)region;

- (void)adviseSequentialAccessForRegion:(SlipStreamMappedRegion *)region;
- (void)adviseWillNeedForRegion:(SlipStreamMappedRegion *)region;  // Prefetch
- (void)adviseDontNeedForRegion:(SlipStreamMappedRegion *)region;  // Release

@end

NS_ASSUME_NONNULL_END
