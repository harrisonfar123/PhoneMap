#import "SlipStreamGGUFMapper.h"
#import <fcntl.h>
#import <unistd.h>

@interface SlipStreamMappedRegion ()
@property (nonatomic, readwrite) void *baseAddress;
@property (nonatomic, readwrite) size_t length;
@property (nonatomic, readwrite) off_t fileOffset;
@property (nonatomic, readwrite) size_t mappedLength;
@property (nonatomic, readwrite) void *mappedAddress;
@end

@implementation SlipStreamMappedRegion
@end

@implementation SlipStreamGGUFMapper {
    int _fd;
    size_t _fileSize;
    size_t _pageSize;
}

- (nullable instancetype)initWithFilePath:(NSString *)filePath error:(NSError **)error {
    self = [super init];
    if (self) {
        _fd = open([filePath UTF8String], O_RDONLY);
        if (_fd < 0) {
            if (error) {
                *error = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:nil];
            }
            return nil;
        }
        
        struct stat st;
        if (fstat(_fd, &st) < 0) {
            close(_fd);
            if (error) {
                *error = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:nil];
            }
            return nil;
        }
        _fileSize = st.st_size;
        _pageSize = getpagesize();
    }
    return self;
}

- (void)dealloc {
    if (_fd >= 0) {
        close(_fd);
    }
}

- (nullable SlipStreamMappedRegion *)mapTensorAtOffset:(off_t)offset
                                                length:(size_t)length
                                                 error:(NSError **)error {
    if (offset + length > _fileSize) {
        if (error) {
            *error = [NSError errorWithDomain:@"SlipStream" code:-1 userInfo:@{NSLocalizedDescriptionKey: @"Offset and length exceed file size"}];
        }
        return nil;
    }
    
    off_t alignedOffset = offset & ~(_pageSize - 1);
    size_t offsetAdjustment = offset - alignedOffset;
    size_t mappedLength = length + offsetAdjustment;
    
    void *mappedAddr = mmap(NULL, mappedLength, PROT_READ, MAP_SHARED, _fd, alignedOffset);
    if (mappedAddr == MAP_FAILED) {
        if (error) {
            *error = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:nil];
        }
        return nil;
    }
    
    SlipStreamMappedRegion *region = [[SlipStreamMappedRegion alloc] init];
    region.mappedAddress = mappedAddr;
    region.mappedLength = mappedLength;
    region.fileOffset = offset;
    region.length = length;
    region.baseAddress = (uint8_t *)mappedAddr + offsetAdjustment;
    
    return region;
}

- (void)unmapRegion:(SlipStreamMappedRegion *)region {
    if (region && region.mappedAddress) {
        munmap(region.mappedAddress, region.mappedLength);
        region.mappedAddress = NULL;
    }
}

- (void)adviseSequentialAccessForRegion:(SlipStreamMappedRegion *)region {
    if (region && region.mappedAddress) {
        madvise(region.mappedAddress, region.mappedLength, MADV_SEQUENTIAL);
    }
}

- (void)adviseWillNeedForRegion:(SlipStreamMappedRegion *)region {
    if (region && region.mappedAddress) {
        madvise(region.mappedAddress, region.mappedLength, MADV_WILLNEED);
    }
}

- (void)adviseDontNeedForRegion:(SlipStreamMappedRegion *)region {
    if (region && region.mappedAddress) {
        madvise(region.mappedAddress, region.mappedLength, MADV_DONTNEED);
    }
}

@end
