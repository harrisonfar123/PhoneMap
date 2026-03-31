import Foundation
import Metal

public enum BufferValidationError: Error {
    case bufferTooSmall(required: Int, actual: Int, bufferName: String)
    case offsetUnaligned(offset: Int, requiredAlignment: Int)
    case offsetExceedsBuffer(offset: Int, bufferLength: Int)
    case nullBuffer(bufferName: String)
}

public struct BufferValidator {
    public static func validateBuffer(
        _ buffer: MTLBuffer?,
        requiredLength: Int,
        offset: Int = 0,
        name: String
    ) throws {
        guard let buffer = buffer else {
            throw BufferValidationError.nullBuffer(bufferName: name)
        }
        
        // CRITICAL: 256-byte alignment required for compute
        let alignment = 256
        guard offset % alignment == 0 else {
            throw BufferValidationError.offsetUnaligned(offset: offset, requiredAlignment: alignment)
        }
        
        guard offset < buffer.length else {
            throw BufferValidationError.offsetExceedsBuffer(offset: offset, bufferLength: buffer.length)
        }
        
        let availableSpace = buffer.length - offset
        guard availableSpace >= requiredLength else {
            throw BufferValidationError.bufferTooSmall(
                required: requiredLength,
                actual: availableSpace,
                bufferName: name
            )
        }
    }
}

public class CPUGPUSynchronizer {
    private let semaphore: DispatchSemaphore
    
    public init(maxInflightBuffers: Int = 3) {
        self.semaphore = DispatchSemaphore(value: maxInflightBuffers)
    }
    
    public func commitWithFlowControl(commandBuffer: MTLCommandBuffer) {
        semaphore.wait()
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.semaphore.signal()
        }
        commandBuffer.commit()
    }
}
