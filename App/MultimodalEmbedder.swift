import Foundation
import CoreImage
import CoreML
import Vision
import SlipStream
import Accelerate

public protocol MotimodelProvider {
    func embed(text: String) async throws -> [Float]
    func embed(image: CGImage) async throws -> [Float]
}

/// The local edge unified motimodel wrapper for the `VectorLens` database.
/// This acts as a facade, calling SlipStream for text constraints, and 
/// invoking a CoreML CLIP/Vision transformer model for pixel constraints.
public class LocalMotimodelEngine: MotimodelProvider {
    private let textEngine: SlipStreamModel
    private var visionModel: MLModel?
    
    public init(textEngine: SlipStreamModel) {
        self.textEngine = textEngine
        
        // Dynamically load CoreML if the user dropped the file into Xcode!
        if let url = Bundle.main.url(forResource: "CLIPVision", withExtension: "mlmodelc") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                self.visionModel = try MLModel(contentsOf: url, configuration: config)
                print("Motimodel: CLIPVision CoreML initialized!")
            } catch {
                print("Motimodel: Failed to bind CLIPVision: \(error)")
            }
        } else {
            print("Motimodel: CLIPVision not found. Falling back to dimensional stubs.")
        }
    }
    
    public func embed(text: String) async throws -> [Float] {
        // Our current Qwen/Nomic models natively spit out 768-D.
        // If we switch to CLIP later, it might be 512-D. DB is agnostic.
        return try await textEngine.embed(prompt: text)
    }
    
    public func embed(image: CGImage) async throws -> [Float] {
        guard let mlModel = visionModel else {
            // Unchanged fallback if no downloaded model exists yet
            let w = Float(image.width) / 2000.0; let h = Float(image.height) / 2000.0
            var fakeVector = [Float](repeating: 0.0, count: 768)
            fakeVector[0] = w; fakeVector[1] = h; fakeVector[10] = 0.5
            try await Task.sleep(nanoseconds: 10_000_000)
            return fakeVector
        }
        
        // The real Apple Neural Engine inference layer!
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let vnModel = try VNCoreMLModel(for: mlModel)
                let request = VNCoreMLRequest(model: vnModel) { request, error in
                    if let err = error {
                        continuation.resume(throwing: err)
                        return
                    }
                    guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                          let firstResult = results.first,
                          let multiArray = firstResult.featureValue.multiArrayValue else {
                        continuation.resume(throwing: NSError(domain: "Motimodel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Vision tensor unpack failed"]))
                        return
                    }
                    
                    let count = multiArray.count
                    var vector = [Float](repeating: 0, count: count)
                    
                    // Pointer binding to safely convert Tensor array directly into swift Float math vector
                    let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
                    for i in 0..<count {
                        vector[i] = ptr[i]
                    }
                    
                    continuation.resume(returning: vector)
                }
                
                request.imageCropAndScaleOption = .centerCrop
                
                let handler = VNImageRequestHandler(cgImage: image, options: [:])
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        try handler.perform([request])
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}
