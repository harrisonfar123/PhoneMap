import Foundation
import CoreImage
import Vision
import SlipStream
import Accelerate

public protocol MotimodelProvider {
    func embed(text: String) async throws -> [Float]
    func embed(image: CGImage) async throws -> (description: String, vector: [Float])
}

/// The local edge unified motimodel wrapper for the `VectorLens` database.
/// This uses native Apple Vision framework to map physical images into english strings, 
/// invoking Nomic SlipStream to perfectly unite everything into language-space vectors!
public class LocalMotimodelEngine: MotimodelProvider {
    private let textEngine: SlipStreamModel
    
    public init(textEngine: SlipStreamModel) {
        self.textEngine = textEngine
    }
    
    public func embed(text: String) async throws -> [Float] {
        return try await textEngine.embed(prompt: text)
    }
    
    public func embed(image: CGImage) async throws -> (description: String, vector: [Float]) {
        // "Translation by Alignment"
        // Generate an English description natively so `nomic` understands it mathematically!
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        
        // 1. Apple On-Device ML (Works 100% on Simulators without hacking computeUnits)
        let classificationRequest = VNClassifyImageRequest()
        let textRequest = VNRecognizeTextRequest()
        textRequest.recognitionLevel = .accurate
        
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([classificationRequest, textRequest])
                    
                    var descriptions: [String] = []
                    
                    // Filter High-Confidence Object Detections (e.g. "Cat", "Beach")
                    if let results = classificationRequest.results {
                        let topTags = results
                            .filter { $0.confidence > 0.1 } // Relaxed to sweep Apple taxonomy!
                            .prefix(10)
                            .map { $0.identifier.replacingOccurrences(of: "_", with: " ") }
                        
                        if !topTags.isEmpty {
                            descriptions.append("Photo depicting " + topTags.joined(separator: ", "))
                        }
                    }
                    
                    // Filter Raw OCR Text in the image
                    if let textResults = textRequest.results {
                        let topWords = textResults
                            .prefix(15) // Max 15 lines of text
                            .compactMap { $0.topCandidates(1).first?.string }
                        
                        if !topWords.isEmpty {
                            descriptions.append("Contains visible text: '" + topWords.joined(separator: " ") + "'")
                        }
                    }
                    
                    let finalSentence = descriptions.isEmpty ? "A blank or unidentifiable photo." : descriptions.joined(separator: ". ")
                    
                    // Task bounce-back to await the C-Engine!
                    Task {
                        do {
                            let vector = try await self.textEngine.embed(prompt: finalSentence)
                            continuation.resume(returning: (description: finalSentence, vector: vector))
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
}
