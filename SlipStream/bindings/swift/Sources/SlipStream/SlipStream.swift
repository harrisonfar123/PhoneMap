import Foundation
import CSlipStream

public enum SlipStreamBackend {
    case auto
    case cpu
    case metal
}

public struct SlipStreamConfig {
    public var backend: SlipStreamBackend = .auto
    public var threads: Int = 0
    public var contextSize: Int = 0
    public var maxTokens: Int = 512
    public var temperature: Float = 0.7
    public var topP: Float = 0.9
    public var topK: Int = 40
    /// Maximum memory budget in bytes. 0 = no limit.
    public var memoryBudget: UInt64 = 0
    
    public init() {}
}

private class CallbackWrapper {
    let continuation: AsyncThrowingStream<String, Error>.Continuation
    init(_ continuation: AsyncThrowingStream<String, Error>.Continuation) {
        self.continuation = continuation
    }
}

/// Build appropriate prompt based on auto-detected chat format.
/// The format is detected by scanning the model's vocabulary for template tokens,
/// which is far more reliable than matching architecture strings.
private func buildPrompt(userMessage: String, chatFormat: String) -> String {
    switch chatFormat {
    case "chatml":
        // ChatML format — used by Qwen, Qwen2, Qwen3.5, Phi-3, StableLM, DeepSeek, SmolLM
        return """
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        \(userMessage)<|im_end|>
        <|im_start|>assistant
        """
        .split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: "\n")
        
    case "llama3":
        // Llama 3 / 3.1 / 3.2 format
        return "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>" +
               "<|start_header_id|>user<|end_header_id|>\n\n\(userMessage)<|eot_id|>" +
               "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
    case "gemma":
        // Gemma / Gemma 2 format
        return "<start_of_turn>user\n\(userMessage)<end_of_turn>\n<start_of_turn>model\n"
        
    case "mistral":
        // Mistral / Llama 2 instruct format
        return "[INST] \(userMessage) [/INST]"

    case "phi3":
        // Phi-3 format: <|user|>...<|end|>\n<|assistant|>
        return "<|user|>\n\(userMessage)<|end|>\n<|assistant|>\n"
        
    default:
        // "raw" fallback — no template wrapping, just send the prompt directly.
        // This works for base (non-chat) models and prevents nonsense from
        // wrapping in a template the model wasn't trained with.
        return userMessage
    }
}

public final class SlipStreamModel {
    private var model: OpaquePointer?
    public let architecture: String
    public let chatFormat: String
    
    public init(path: String, config: SlipStreamConfig = SlipStreamConfig()) throws {
        var cConfig = ss_default_config()
        switch config.backend {
        case .auto: cConfig.backend = SS_BACKEND_AUTO
        case .cpu: cConfig.backend = SS_BACKEND_CPU
        case .metal: cConfig.backend = SS_BACKEND_METAL
        }
        cConfig.n_threads = UInt32(config.threads)
        cConfig.context_size = UInt32(config.contextSize)
        cConfig.max_memory = config.memoryBudget
        
        self.model = ss_model_load(path, &cConfig)
        if self.model == nil {
            let errString = String(cString: ss_error_string(ss_last_error()))
            throw NSError(domain: "SlipStream", code: Int(ss_last_error().rawValue), 
                          userInfo: [NSLocalizedDescriptionKey: errString])
        }
        
        var info = ss_model_info_t()
        ss_model_get_info(self.model, &info)
        
        let archArray = Mirror(reflecting: info.architecture).children.map { $0.value as! CChar }
        self.architecture = String(cString: archArray)
        
        let formatArray = Mirror(reflecting: info.chat_format).children.map { $0.value as! CChar }
        self.chatFormat = String(cString: formatArray)
        
        print("SlipStream: Detected chat format '\(self.chatFormat)' for architecture '\(self.architecture)'")
    }
    
    deinit {
        if let m = model {
            ss_model_free(m)
        }
    }
    
    public func cancel() {
        if let m = model {
            ss_cancel(m)
        }
    }
    
    /// Set a per-token throttle delay in microseconds.
    /// Use to reduce CPU when backgrounded. Set to 0 for full speed.
    public func setThrottle(delayMicroseconds: UInt32) {
        if let m = model {
            ss_set_throttle(m, delayMicroseconds)
        }
    }
    
    public func generate(prompt: String, params: SlipStreamConfig = SlipStreamConfig()) -> AsyncThrowingStream<String, Error> {
        let maxTokens = UInt32(params.maxTokens)
        let temperature = params.temperature
        let topP = params.topP
        let topK = UInt32(params.topK)
        
        return AsyncThrowingStream { continuation in
            var cParams = ss_default_params()
            cParams.max_tokens = maxTokens
            cParams.temperature = temperature
            cParams.top_p = topP
            cParams.top_k = topK
            
            let wrapper = CallbackWrapper(continuation)
            let context = Unmanaged.passRetained(wrapper).toOpaque()
            
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self, let m = self.model else {
                    let w = Unmanaged<CallbackWrapper>.fromOpaque(context).takeRetainedValue()
                    w.continuation.finish(throwing: NSError(domain: "SlipStream", code: -1, userInfo: nil))
                    return
                }
                
                ss_set_progress_callback(m, { current, total, phasePtr, ctx in
                    guard let ctx = ctx, let phasePtr = phasePtr else { return }
                    let w = Unmanaged<CallbackWrapper>.fromOpaque(ctx).takeUnretainedValue()
                    let phase = String(cString: phasePtr)
                    w.continuation.yield("|PROGRESS|\(phase)|\(current)|\(total)")
                }, context)
                
                // Use auto-detected chat format instead of architecture guessing
                let formattedPrompt = buildPrompt(userMessage: prompt, chatFormat: self.chatFormat)
                
                let err = ss_generate(m, formattedPrompt, &cParams, { textPtr, _, ctx in
                    guard let ctx = ctx, let textPtr = textPtr else { return false }
                    let w = Unmanaged<CallbackWrapper>.fromOpaque(ctx).takeUnretainedValue()
                    let str = String(cString: textPtr)
                    if !str.isEmpty {
                        w.continuation.yield(str)
                    }
                    return true
                }, context)
                
                let w = Unmanaged<CallbackWrapper>.fromOpaque(context).takeRetainedValue()
                
                if err == SS_OK {
                    w.continuation.finish()
                } else if err == SS_ERROR_CANCELLED {
                    w.continuation.finish(throwing: CancellationError())
                } else {
                    let errString = String(cString: ss_error_string(err))
                    w.continuation.finish(throwing: NSError(domain: "SlipStream", code: Int(err.rawValue), 
                                                            userInfo: [NSLocalizedDescriptionKey: errString]))
                }
            }
        }
    }
    
    /// Generate a dense vector embedding for the given prompt.
    public func embed(prompt: String) async throws -> [Float] {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self, let m = self.model else {
                    continuation.resume(throwing: NSError(domain: "SlipStream", code: -1, userInfo: nil))
                    return
                }
                
                var outEmbedding: UnsafeMutablePointer<Float>? = nil
                var outDim: UInt32 = 0
                
                // Use raw prompt for embeddings (no chat templates)
                let err = ss_embed(m, prompt, &outEmbedding, &outDim)
                
                if err == SS_OK, let embPtr = outEmbedding {
                    let array = Array(UnsafeBufferPointer(start: embPtr, count: Int(outDim)))
                    continuation.resume(returning: array)
                } else {
                    let errString = String(cString: ss_error_string(err))
                    continuation.resume(throwing: NSError(domain: "SlipStream", code: Int(err.rawValue), 
                                                          userInfo: [NSLocalizedDescriptionKey: errString]))
                }
            }
        }
    }
    
    public static func getLastProfilePath() -> String? {
        if let pathPtr = ss_get_last_profile_path() {
            return String(cString: pathPtr)
        }
        return nil
    }
    
    public static func setProfileOutputDir(_ path: String) {
        path.withCString { cStr in
            ss_set_profile_output_dir(cStr)
        }
    }
    
    // MARK: - Memory Utilities
    
    /// Estimate peak memory for a model at a given context size.
    /// Returns bytes, or nil on failure.
    public static func estimateMemory(modelPath: String, contextSize: Int) -> UInt64? {
        var peakMemory: UInt64 = 0
        let err = ss_estimate_memory(modelPath, UInt32(contextSize), &peakMemory)
        return err == SS_OK ? peakMemory : nil
    }
    
    /// Get the current process resident memory usage in bytes.
    public static func getCurrentMemoryUsage() -> UInt64 {
        return ss_get_memory_usage()
    }
}
