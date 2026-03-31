import XCTest
@testable import SlipStream

final class SlipStreamTests: XCTestCase {
    
    func testModelLoadFailure() throws {
        // Test that loading a non-existent model throws an error
        XCTAssertThrowsError(try SlipStreamModel(path: "nonexistent.gguf")) { error in
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "SlipStream")
            // SS_ERROR_FILE_NOT_FOUND is likely -2 according to test output
            XCTAssertEqual(nsError.code, -2)
        }
    }
    
    func testAsyncGeneration() async throws {
        // Use an absolute path for testing
        let modelPath = "/Users/harrisonfarrell/Desktop/IB/embeddingphoine/models/tinyllama-q4_0.gguf"
        
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: modelPath) else {
            print("Skipping generation test: model not found at \(modelPath)")
            return
        }
        
        var config = SlipStreamConfig()
        config.maxTokens = 5
        let model = try SlipStreamModel(path: modelPath, config: config)
        
        var receivedTokens = 0
        do {
            let stream = model.generate(prompt: "Hello", params: config)
            for try await token in stream {
                print("Token: \(token)")
                receivedTokens += 1
            }
        } catch {
            XCTFail("Generation threw error: \(error)")
        }
        
        XCTAssertGreaterThan(receivedTokens, 0)
    }
}
