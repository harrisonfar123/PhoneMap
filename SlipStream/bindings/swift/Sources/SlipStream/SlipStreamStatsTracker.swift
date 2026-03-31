import Foundation
import CSlipStream
import Combine

/// Observable stat tracker that polls system metrics at ~1Hz during inference.
/// Attach to SwiftUI views to show live memory, thermal, CPU, and token throughput.
public final class SlipStreamStatsTracker: ObservableObject {
    
    // MARK: - Published Stats
    
    /// Current process resident memory in bytes
    @Published public private(set) var memoryUsedBytes: UInt64 = 0
    
    /// Available system memory in bytes
    @Published public private(set) var memoryAvailableBytes: UInt64 = 0
    
    /// User-configured memory budget in bytes
    @Published public var memoryBudgetBytes: UInt64 = 0
    
    /// Total device RAM in bytes
    @Published public private(set) var totalMemoryBytes: UInt64 = 0
    
    /// Memory usage as fraction of budget (0.0 - 1.0+)
    @Published public private(set) var memoryUsageFraction: Double = 0
    
    /// Current thermal state
    @Published public private(set) var thermalState: DeviceInfo.ThermalLevel = .nominal
    
    /// Process CPU usage percentage
    @Published public private(set) var cpuUsagePercent: Double = 0
    
    /// Tokens generated in current session
    @Published public private(set) var tokensGenerated: Int = 0
    
    /// Current tokens per second
    @Published public private(set) var tokensPerSecond: Double = 0
    
    /// Time elapsed since inference started (seconds)
    @Published public private(set) var elapsedSeconds: TimeInterval = 0
    
    /// Whether inference is currently running
    @Published public private(set) var isRunning: Bool = false
    
    /// Recent tokens/sec history for sparkline (last 30 samples)
    @Published public private(set) var speedHistory: [Double] = []
    
    /// Estimated remaining time (seconds)
    @Published public private(set) var estimatedRemainingSeconds: TimeInterval? = nil
    
    /// Max tokens for current generation (used to calculate ETA)
    public var maxTokens: Int = 0
    
    // MARK: - Private
    
    private var pollTimer: Timer?
    private var inferenceStartTime: Date?
    private var lastTokenCount: Int = 0
    private var lastTokenCheckTime: Date?
    private let maxHistoryLength = 30
    
    public init() {
        // Get total memory once
        if let device = DeviceInfo.current() {
            totalMemoryBytes = device.totalMemoryBytes
        }
    }
    
    // MARK: - Control
    
    /// Start polling stats. Call when inference begins.
    public func startTracking() {
        isRunning = true
        tokensGenerated = 0
        tokensPerSecond = 0
        elapsedSeconds = 0
        speedHistory = []
        inferenceStartTime = Date()
        lastTokenCheckTime = Date()
        lastTokenCount = 0
        
        pollOnce() // immediate first sample
        
        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.pollOnce()
        }
    }
    
    /// Stop polling. Call when inference completes.
    public func stopTracking() {
        isRunning = false
        pollTimer?.invalidate()
        pollTimer = nil
    }
    
    /// Record a generated token. Call from the token callback.
    public func recordToken() {
        tokensGenerated += 1
    }
    
    // MARK: - Polling
    
    private func pollOnce() {
        // Memory
        memoryUsedBytes = ss_get_memory_usage()
        
        if let device = DeviceInfo.current() {
            memoryAvailableBytes = device.availableMemoryBytes
            thermalState = device.thermalState
        }
        
        if memoryBudgetBytes > 0 {
            memoryUsageFraction = Double(memoryUsedBytes) / Double(memoryBudgetBytes)
        }
        
        // CPU
        let cpu = ss_get_cpu_usage()
        if cpu >= 0 {
            cpuUsagePercent = cpu
        }
        
        // Elapsed time
        if let start = inferenceStartTime {
            elapsedSeconds = Date().timeIntervalSince(start)
        }
        
        // Tokens per second (rolling 1-second window)
        let now = Date()
        if let lastCheck = lastTokenCheckTime {
            let dt = now.timeIntervalSince(lastCheck)
            if dt >= 0.5 {
                let newTokens = tokensGenerated - lastTokenCount
                tokensPerSecond = Double(newTokens) / dt
                lastTokenCount = tokensGenerated
                lastTokenCheckTime = now
                
                // History for sparkline
                speedHistory.append(tokensPerSecond)
                if speedHistory.count > maxHistoryLength {
                    speedHistory.removeFirst()
                }
                
                // ETA computation using average speed
                let avgTokensPerSecond = elapsedSeconds > 0 ? (Double(tokensGenerated) / elapsedSeconds) : 0
                if avgTokensPerSecond > 0 && maxTokens > 0 {
                    let remaining = max(0, maxTokens - tokensGenerated)
                    estimatedRemainingSeconds = Double(remaining) / avgTokensPerSecond
                } else {
                    estimatedRemainingSeconds = nil
                }
            }
        }
    }
    
    // MARK: - Formatted Strings
    
    public var memoryUsedFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(memoryUsedBytes), countStyle: .memory)
    }
    
    public var memoryBudgetFormatted: String {
        memoryBudgetBytes > 0 
            ? ByteCountFormatter.string(fromByteCount: Int64(memoryBudgetBytes), countStyle: .memory)
            : "No Limit"
    }
    
    public var memoryAvailableFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(memoryAvailableBytes), countStyle: .memory)
    }
    
    public var elapsedFormatted: String {
        let mins = Int(elapsedSeconds) / 60
        let secs = Int(elapsedSeconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
    
    public var speedFormatted: String {
        let avgTokensPerSecond = elapsedSeconds > 0 ? (Double(tokensGenerated) / elapsedSeconds) : 0
        return String(format: "%.1f tok/s", avgTokensPerSecond)
    }
    
    public var cpuFormatted: String {
        String(format: "%.0f%%", cpuUsagePercent)
    }
}
