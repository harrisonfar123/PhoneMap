import Foundation
#if canImport(UIKit)
import UIKit
#endif
import Combine
import os

public struct CrashInfo: Codable {
    public enum CrashType: String, Codable {
        case exception
        case signal
    }
    
    public let type: CrashType
    public let name: String
    public let reason: String
    public let stackTrace: [String]
    public let timestamp: Date
    public let memoryInfo: UInt64
    public let thermalState: Int
}

public extension Notification.Name {
    static let slipstreamEvictLRULayers = Notification.Name("slipstreamEvictLRULayers")
    static let slipstreamEvictAllInactiveLayers = Notification.Name("slipstreamEvictAllInactiveLayers")
    static let slipstreamPauseInference = Notification.Name("slipstreamPauseInference")
}

public class MemoryPressureMonitor {
    public static let shared = MemoryPressureMonitor()
    
    private let warningThreshold: UInt64 = 100
    private let criticalThreshold: UInt64 = 50
    
    private init() {
        setupMemoryWarningObserver()
    }
    
    private func setupMemoryWarningObserver() {
#if os(iOS)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
#endif
        
        // Dispatch source for more granular monitoring
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        source.setEventHandler { [weak self] in
            let event = source.data
            if event.contains(.critical) {
                self?.performAggressiveWeightEviction()
            } else if event.contains(.warning) {
                self?.performStandardWeightEviction()
            }
        }
        source.resume()
    }
    
    public func getAvailableMemory() -> UInt64 {
#if os(iOS)
        return UInt64(os_proc_available_memory())
#else
        return 0 // host_statistics can be used here for macOS
#endif
    }
    
    @objc private func handleMemoryWarning() {
        performStandardWeightEviction()
    }
    
    private func performStandardWeightEviction() {
        NotificationCenter.default.post(name: .slipstreamEvictLRULayers, object: nil)
    }
    
    private func performAggressiveWeightEviction() {
        NotificationCenter.default.post(name: .slipstreamEvictAllInactiveLayers, object: nil)
        NotificationCenter.default.post(name: .slipstreamPauseInference, object: nil)
    }
}

public class ThermalStateMonitor: ObservableObject {
    public static let shared = ThermalStateMonitor()
    
    @Published public private(set) var currentThermalState: ProcessInfo.ThermalState = .nominal
    private var cancellables = Set<AnyCancellable>()
    
    private init() {
        NotificationCenter.default
            .publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .sink { [weak self] _ in
                self?.handleThermalStateChange()
            }
            .store(in: &cancellables)
    }
    
    private func handleThermalStateChange() {
        let newState = ProcessInfo.processInfo.thermalState
        currentThermalState = newState
        
        switch newState {
        case .nominal:
            // Placeholder for API: InferenceEngine.shared.setBatchSize(32)
            break
        case .fair:
            // Placeholder for API: InferenceEngine.shared.setBatchSize(16)
            break
        case .serious:
            // Placeholder for API: InferenceEngine.shared.setBatchSize(8)
            break
        case .critical:
            // Placeholder for API: InferenceEngine.shared.pauseInference()
            break
        @unknown default:
            break
        }
    }
}

public class CrashDiagnosticsManager {
    public static let shared = CrashDiagnosticsManager()
    
    public func setupCrashHandlers() {
        // Exception handler
        NSSetUncaughtExceptionHandler { exception in
            CrashDiagnosticsManager.shared.handleException(exception)
        }
    }
    
    private func handleException(_ exception: NSException) {
        let crashInfo = CrashInfo(
            type: .exception,
            name: exception.name.rawValue,
            reason: exception.reason ?? "Unknown",
            stackTrace: exception.callStackSymbols,
            timestamp: Date(),
            memoryInfo: CrashDiagnosticsManager.shared.getAvailableMemoryHelper(),
            thermalState: ProcessInfo.processInfo.thermalState.rawValue
        )
        
        saveCrashInfo(crashInfo)
    }
    
    public func handleSignal(signal: Int32) {
        let crashInfo = CrashInfo(
            type: .signal,
            name: "Signal",
            reason: "Received signal \(signal)",
            stackTrace: Thread.callStackSymbols,
            timestamp: Date(),
            memoryInfo: CrashDiagnosticsManager.shared.getAvailableMemoryHelper(),
            thermalState: ProcessInfo.processInfo.thermalState.rawValue
        )
        
        saveCrashInfo(crashInfo)
    }
    
    private func saveCrashInfo(_ crashInfo: CrashInfo) {
        if let encoded = try? JSONEncoder().encode(crashInfo) {
            UserDefaults.standard.set(encoded, forKey: "last_crash_info")
        }
    }
    
    fileprivate func getAvailableMemoryHelper() -> UInt64 {
#if os(iOS)
        return UInt64(os_proc_available_memory())
#else
        return 0
#endif
    }
}
