import Foundation
import CSlipStream

// MARK: - Device Info

/// Swift-friendly wrapper around ss_device_info_t.
public struct DeviceInfo {
    public let deviceModel: String       // e.g. "iPhone15,2"
    public let chipName: String          // e.g. "Apple A16 Bionic"
    public let cpuCoreCount: Int
    public let perfCoreCount: Int
    public let efficiencyCoreCount: Int
    public let totalMemoryBytes: UInt64
    public let availableMemoryBytes: UInt64
    public let gpuCoreCount: Int
    public let thermalState: ThermalLevel
    
    /// Human-readable total RAM string, e.g. "6 GB"
    public var totalMemoryFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(totalMemoryBytes), countStyle: .memory)
    }
    
    /// Human-readable available RAM string
    public var availableMemoryFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(availableMemoryBytes), countStyle: .memory)
    }
    
    /// Friendly device name derived from model identifier
    public var friendlyDeviceName: String {
        Self.friendlyName(for: deviceModel)
    }
    
    /// One-line summary: "iPhone 15 Pro · A17 Pro · 8 GB"
    public var summary: String {
        "\(friendlyDeviceName) · \(chipName) · \(totalMemoryFormatted)"
    }
    
    public enum ThermalLevel: Int, CaseIterable, CustomStringConvertible {
        case nominal = 0
        case fair = 1
        case serious = 2
        case critical = 3
        
        public var description: String {
            switch self {
            case .nominal:  return "Cool"
            case .fair:     return "Warm"
            case .serious:  return "Hot"
            case .critical: return "Critical"
            }
        }
        
        public var emoji: String {
            switch self {
            case .nominal:  return "🟢"
            case .fair:     return "🟡"
            case .serious:  return "🟠"
            case .critical: return "🔴"
            }
        }
    }
    
    /// Query current device hardware. Returns nil on failure.
    public static func current() -> DeviceInfo? {
        var cInfo = ss_device_info_t()
        let err = ss_get_device_info(&cInfo)
        guard err == SS_OK else { return nil }
        
        let model = withUnsafePointer(to: cInfo.device_model) {
            $0.withMemoryRebound(to: CChar.self, capacity: 64) { String(cString: $0) }
        }
        let chip = withUnsafePointer(to: cInfo.chip_name) {
            $0.withMemoryRebound(to: CChar.self, capacity: 64) { String(cString: $0) }
        }
        
        return DeviceInfo(
            deviceModel: model,
            chipName: chip,
            cpuCoreCount: Int(cInfo.cpu_core_count),
            perfCoreCount: Int(cInfo.perf_core_count),
            efficiencyCoreCount: Int(cInfo.efficiency_core_count),
            totalMemoryBytes: cInfo.total_memory,
            availableMemoryBytes: cInfo.available_memory,
            gpuCoreCount: Int(cInfo.gpu_core_count),
            thermalState: ThermalLevel(rawValue: Int(cInfo.thermal_state)) ?? .nominal
        )
    }
    
    private static func friendlyName(for model: String) -> String {
        // Common iPhone models
        let map: [(prefix: String, name: String)] = [
            ("iPhone17,3", "iPhone 16 Pro Max"),
            ("iPhone17,4", "iPhone 16 Pro Max"),
            ("iPhone17,1", "iPhone 16 Pro"),
            ("iPhone17,2", "iPhone 16 Pro"),
            ("iPhone17,5", "iPhone 16 Plus"),
            ("iPhone17,6", "iPhone 16"),
            ("iPhone16,2", "iPhone 15 Pro Max"),
            ("iPhone16,1", "iPhone 15 Pro"),
            ("iPhone15,5", "iPhone 15 Plus"),
            ("iPhone15,4", "iPhone 15"),
            ("iPhone15,3", "iPhone 14 Pro Max"),
            ("iPhone15,2", "iPhone 14 Pro"),
            ("iPhone14,8", "iPhone 14 Plus"),
            ("iPhone14,7", "iPhone 14"),
            ("iPhone14,3", "iPhone 13 Pro Max"),
            ("iPhone14,2", "iPhone 13 Pro"),
            ("iPhone14,5", "iPhone 13"),
            ("iPhone14,4", "iPhone 13 mini"),
            ("iPhone13,4", "iPhone 12 Pro Max"),
            ("iPhone13,3", "iPhone 12 Pro"),
            ("iPhone13,2", "iPhone 12"),
            ("iPhone13,1", "iPhone 12 mini"),
            // iPads
            ("iPad16", "iPad Pro (M4)"),
            ("iPad14", "iPad Pro (M2)"),
            ("iPad13", "iPad Air (M1)"),
        ]
        
        for entry in map {
            if model.hasPrefix(entry.prefix) {
                return entry.name
            }
        }
        
        if model.hasPrefix("iPhone") { return "iPhone" }
        if model.hasPrefix("iPad")   { return "iPad" }
        if model.contains("arm64")   { return "Mac (Apple Silicon)" }
        if model.contains("x86_64")  { return "Simulator" }
        
        return model
    }
}

// MARK: - Memory Presets

/// Beginner-friendly memory allocation presets.
/// Each preset uses a percentage of total device RAM as the inference budget.
public enum MemoryPreset: String, CaseIterable, Identifiable {
    case light    = "light"
    case balanced = "balanced"
    case maximum  = "maximum"
    case custom   = "custom"
    
    public var id: String { rawValue }
    
    /// Human-friendly display name
    public var displayName: String {
        switch self {
        case .light:    return "Light"
        case .balanced: return "Balanced"
        case .maximum:  return "Maximum"
        case .custom:   return "Custom"
        }
    }
    
    /// Subtitle explaining the preset
    public var subtitle: String {
        switch self {
        case .light:    return "Safe for older devices · Uses 40% RAM"
        case .balanced: return "Recommended for most devices · Uses 55% RAM"
        case .maximum:  return "For newer iPhones with 8GB+ · Uses 70% RAM"
        case .custom:   return "Set your own memory limit"
        }
    }
    
    /// Icon name (SF Symbols)
    public var iconName: String {
        switch self {
        case .light:    return "leaf.fill"
        case .balanced: return "equal.circle.fill"
        case .maximum:  return "bolt.fill"
        case .custom:   return "slider.horizontal.3"
        }
    }
    
    /// Fraction of total RAM to use (0.0 - 1.0)
    public var ramFraction: Double {
        switch self {
        case .light:    return 0.40
        case .balanced: return 0.55
        case .maximum:  return 0.70
        case .custom:   return 0.55 // default, overridden by user
        }
    }
    
    /// Calculate the memory budget in bytes for the given device.
    public func budgetBytes(for device: DeviceInfo) -> UInt64 {
        UInt64(Double(device.totalMemoryBytes) * ramFraction)
    }
    
    /// Formatted budget string
    public func budgetFormatted(for device: DeviceInfo) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(budgetBytes(for: device)), countStyle: .memory)
    }
    
    /// Recommended preset based on total device RAM
    public static func recommended(for device: DeviceInfo) -> MemoryPreset {
        let totalGB = Double(device.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0)
        if totalGB >= 8.0 {
            return .balanced
        } else if totalGB >= 6.0 {
            return .balanced
        } else {
            return .light
        }
    }
}

// MARK: - Model Compatibility Checker

/// Checks whether a GGUF model fits within a given memory budget.
public struct ModelCompatibilityResult {
    public let modelPath: String
    public let estimatedPeakMemory: UInt64
    public let budgetBytes: UInt64
    public let fits: Bool
    public let usagePercent: Double  // 0.0 - 100.0+
    
    public var estimatedFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(estimatedPeakMemory), countStyle: .memory)
    }
    
    public var budgetFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(budgetBytes), countStyle: .memory)
    }
    
    public var statusMessage: String {
        if fits {
            if usagePercent < 60 {
                return "✅ Plenty of room — this model runs comfortably"
            } else if usagePercent < 85 {
                return "✅ Good fit — this model should run well"
            } else {
                return "⚠️ Tight fit — may work but performance could suffer"
            }
        } else {
            return "❌ Too large — this model needs more RAM than your budget"
        }
    }
}

public struct ModelCompatibilityChecker {
    /// Check if a .gguf model fits the memory budget at a given context size.
    public static func check(
        modelPath: String,
        budgetBytes: UInt64,
        contextSize: Int = 2048
    ) -> ModelCompatibilityResult? {
        guard let peakMemory = SlipStreamModel.estimateMemory(
            modelPath: modelPath,
            contextSize: contextSize
        ) else { return nil }
        
        let fits = peakMemory <= budgetBytes
        let usage = budgetBytes > 0 ? (Double(peakMemory) / Double(budgetBytes)) * 100.0 : 100.0
        
        return ModelCompatibilityResult(
            modelPath: modelPath,
            estimatedPeakMemory: peakMemory,
            budgetBytes: budgetBytes,
            fits: fits,
            usagePercent: usage
        )
    }
}
