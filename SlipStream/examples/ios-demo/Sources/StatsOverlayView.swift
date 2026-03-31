import SwiftUI
import SlipStream

// MARK: - Stats Overlay

struct StatsOverlayView: View {
    @ObservedObject var tracker: SlipStreamStatsTracker
    @State private var isExpanded = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Collapsed: compact pill
            if !isExpanded {
                collapsedPill
            }
            
            // Expanded: full stats panel
            if isExpanded {
                expandedPanel
            }
        }
        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: isExpanded)
        .environment(\.colorScheme, .dark) // Force dark theme for the stats overlay
    }
    
    // MARK: - Collapsed Pill
    
    private var collapsedPill: some View {
        Button(action: { withAnimation { isExpanded = true } }) {
            HStack(spacing: 10) {
                // Memory ring
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.15), lineWidth: 3)
                        .frame(width: 28, height: 28)
                    
                    Circle()
                        .trim(from: 0, to: min(tracker.memoryUsageFraction, 1.0))
                        .stroke(
                            memoryColor,
                            style: StrokeStyle(lineWidth: 3, lineCap: .round)
                        )
                        .rotationEffect(.degrees(-90))
                        .frame(width: 28, height: 28)
                        .animation(.easeInOut(duration: 0.5), value: tracker.memoryUsageFraction)
                    
                    Image(systemName: "memorychip")
                        .font(.system(size: 10))
                        .foregroundColor(memoryColor)
                }
                
                // Speed
                Text(tracker.speedFormatted)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
                
                // Thermal dot
                Circle()
                    .fill(thermalColor)
                    .frame(width: 8, height: 8)
                
                Image(systemName: "chevron.up")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(.white.opacity(0.4))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                Capsule()
                    .fill(Color(UIColor.secondarySystemGroupedBackground))
                    .overlay(
                        Capsule()
                            .stroke(Color.white.opacity(0.1), lineWidth: 0.5)
                    )
            )
        }
    }
    
    // MARK: - Expanded Panel
    
    private var expandedPanel: some View {
        VStack(spacing: 14) {
            // Header with close button
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(.cyan)
                Text("Live Stats")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(.white)
                
                Spacer()
                
                Button(action: { withAnimation { isExpanded = false } }) {
                    Image(systemName: "chevron.down.circle.fill")
                        .font(.system(size: 20))
                        .foregroundColor(.white.opacity(0.4))
                }
            }
            
            Divider().background(Color.white.opacity(0.1))
            
            // Memory gauge
            VStack(spacing: 8) {
                HStack {
                    Label("Memory", systemImage: "memorychip.fill")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white.opacity(0.6))
                    
                    Spacer()
                    
                    Text("\(tracker.memoryUsedFormatted) / \(tracker.memoryBudgetFormatted)")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                }
                
                // Memory bar
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.white.opacity(0.08))
                        
                        RoundedRectangle(cornerRadius: 4)
                            .fill(LinearGradient(
                                colors: memoryGradientColors,
                                startPoint: .leading, endPoint: .trailing
                            ))
                            .frame(width: geo.size.width * min(CGFloat(tracker.memoryUsageFraction), 1.0))
                            .animation(.easeInOut(duration: 0.5), value: tracker.memoryUsageFraction)
                    }
                }
                .frame(height: 8)
                
                HStack {
                    Text("Available: \(tracker.memoryAvailableFormatted)")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.35))
                    Spacer()
                    Text(String(format: "%.0f%%", min(tracker.memoryUsageFraction * 100, 999)))
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundColor(memoryColor)
                }
            }
            
            // Stats grid
            HStack(spacing: 10) {
                statCell(
                    icon: "thermometer.medium",
                    label: "Thermal",
                    value: tracker.thermalState.description,
                    color: thermalColor
                )
                
                statCell(
                    icon: "cpu",
                    label: "CPU",
                    value: tracker.cpuFormatted,
                    color: cpuColor
                )
            }
            
            HStack(spacing: 10) {
                statCell(
                    icon: "bolt.fill",
                    label: "Speed",
                    value: tracker.speedFormatted,
                    color: .cyan
                )
                
                statCell(
                    icon: "number",
                    label: "Tokens",
                    value: "\(tracker.tokensGenerated)",
                    color: .purple
                )
            }
            
            // Speed sparkline
            if !tracker.speedHistory.isEmpty {
                sparklineView
            }
            
            // Elapsed & ETA
            HStack {
                Image(systemName: "clock.fill")
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.4))
                
                if let eta = tracker.estimatedRemainingSeconds {
                    let mins = Int(eta) / 60
                    let secs = Int(eta) % 60
                    Text("ETA: \(mins):\(String(format: "%02d", secs))")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.white.opacity(0.7))
                } else {
                    Text("Elapsed: \(tracker.elapsedFormatted)")
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.5))
                }
                Spacer()
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(UIColor.secondarySystemGroupedBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.white.opacity(0.1), lineWidth: 0.5)
                )
        )
        .frame(maxWidth: 320)
        .transition(.scale(scale: 0.8).combined(with: .opacity))
    }
    
    private func statCell(icon: String, label: String, value: String, color: Color) -> some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12))
                    .foregroundColor(color)
                
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.5))
            }
            
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundColor(.white)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color.white.opacity(0.04))
        .cornerRadius(10)
    }
    
    // MARK: - Sparkline
    
    private var sparklineView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Speed History")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.4))
            
            GeometryReader { geo in
                let data = tracker.speedHistory
                let maxVal = max(data.max() ?? 1, 0.1)
                
                Path { path in
                    guard data.count >= 2 else { return }
                    let stepX = geo.size.width / CGFloat(max(data.count - 1, 1))
                    
                    for (i, val) in data.enumerated() {
                        let x = CGFloat(i) * stepX
                        let y = geo.size.height * (1.0 - CGFloat(val / maxVal))
                        if i == 0 {
                            path.move(to: CGPoint(x: x, y: y))
                        } else {
                            path.addLine(to: CGPoint(x: x, y: y))
                        }
                    }
                }
                .stroke(
                    LinearGradient(colors: [.cyan, .blue], startPoint: .leading, endPoint: .trailing),
                    style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round)
                )
            }
            .frame(height: 40)
        }
        .padding(10)
        .background(Color.white.opacity(0.03))
        .cornerRadius(10)
    }
    
    // MARK: - Colors
    
    private var memoryColor: Color {
        let f = tracker.memoryUsageFraction
        if f < 0.6 { return .green }
        if f < 0.8 { return .orange }
        return .red
    }
    
    private var memoryGradientColors: [Color] {
        let f = tracker.memoryUsageFraction
        if f < 0.6 { return [.green, .cyan] }
        if f < 0.8 { return [.yellow, .orange] }
        return [.orange, .red]
    }
    
    private var thermalColor: Color {
        switch tracker.thermalState {
        case .nominal:  return .green
        case .fair:     return .yellow
        case .serious:  return .orange
        case .critical: return .red
        }
    }
    
    private var cpuColor: Color {
        if tracker.cpuUsagePercent < 50 { return .green }
        if tracker.cpuUsagePercent < 80 { return .yellow }
        return .orange
    }
}
