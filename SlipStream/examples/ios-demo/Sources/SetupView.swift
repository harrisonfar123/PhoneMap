import SwiftUI
import SlipStream

// MARK: - Setup View (Guided Onboarding)

struct SetupView: View {
    @Binding var isSetupComplete: Bool
    @Binding var selectedModelURL: URL?
    @Binding var memoryBudget: UInt64
    @Binding var modelBookmarkData: Data
    
    @State private var currentStep: SetupStep = .hardware
    @State private var deviceInfo: DeviceInfo?
    @State private var selectedPreset: MemoryPreset = .balanced
    @State private var customRamGB: Double = 3.0
    @State private var showFilePicker = false
    @State private var compatibilityResult: ModelCompatibilityResult?
    @State private var animateIn = false
    
    enum SetupStep: Int, CaseIterable {
        case hardware = 0
        case memory = 1
        case model = 2
        case summary = 3
        
        var title: String {
            switch self {
            case .hardware: return "Your Device"
            case .memory:   return "RAM Budget"
            case .model:    return "Load Model"
            case .summary:  return "Ready!"
            }
        }
    }
    
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                colors: [
                    Color(red: 0.08, green: 0.08, blue: 0.14),
                    Color(red: 0.05, green: 0.05, blue: 0.10)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Progress bar
                progressBar
                    .padding(.horizontal, 24)
                    .padding(.top, 16)
                
                // Step content
                TabView(selection: $currentStep) {
                    hardwareStep.tag(SetupStep.hardware)
                    memoryStep.tag(SetupStep.memory)
                    modelStep.tag(SetupStep.model)
                    summaryStep.tag(SetupStep.summary)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .animation(.easeInOut(duration: 0.3), value: currentStep)
                
                // Navigation buttons
                navigationButtons
                    .padding(.horizontal, 24)
                    .padding(.bottom, 24)
            }
        }
        .onAppear {
            deviceInfo = DeviceInfo.current()
            if let device = deviceInfo {
                selectedPreset = MemoryPreset.recommended(for: device)
                customRamGB = Double(device.totalMemoryBytes) * 0.55 / (1024 * 1024 * 1024)
            }
            withAnimation(.easeOut(duration: 0.6).delay(0.2)) {
                animateIn = true
            }
        }
        .fileImporter(isPresented: $showFilePicker, allowedContentTypes: [.data]) { result in
            handleFilePicked(result: result)
        }
    }
    
    // MARK: - Progress Bar
    
    private var progressBar: some View {
        HStack(spacing: 8) {
            ForEach(SetupStep.allCases, id: \.rawValue) { step in
                VStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(step.rawValue <= currentStep.rawValue
                              ? LinearGradient(colors: [.cyan, .blue], startPoint: .leading, endPoint: .trailing)
                              : LinearGradient(colors: [Color.white.opacity(0.15)], startPoint: .leading, endPoint: .trailing))
                        .frame(height: 4)
                    
                    Text(step.title)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(step.rawValue <= currentStep.rawValue ? .cyan : .white.opacity(0.35))
                }
            }
        }
    }
    
    // MARK: - Step 1: Hardware Detection
    
    private var hardwareStep: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer().frame(height: 20)
                
                // Icon
                ZStack {
                    Circle()
                        .fill(LinearGradient(colors: [.cyan.opacity(0.2), .blue.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 100, height: 100)
                    
                    Image(systemName: "cpu.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(LinearGradient(colors: [.cyan, .blue], startPoint: .topLeading, endPoint: .bottomTrailing))
                }
                .scaleEffect(animateIn ? 1.0 : 0.5)
                .opacity(animateIn ? 1.0 : 0)
                
                Text("Device Detected")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("SlipStream has analyzed your hardware")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.6))
                
                if let device = deviceInfo {
                    hardwareCard(device: device)
                        .transition(.scale.combined(with: .opacity))
                } else {
                    ProgressView()
                        .tint(.cyan)
                        .scaleEffect(1.5)
                }
            }
            .padding(.horizontal, 24)
        }
    }
    
    private func hardwareCard(device: DeviceInfo) -> some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "iphone")
                    .font(.title2)
                    .foregroundColor(.cyan)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(device.friendlyDeviceName)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.white)
                    Text(device.chipName)
                        .font(.system(size: 13))
                        .foregroundColor(.cyan.opacity(0.8))
                }
                
                Spacer()
                
                Text(device.thermalState.emoji)
                    .font(.title2)
            }
            .padding(16)
            
            Divider().background(Color.white.opacity(0.1))
            
            // Stats grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                hardwareStat(icon: "memorychip", label: "Total RAM", value: device.totalMemoryFormatted)
                hardwareStat(icon: "memorychip.fill", label: "Available", value: device.availableMemoryFormatted)
                hardwareStat(icon: "cpu", label: "CPU Cores", value: "\(device.cpuCoreCount)")
                hardwareStat(icon: "bolt.fill", label: "Perf Cores", value: "\(device.perfCoreCount)")
            }
            .padding(16)
        }
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.white.opacity(0.06))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(LinearGradient(colors: [.cyan.opacity(0.3), .blue.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1)
                )
        )
    }
    
    private func hardwareStat(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 16))
                .foregroundColor(.cyan.opacity(0.7))
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.45))
                Text(value)
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
            }
            
            Spacer()
        }
        .padding(10)
        .background(Color.white.opacity(0.04))
        .cornerRadius(12)
    }
    
    // MARK: - Step 2: Memory Configuration
    
    private var memoryStep: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer().frame(height: 20)
                
                ZStack {
                    Circle()
                        .fill(LinearGradient(colors: [.purple.opacity(0.2), .pink.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 100, height: 100)
                    
                    Image(systemName: "slider.horizontal.3")
                        .font(.system(size: 44))
                        .foregroundStyle(LinearGradient(colors: [.purple, .pink], startPoint: .topLeading, endPoint: .bottomTrailing))
                }
                
                Text("Set RAM Budget")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("How much memory should SlipStream use?")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.6))
                
                // Preset buttons
                VStack(spacing: 10) {
                    ForEach(MemoryPreset.allCases.filter { $0 != .custom }, id: \.id) { preset in
                        presetButton(preset)
                    }
                }
                
                // Custom slider
                if selectedPreset == .custom, let device = deviceInfo {
                    customSlider(device: device)
                }
                
                // Budget display
                if let device = deviceInfo {
                    budgetDisplay(device: device)
                }
            }
            .padding(.horizontal, 24)
        }
    }
    
    private func presetButton(_ preset: MemoryPreset) -> some View {
        Button(action: {
            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                selectedPreset = preset
                if let device = deviceInfo {
                    customRamGB = Double(preset.budgetBytes(for: device)) / (1024 * 1024 * 1024)
                }
            }
        }) {
            HStack(spacing: 14) {
                Image(systemName: preset.iconName)
                    .font(.system(size: 20))
                    .foregroundColor(selectedPreset == preset ? .white : .purple.opacity(0.7))
                    .frame(width: 32)
                
                VStack(alignment: .leading, spacing: 3) {
                    Text(preset.displayName)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                    
                    Text(preset.subtitle)
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.5))
                }
                
                Spacer()
                
                if let device = deviceInfo {
                    Text(preset.budgetFormatted(for: device))
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundColor(selectedPreset == preset ? .white : .purple.opacity(0.7))
                }
                
                Image(systemName: selectedPreset == preset ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(selectedPreset == preset ? .white : .white.opacity(0.2))
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(selectedPreset == preset
                          ? LinearGradient(colors: [.purple.opacity(0.4), .pink.opacity(0.3)], startPoint: .leading, endPoint: .trailing)
                          : LinearGradient(colors: [Color.white.opacity(0.05)], startPoint: .leading, endPoint: .trailing))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(selectedPreset == preset ? Color.purple.opacity(0.5) : Color.clear, lineWidth: 1)
                    )
            )
        }
    }
    
    private func customSlider(device: DeviceInfo) -> some View {
        let maxGB = Double(device.totalMemoryBytes) / (1024 * 1024 * 1024) * 0.85
        let percentage = customRamGB / (Double(device.totalMemoryBytes) / (1024 * 1024 * 1024)) * 100
        
        return VStack(spacing: 12) {
            HStack {
                Text("Custom Amount")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white.opacity(0.6))
                
                Spacer()
                
                Text(String(format: "%.1f GB (%.0f%%)", customRamGB, percentage))
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(.purple)
            }
            
            Slider(value: $customRamGB, in: 1.0...maxGB, step: 0.1)
                .tint(sliderColor(for: percentage))
            
            HStack {
                Text("1 GB")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.3))
                Spacer()
                Text(String(format: "%.1f GB", maxGB))
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.3))
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.white.opacity(0.05))
        )
    }
    
    private func sliderColor(for percentage: Double) -> Color {
        if percentage < 45 { return .green }
        if percentage < 60 { return .cyan }
        if percentage < 75 { return .orange }
        return .red
    }
    
    private func budgetDisplay(device: DeviceInfo) -> some View {
        let budgetBytes = selectedPreset == .custom
            ? UInt64(customRamGB * 1024 * 1024 * 1024)
            : selectedPreset.budgetBytes(for: device)
        let fraction = Double(budgetBytes) / Double(device.totalMemoryBytes)
        
        return VStack(spacing: 8) {
            // Memory bar
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.white.opacity(0.08))
                    
                    RoundedRectangle(cornerRadius: 8)
                        .fill(LinearGradient(
                            colors: fraction < 0.5 ? [.green, .cyan] : (fraction < 0.7 ? [.cyan, .orange] : [.orange, .red]),
                            startPoint: .leading, endPoint: .trailing
                        ))
                        .frame(width: geo.size.width * min(fraction, 1.0))
                        .animation(.spring(response: 0.5), value: fraction)
                }
            }
            .frame(height: 12)
            
            HStack {
                Text("Budget: \(ByteCountFormatter.string(fromByteCount: Int64(budgetBytes), countStyle: .memory))")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white.opacity(0.7))
                Spacer()
                Text("of \(device.totalMemoryFormatted) total")
                    .font(.system(size: 13))
                    .foregroundColor(.white.opacity(0.4))
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.white.opacity(0.05))
        )
    }
    
    // MARK: - Step 3: Model Picker
    
    private var modelStep: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer().frame(height: 20)
                
                ZStack {
                    Circle()
                        .fill(LinearGradient(colors: [.green.opacity(0.2), .mint.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 100, height: 100)
                    
                    Image(systemName: "doc.fill.badge.plus")
                        .font(.system(size: 44))
                        .foregroundStyle(LinearGradient(colors: [.green, .mint], startPoint: .topLeading, endPoint: .bottomTrailing))
                }
                
                Text("Load a Model")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("Select a .gguf model file from your device")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                
                // File picker button
                Button(action: { showFilePicker = true }) {
                    HStack(spacing: 12) {
                        Image(systemName: selectedModelURL != nil ? "checkmark.circle.fill" : "folder.badge.plus")
                            .font(.system(size: 24))
                        
                        VStack(alignment: .leading, spacing: 3) {
                            if let url = selectedModelURL {
                                Text(url.lastPathComponent)
                                    .font(.system(size: 15, weight: .semibold))
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Text("Tap to change model")
                                    .font(.system(size: 12))
                                    .opacity(0.6)
                            } else {
                                Text("Browse Files")
                                    .font(.system(size: 15, weight: .semibold))
                                Text("Select a .gguf model file")
                                    .font(.system(size: 12))
                                    .opacity(0.6)
                            }
                        }
                        
                        Spacer()
                        
                        Image(systemName: "chevron.right")
                            .font(.system(size: 14, weight: .semibold))
                            .opacity(0.5)
                    }
                    .foregroundColor(.white)
                    .padding(16)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(selectedModelURL != nil
                                  ? LinearGradient(colors: [.green.opacity(0.3), .mint.opacity(0.2)], startPoint: .leading, endPoint: .trailing)
                                  : LinearGradient(colors: [Color.white.opacity(0.06)], startPoint: .leading, endPoint: .trailing))
                            .overlay(
                                RoundedRectangle(cornerRadius: 14)
                                    .stroke(selectedModelURL != nil ? Color.green.opacity(0.4) : Color.white.opacity(0.1), lineWidth: 1)
                            )
                    )
                }
                
                // Compatibility result
                if let result = compatibilityResult {
                    compatibilityCard(result: result)
                }
                
                // Tips
                tipCard(
                    icon: "lightbulb.fill",
                    title: "Tip: Model Sizes",
                    body: "Q4 quantized models (1-4 GB) work best on iPhones. Start with a small model like TinyLlama or Phi-3 Mini."
                )
            }
            .padding(.horizontal, 24)
        }
    }
    
    private func compatibilityCard(result: ModelCompatibilityResult) -> some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: result.fits ? "checkmark.shield.fill" : "xmark.shield.fill")
                    .font(.system(size: 20))
                    .foregroundColor(result.fits ? .green : .red)
                
                Text(result.statusMessage)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white.opacity(0.8))
                
                Spacer()
            }
            
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Model needs")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.4))
                    Text(result.estimatedFormatted)
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                }
                
                Spacer()
                
                Image(systemName: "arrow.right")
                    .foregroundColor(.white.opacity(0.2))
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Your budget")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.4))
                    Text(result.budgetFormatted)
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundColor(result.fits ? .green : .red)
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(result.fits ? Color.green.opacity(0.08) : Color.red.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(result.fits ? Color.green.opacity(0.25) : Color.red.opacity(0.25), lineWidth: 1)
                )
        )
    }
    
    private func tipCard(icon: String, title: String, body: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundColor(.yellow)
                .frame(width: 28)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white.opacity(0.8))
                Text(body)
                    .font(.system(size: 13))
                    .foregroundColor(.white.opacity(0.5))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.yellow.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.yellow.opacity(0.15), lineWidth: 1)
                )
        )
    }
    
    // MARK: - Step 4: Summary
    
    private var summaryStep: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer().frame(height: 20)
                
                ZStack {
                    Circle()
                        .fill(LinearGradient(colors: [.green.opacity(0.3), .cyan.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 100, height: 100)
                    
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 52))
                        .foregroundStyle(LinearGradient(colors: [.green, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing))
                }
                
                Text("You're All Set!")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("Here's your configuration summary")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.6))
                
                // Summary cards
                VStack(spacing: 10) {
                    summaryRow(icon: "iphone", label: "Device", value: deviceInfo?.friendlyDeviceName ?? "Unknown")
                    summaryRow(icon: "cpu", label: "Chip", value: deviceInfo?.chipName ?? "Unknown")
                    summaryRow(icon: "memorychip", label: "RAM Budget", value: currentBudgetFormatted)
                    summaryRow(icon: "doc.fill", label: "Model", value: selectedModelURL?.lastPathComponent ?? "None selected")
                }
                .padding(16)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.white.opacity(0.05))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.08), lineWidth: 1)
                        )
                )
                
                tipCard(
                    icon: "chart.bar.fill",
                    title: "Live Stats",
                    body: "While chatting, tap the stats button to see real-time memory, CPU, thermal, and speed stats."
                )
            }
            .padding(.horizontal, 24)
        }
    }
    
    private func summaryRow(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16))
                .foregroundColor(.cyan)
                .frame(width: 24)
            
            Text(label)
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.5))
            
            Spacer()
            
            Text(value)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.white)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.vertical, 6)
    }
    
    // MARK: - Navigation
    
    private var navigationButtons: some View {
        HStack(spacing: 12) {
            if currentStep != .hardware {
                Button(action: {
                    withAnimation { currentStep = SetupStep(rawValue: currentStep.rawValue - 1) ?? .hardware }
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
                    .padding(.horizontal, 20)
                    .padding(.vertical, 14)
                    .background(Color.white.opacity(0.08))
                    .cornerRadius(14)
                }
            }
            
            Spacer()
            
            if currentStep == .summary {
                Button(action: finishSetup) {
                    HStack(spacing: 8) {
                        Text("Start Chatting")
                        Image(systemName: "arrow.right")
                    }
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 14)
                    .background(
                        LinearGradient(colors: [.cyan, .blue], startPoint: .leading, endPoint: .trailing)
                    )
                    .cornerRadius(14)
                }
            } else {
                Button(action: {
                    withAnimation { currentStep = SetupStep(rawValue: currentStep.rawValue + 1) ?? .summary }
                }) {
                    HStack(spacing: 6) {
                        Text("Next")
                        Image(systemName: "chevron.right")
                    }
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 14)
                    .background(
                        LinearGradient(colors: [.cyan.opacity(0.8), .blue.opacity(0.8)], startPoint: .leading, endPoint: .trailing)
                    )
                    .cornerRadius(14)
                }
                .disabled(currentStep == .model && selectedModelURL == nil)
                .opacity(currentStep == .model && selectedModelURL == nil ? 0.4 : 1.0)
            }
        }
    }
    
    // MARK: - Helpers
    
    private var currentBudgetBytes: UInt64 {
        if let device = deviceInfo {
            if selectedPreset == .custom {
                return UInt64(customRamGB * 1024 * 1024 * 1024)
            }
            return selectedPreset.budgetBytes(for: device)
        }
        return 0
    }
    
    private var currentBudgetFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(currentBudgetBytes), countStyle: .memory)
    }
    
    private func handleFilePicked(result: Result<URL, Error>) {
        do {
            let url = try result.get()
            
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }
            
            let bookmarkData = try url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
            modelBookmarkData = bookmarkData
            selectedModelURL = url
            
            // Check compatibility
            compatibilityResult = ModelCompatibilityChecker.check(
                modelPath: url.path,
                budgetBytes: currentBudgetBytes,
                contextSize: 512
            )
        } catch {
            print("File pick error: \(error)")
        }
    }
    
    private func finishSetup() {
        memoryBudget = currentBudgetBytes
        withAnimation(.easeInOut(duration: 0.4)) {
            isSetupComplete = true
        }
    }
}
