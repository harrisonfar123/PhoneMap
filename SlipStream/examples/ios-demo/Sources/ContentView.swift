import SwiftUI
import UniformTypeIdentifiers
import UIKit
import AVFoundation
import SlipStream

// MARK: - Models
struct ChatMessage: Identifiable, Equatable, Codable {
    var id = UUID()
    var isUser: Bool
    var text: String
    var isThinking: Bool = false
    var layerProgress: String? = nil
}

struct ChatSession: Identifiable, Codable, Equatable {
    var id = UUID()
    var date: Date
    var previewText: String
    var messages: [ChatMessage]
}

// MARK: - Main View
struct ContentView: View {
    @State private var messages: [ChatMessage] = []
    @State private var promptText: String = ""
    @State private var isGenerating: Bool = false
    @State private var errorMessage: String? = nil
    @State private var lastProfilePath: URL? = nil
    
    @State private var showFilePicker = false
    @State private var showSettings = false
    @State private var showHistory = false
    
    // Persisted settings
    @AppStorage("modelBookmarkData") private var modelBookmarkData: Data = Data()
    @AppStorage("memoryBudget") private var storedMemoryBudget: Int = 0
    
    // Persist chat history
    @AppStorage("pastSessions") private var pastSessionsData: Data = Data()
    @State private var currentSessionId: UUID = UUID()
    @State private var pastSessions: [ChatSession] = []
    
    @State private var activeModelURL: URL? = nil
    
    // Silent audio player to keep process alive in background
    @State private var silentPlayer: AVAudioPlayer? = nil
    
    // Active model reference for throttle control
    @State private var activeModel: SlipStreamModel? = nil
    
    // Stats tracker
    @StateObject private var statsTracker = SlipStreamStatsTracker()
    @State private var showStats = false
    
    var body: some View {
        NavigationView {
            ZStack {
                VStack(spacing: 0) {
                    // Chat Scroll
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(spacing: 16) {
                                if messages.isEmpty {
                                    emptyStateView
                                } else {
                                    ForEach(messages) { message in
                                        MessageBubble(message: message)
                                            .id(message.id)
                                    }
                                }
                            }
                            .padding()
                        }
                        .onChange(of: messages) { _ in
                            saveChatHistory()
                            if let lastMessage = messages.last {
                                withAnimation {
                                    proxy.scrollTo(lastMessage.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                    .background(Color(UIColor.systemGroupedBackground))
                    
                    // Error Bar
                    if let error = errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(8)
                            .frame(maxWidth: .infinity)
                            .background(Color.red.opacity(0.8))
                    }
                    
                    // Input Area
                    inputArea
                }
                
                // Stats overlay (floating bottom-right)
                if showStats || isGenerating {
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            StatsOverlayView(tracker: statsTracker)
                                .padding(.trailing, 16)
                                .padding(.bottom, 100)
                        }
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .navigationTitle("SlipStream AI")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    HStack(spacing: 12) {
                        Button(action: {
                            // Reset setup
                            UserDefaults.standard.set(false, forKey: "setupComplete")
                            // Force re-render by posting notification
                            NotificationCenter.default.post(name: NSNotification.Name("resetSetup"), object: nil)
                        }) {
                            Image(systemName: "gearshape.fill")
                                .imageScale(.medium)
                                .foregroundColor(.secondary)
                        }
                        
                        // History button
                        Button(action: {
                            withAnimation { showHistory = true }
                        }) {
                            Image(systemName: "clock.arrow.circlepath")
                                .imageScale(.medium)
                                .foregroundColor(.cyan)
                        }
                        
                        // Clear chat button
                        if !messages.isEmpty {
                            Button(action: {
                                withAnimation {
                                    messages.removeAll()
                                    saveChatHistory()
                                }
                            }) {
                                Image(systemName: "trash")
                                    .imageScale(.medium)
                                    .foregroundColor(.red)
                            }
                        }
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack(spacing: 12) {
                        // Stats toggle
                        Button(action: { withAnimation { showStats.toggle() } }) {
                            Image(systemName: showStats ? "chart.bar.fill" : "chart.bar")
                                .imageScale(.large)
                                .foregroundColor(showStats ? .cyan : .secondary)
                        }
                        
                        // Hardware info badge
                        if let device = DeviceInfo.current() {
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(thermalColor(for: device.thermalState))
                                    .frame(width: 8, height: 8)
                                
                                Text(device.thermalState.description)
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundColor(.secondary)
                            }
                        }
                        
                        if lastProfilePath != nil {
                            Image(systemName: "checkmark.doc.fill")
                                .imageScale(.large)
                                .foregroundColor(.green)
                        }
                        
                        Button(action: { showFilePicker = true }) {
                            Image(systemName: "folder.badge.plus")
                                .imageScale(.large)
                                .foregroundColor(activeModelURL == nil ? .red : .accentColor)
                        }
                    }
                }
            }
            .fileImporter(isPresented: $showFilePicker, allowedContentTypes: [.data]) { result in
                handleFilePicked(result: result)
            }
            .sheet(isPresented: $showHistory) {
                PastChatsView(
                    sessions: $pastSessions,
                    currentSessionId: $currentSessionId,
                    messages: $messages,
                    showHistory: $showHistory
                )
            }
            .onAppear {
                loadChatHistory()
                restoreModelFromBookmark()
                setupBackgroundObservers()
                // Set memory budget on stats tracker
                if storedMemoryBudget > 0 {
                    statsTracker.memoryBudgetBytes = UInt64(storedMemoryBudget)
                } else if let device = DeviceInfo.current() {
                    let preset = MemoryPreset.recommended(for: device)
                    statsTracker.memoryBudgetBytes = preset.budgetBytes(for: device)
                }
            }
        }
    }
    
    // MARK: - Chat History
    private func saveChatHistory() {
        guard !messages.isEmpty else { return }
        let preview = messages.last { !$0.isUser }?.text ?? messages.first?.text ?? "New Chat"
        
        let session = ChatSession(
            id: currentSessionId,
            date: Date(), // update date
            previewText: preview,
            messages: messages
        )
        
        // Update or append
        if let index = pastSessions.firstIndex(where: { $0.id == currentSessionId }) {
            pastSessions[index] = session
        } else {
            pastSessions.insert(session, at: 0)
        }
        
        if let data = try? JSONEncoder().encode(pastSessions) {
            pastSessionsData = data
        }
    }
    
    private func loadChatHistory() {
        if !pastSessionsData.isEmpty {
            if let saved = try? JSONDecoder().decode([ChatSession].self, from: pastSessionsData) {
                pastSessions = saved
                
                // If there's an existing recent session, just load it or start fresh.
                // Alternatively, just start a new session ID if messages is empty
                if messages.isEmpty {
                    currentSessionId = UUID()
                }
            }
        }
    }
    
    // MARK: - Thermal Color
    private func thermalColor(for state: DeviceInfo.ThermalLevel) -> Color {
        switch state {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        }
    }
    
    // MARK: - Empty State
    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Spacer().frame(height: 60)
            
            Image(systemName: "cpu")
                .font(.system(size: 60))
                .foregroundColor(.accentColor.opacity(0.8))
            
            Text("SlipStream Engine")
                .font(.title2).bold()
            
            if activeModelURL == nil {
                Text("No model loaded.")
                    .foregroundColor(.secondary)
                
                Button("Select .gguf Model") {
                    showFilePicker = true
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else {
                Text("Model Ready!")
                    .foregroundColor(.green)
                Text(activeModelURL?.lastPathComponent ?? "")
                    .font(.footnote)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
            }
            
            // Device info summary
            if let device = DeviceInfo.current() {
                HStack(spacing: 8) {
                    Image(systemName: "iphone")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(device.summary)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 10)
                
                HStack(spacing: 8) {
                    Image(systemName: "memorychip")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("Budget: \(ByteCountFormatter.string(fromByteCount: Int64(statsTracker.memoryBudgetBytes), countStyle: .memory))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
    }
    
    // MARK: - Input Area
    private var inputArea: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(alignment: .bottom, spacing: 12) {
                TextField("Message AI...", text: $promptText)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .background(Color(UIColor.secondarySystemBackground))
                    .cornerRadius(20)
                    .disabled(isGenerating || activeModelURL == nil)
                
                Button(action: sendMessage) {
                    Image(systemName: isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(activeModelURL == nil ? .gray : (isGenerating ? .red : .accentColor))
                }
                .disabled(activeModelURL == nil || (promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating))
            }
            .padding()
            .background(Color(UIColor.systemBackground))
        }
    }
    
    // MARK: - Logic
    private func sendMessage() {
        if isGenerating {
            activeModel?.cancel()
            return
        }
        
        // Haptic feedback
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
        
        let userPrompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !userPrompt.isEmpty else { return }
        
        // Add User Message
        messages.append(ChatMessage(isUser: true, text: userPrompt))
        promptText = ""
        errorMessage = nil
        
        // Add Empty Assistant Message
        messages.append(ChatMessage(isUser: false, text: "", isThinking: true))
        
        Task {
            await startGeneration(for: userPrompt)
            
            // Haptic feedback upon completion
            let finishedGenerator = UINotificationFeedbackGenerator()
            finishedGenerator.notificationOccurred(.success)
        }
    }
    
    @MainActor
    private func startGeneration(for prompt: String) async {
        guard let url = activeModelURL else {
            errorMessage = "Please select a model first."
            return
        }
        
        // Request security-scoped access
        let accessing = url.startAccessingSecurityScopedResource()
        defer {
            if accessing { url.stopAccessingSecurityScopedResource() }
        }
        
        // Silent Audio keeps the process alive indefinitely in the background.
        // There is no need for `beginBackgroundTask` which triggers 30s Jetsam warnings.
        
        isGenerating = true
        statsTracker.startTracking()
        startSilentAudio()
        
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        SlipStreamModel.setProfileOutputDir(docsDir.path)
        
        do {
            var config = SlipStreamConfig()
            config.contextSize = 0 // No size limit
            
            // Apply memory budget from setup, but let user exceed it if they want by ignoring it here
            // Removing memory enforcement completely on engine side as requested by user
            config.memoryBudget = 0 
            
            #if targetEnvironment(simulator)
            config.backend = .cpu
            #endif
            
            let model = try SlipStreamModel(path: url.path, config: config)
            activeModel = model
            statsTracker.maxTokens = 500 // Arbitrary max to show ETA, usually passed from prompt
            let stream = model.generate(prompt: prompt)
            
            // Batch token updates to reduce SwiftUI re-renders.
            var tokenBuffer = ""
            var lastFlush = CFAbsoluteTimeGetCurrent()
            let flushInterval: CFAbsoluteTime = 0.1 // 100ms
            
            for try await token in stream {
                guard let lastIndex = messages.indices.last else { break }
                
                if token.hasPrefix("|PROGRESS|") {
                    if messages[lastIndex].text.isEmpty {
                        let parts = token.components(separatedBy: "|")
                        if parts.count >= 5 {
                            let current = parts[3]
                            let total = parts[4]
                            messages[lastIndex].isThinking = true
                            messages[lastIndex].layerProgress = "Warming up: Layer \(current)/\(total)"
                        }
                    }
                } else {
                    if messages[lastIndex].isThinking {
                        messages[lastIndex].isThinking = false
                        messages[lastIndex].layerProgress = nil
                    }
                    tokenBuffer += token
                    statsTracker.recordToken()
                    
                    // Flush buffer to UI if enough time has passed
                    let now = CFAbsoluteTimeGetCurrent()
                    if now - lastFlush >= flushInterval {
                        messages[lastIndex].text += tokenBuffer
                        tokenBuffer = ""
                        lastFlush = now
                    }
                }
            }
            
            // Flush any remaining buffered tokens
            if !tokenBuffer.isEmpty, let lastIndex = messages.indices.last {
                messages[lastIndex].text += tokenBuffer
            }
            
            // Cleanup empty messages if needed
            if messages[messages.indices.last!].text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                messages[messages.indices.last!].text = "*(No response generated)*"
            }
            
            activeModel = nil
            saveChatHistory()
            
        } catch {
            errorMessage = error.localizedDescription
            if let lastIndex = messages.indices.last, messages[lastIndex].text.isEmpty {
                messages.removeLast()
            }
            saveChatHistory()
        }
        
        isGenerating = false
        statsTracker.stopTracking()
        stopSilentAudio()
        if let profPath = SlipStreamModel.getLastProfilePath() {
            lastProfilePath = URL(fileURLWithPath: profPath)
            print("Profile saved to: \(profPath)")
        }
    }
    
    // (Duplicate save/load chat history functions removed here)
    
    // MARK: - File Handling
    private func handleFilePicked(result: Result<URL, Error>) {
        do {
            let selectedUrl = try result.get()
            
            let accessing = selectedUrl.startAccessingSecurityScopedResource()
            defer {
                if accessing { selectedUrl.stopAccessingSecurityScopedResource() }
            }
            
            let bookmarkData = try selectedUrl.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
            modelBookmarkData = bookmarkData
            activeModelURL = selectedUrl
            errorMessage = nil
        } catch {
            errorMessage = "Failed to bookmark file: \(error.localizedDescription)"
        }
    }
    
    private func restoreModelFromBookmark() {
        if modelBookmarkData.isEmpty {
            if let defaultUrl = Bundle.main.url(forResource: "tinyllama-q4_0", withExtension: "gguf") {
                activeModelURL = defaultUrl
            }
            return
        }
        
        do {
            var isStale = false
            let restoredUrl = try URL(resolvingBookmarkData: modelBookmarkData, options: .withoutUI, relativeTo: nil, bookmarkDataIsStale: &isStale)
            activeModelURL = restoredUrl
            
            if isStale {
                handleFilePicked(result: .success(restoredUrl))
            }
        } catch {
            print("Bookmark restoration failed: \(error)")
            modelBookmarkData = Data()
            activeModelURL = nil
        }
    }
    
    // MARK: - Background Audio Keep-Alive
    private func startSilentAudio() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: .mixWithOthers)
            try session.setActive(true)
            
            if let url = Bundle.main.url(forResource: "silence", withExtension: "wav") {
                let player = try AVAudioPlayer(contentsOf: url)
                player.numberOfLoops = -1
                player.volume = 0.0
                player.play()
                silentPlayer = player
            }
        } catch {
            print("Silent audio setup failed: \(error)")
        }
    }
    
    private func stopSilentAudio() {
        silentPlayer?.stop()
        silentPlayer = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
    
    // MARK: - Background / Foreground Throttle
    private func setupBackgroundObservers() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil, queue: .main
        ) { _ in
            activeModel?.setThrottle(delayMicroseconds: 50_000)
            print("SlipStream: Backgrounded — throttle ON (50ms/token)")
        }
        
        NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil, queue: .main
        ) { _ in
            activeModel?.setThrottle(delayMicroseconds: 0)
            print("SlipStream: Foregrounded — throttle OFF (full speed)")
        }
    }
}

// MARK: - Chat Bubble View
struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser { Spacer() }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 5) {
                if message.isThinking {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text(message.layerProgress ?? "Thinking...")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color(UIColor.secondarySystemGroupedBackground))
                    .cornerRadius(20)
                } else {
                    Text(message.text)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .foregroundColor(message.isUser ? .white : .primary)
                        .background(message.isUser ? Color.accentColor : Color(UIColor.secondarySystemGroupedBackground))
                        .cornerRadius(20)
                }
            }
            
            if !message.isUser { Spacer() }
        }
    }
}
