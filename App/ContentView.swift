import SwiftUI
import SlipStream

struct ContentView: View {
    @State private var db: PhoneMapDatabase?
    @State private var embedder: SlipStreamModel?
    @State private var pipeline: IngestionPipeline?
    
    @State private var items: [PhoneItemMeta] = []
    
    @State private var query: String = ""
    @State private var searchResults: [String: Float] = [:] // id -> score
    
    @State private var isIndexing = false
    @State private var indexingStatus = "Initializing..."
    @State private var progress: Double = 0.0
    @AppStorage("hasCompletedFirstRun") private var hasCompletedFirstRun: Bool = false
    
    @State private var selectedItem: PhoneItemMeta?
    @State private var activeFilters: Set<String> = ["contact", "note", "photo", "message"]
    
    let clusterColors: [Color] = [
        .purple, .blue, .green, .orange, .pink, .teal, .indigo, .mint
    ]
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // ─── 1. Semantic Map Canvas (UMAP Fallback) ───
            GeometryReader { geo in
                let scale = min(geo.size.width, geo.size.height) / 100.0 // Adjusted for force-directed layout
                
                Canvas { context, size in
                    context.translateBy(x: size.width / 2, y: size.height / 2)
                    
                    for item in items where activeFilters.contains(item.type) {
                        let rect = CGRect(
                            x: item.x * scale,
                            y: item.y * scale,
                            width: 16, height: 16
                        )
                        let path = Path(ellipseIn: rect)
                        let color = clusterColors[abs(item.cluster) % clusterColors.count]
                        
                        let isHighlighted = searchResults.keys.contains(item.id)
                        let opacity = searchResults.isEmpty ? 0.8 : (isHighlighted ? 1.0 : 0.2)
                        let highlightGlow = isHighlighted ? 2.0 : 0.0
                        
                        context.fill(path, with: .color(color.opacity(opacity)))
                        context.stroke(path, with: .color(.white.opacity(isHighlighted ? 1.0 : 0.5)), lineWidth: 1 + highlightGlow)
                    }
                }
                .drawingGroup()
                .gesture(
                    SpatialTapGesture()
                        .onEnded { value in
                            // Very basic collision detection for MVP point hover/preview
                            let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                            let tapX = (value.location.x - center.x) / scale
                            let tapY = (value.location.y - center.y) / scale
                            
                            if let tapped = items.first(where: { activeFilters.contains($0.type) && abs($0.x - tapX) < 10 && abs($0.y - tapY) < 10 }) {
                                withAnimation { selectedItem = tapped }
                            } else {
                                withAnimation { selectedItem = nil }
                            }
                        }
                )
            }
            .ignoresSafeArea()
            
            // ─── 2. Overlay UI ───
            VStack {
                // Header & Indexing Controls
                HStack(alignment: .top) {
                    VStack(alignment: .leading) {
                        Text("VectorLens")
                            .font(.custom("Inter-Bold", size: 24, relativeTo: .title))
                            .fontWeight(.heavy)
                            .foregroundColor(.white)
                            .shadow(color: .purple.opacity(0.5), radius: 10, x: 0, y: 0)
                        
                        // Type Filters
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack {
                                ForEach(["contact", "note", "photo", "message"], id: \.self) { type in
                                    Button(action: {
                                        withAnimation {
                                            if activeFilters.contains(type) { activeFilters.remove(type) }
                                            else { activeFilters.insert(type) }
                                        }
                                    }) {
                                        Text(type.capitalized)
                                            .font(.caption).bold()
                                            .padding(.horizontal, 10).padding(.vertical, 5)
                                            .background(activeFilters.contains(type) ? Color.purple.opacity(0.8) : Color.white.opacity(0.2))
                                            .foregroundColor(.white)
                                            .clipShape(Capsule())
                                    }
                                }
                            }
                        }
                    }
                    
                    Spacer()
                    
                    if !hasCompletedFirstRun {
                        Button(action: startIndexing) {
                            Image(systemName: "cpu")
                                .font(.system(size: 20))
                                .foregroundColor(isIndexing ? .green : .white)
                                .padding(10)
                                .background(.ultraThinMaterial)
                                .clipShape(Circle())
                        }
                        .disabled(isIndexing)
                    } else {
                        Image(systemName: "checkmark.seal.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.green)
                            .shadow(color: .green.opacity(0.8), radius: 5)
                            .padding(.trailing, 10)
                    }
                }
                .padding()
                
                if isIndexing {
                    VStack(alignment: .leading) {
                        Text(indexingStatus)
                            .font(.caption)
                            .foregroundColor(.gray)
                        ProgressView(value: progress)
                            .progressViewStyle(LinearProgressViewStyle(tint: .purple))
                    }
                    .padding()
                    .background(.ultraThinMaterial)
                    .cornerRadius(12)
                    .padding(.horizontal)
                    .transition(.move(edge: .top).combined(with: .opacity))
                }
                
                Spacer()
                
                // Item Preview Overlay
                if let selected = selectedItem {
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text(selected.type.capitalized)
                                .font(.caption).bold()
                                .foregroundColor(clusterColors[abs(selected.cluster) % clusterColors.count])
                            Spacer()
                            Button(action: { withAnimation { selectedItem = nil } }) {
                                Image(systemName: "xmark.circle.fill").foregroundColor(.gray)
                            }
                        }
                        Text(selected.contentText)
                            .font(.subheadline)
                            .foregroundColor(.white)
                            .lineLimit(4)
                    }
                    .padding()
                    .background(.ultraThinMaterial)
                    .cornerRadius(16)
                    .padding(.horizontal)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
                
                // Natural Language Search Bar
                HStack {
                    Image(systemName: "sparkles")
                        .foregroundColor(.purple)
                    
                    TextField("Explore your data semantically...", text: $query)
                        .foregroundColor(.white)
                        .font(.subheadline)
                        .onSubmit { Task { await performSearch() } }
                    
                    if !searchResults.isEmpty {
                        Button(action: {
                            withAnimation {
                                query = ""
                                searchResults.removeAll()
                            }
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.gray)
                        }
                    } else {
                        Button(action: { Task { await performSearch() } }) {
                            Image(systemName: "arrow.up.circle.fill")
                                .foregroundColor(query.isEmpty ? .gray : .purple)
                                .font(.system(size: 28))
                        }
                        .disabled(query.isEmpty)
                    }
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 30)
                        .fill(.ultraThinMaterial)
                        .shadow(color: .black.opacity(0.3), radius: 10, x: 0, y: 5)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 30)
                        .stroke(
                            LinearGradient(
                                colors: [.purple.opacity(0.5), .blue.opacity(0.3), .clear],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1.5
                        )
                )
                .padding(.horizontal)
                .padding(.bottom, 30)
            }
        }
        .onAppear {
            initializeBackend()
        }
    }
    
    // ─── 3. Backend Integration ───
    
    private func initializeBackend() {
        // Load SlipStream embedding model and SQLite DB
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // In a real device app, model is bundled or downloaded to documents path
                let modelPath = Bundle.main.path(forResource: "nomic-embed-text-v1.5-q4_k_m", ofType: "gguf") ?? "/tmp/model.gguf"
                let e = try SlipStreamModel(path: modelPath)
                
                let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let dbPath = docsURL.appendingPathComponent("phonemap.sqlite").path
                let d = try PhoneMapDatabase(databasePath: dbPath)
                
                let p = IngestionPipeline(db: d, embedder: e)
                
                DispatchQueue.main.async {
                    self.embedder = e
                    self.db = d
                    self.pipeline = p
                    self.items = d.getAllItems()
                    
                    if self.items.isEmpty {
                        self.indexingStatus = "Ready for First-Run Index!"
                    }
                }
            } catch {
                print("Backend Init Failed: \(error)")
            }
        }
    }
    
    private func startIndexing() {
        guard let pipeline = pipeline, !isIndexing, !hasCompletedFirstRun else { return }
        
        withAnimation { isIndexing = true }
        
        Task {
            await pipeline.startIndexing { prog, msg in
                DispatchQueue.main.async {
                    self.progress = prog
                    self.indexingStatus = msg
                }
            }
            
            // Re-fetch items from DB
            DispatchQueue.main.async {
                self.items = self.db?.getAllItems() ?? []
            }
            
            // Run Projection Clustering
            var localItems = self.items
            await ProjectionLayer.clusterAndProject(items: &localItems, iterations: 150)
            
            let finalItems = localItems
            DispatchQueue.main.async {
                self.items = finalItems
                self.db?.updateItemsCache(self.items)
                withAnimation { 
                    self.isIndexing = false 
                    self.hasCompletedFirstRun = true
                }
            }
        }
    }
    
    private func performSearch() async {
        guard let db = db, let embedder = embedder, !query.isEmpty else { return }
        
        do {
            let qVector = try await embedder.embed(prompt: query)
            let results = db.search(queryEmbedding: qVector, topK: 5)
            
            DispatchQueue.main.async {
                withAnimation {
                    self.searchResults.removeAll()
                    for r in results {
                        if r.score > 0.4 { // Threshold
                            self.searchResults[r.item.id] = r.score
                        }
                    }
                    
                    // Show top hit immediately
                    if let top = results.first?.item {
                        self.selectedItem = top
                    }
                }
            }
        } catch {
            print("Search error: \(error)")
        }
    }
}
