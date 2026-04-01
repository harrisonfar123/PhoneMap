import SwiftUI
import SlipStream

struct ContentView: View {
    @State private var db: PhoneMapDatabase?
    @State private var embedder: MotimodelProvider?
    @State private var pipeline: IngestionPipeline?
    
    @State private var items: [PhoneItemMeta] = []
    
    @State private var query: String = ""
    @State private var searchResults: [String: Float] = [:] // id -> score
    
    @State private var isIndexing = false
    @State private var indexingStatus = "Initializing..."
    @State private var progress: Double = 0.0
    
    @State private var isShowing3DCanvas = false
    
    let clusterColors: [Color] = [
        .purple, .blue, .green, .orange, .pink, .teal, .indigo, .mint
    ]
    
    var body: some View {
        NavigationStack {
            List {
                searchResultsList
            }
            .listStyle(InsetGroupedListStyle())
            .navigationTitle("Search")
            .searchable(text: $query, prompt: "Explore your data semantically...")
            .onSubmit(of: .search) {
                Task { await performSearch() }
            }
            .onChange(of: query) { newValue in
                if newValue.isEmpty {
                    withAnimation { searchResults.removeAll() }
                }
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: {
                        isShowing3DCanvas = true
                    }) {
                        Image(systemName: "cube.transparent")
                            .foregroundColor(.purple)
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: startIndexing) {
                        if isIndexing {
                            ProgressView()
                        } else {
                            Image(systemName: "arrow.triangle.2.circlepath")
                        }
                    }
                    .disabled(isIndexing)
                }
            }
            .navigationDestination(isPresented: $isShowing3DCanvas) {
                VectorCanvasView(items: $items, searchResults: $searchResults, clusterColors: clusterColors)
            }
            .overlay(
                Group {
                    if isIndexing {
                        loadingOverlay
                    }
                }
            )
        }
        .onAppear {
            initializeBackend()
        }
    }
    
    @ViewBuilder
    private var searchResultsList: some View {
        if !searchResults.isEmpty {
            Section(header: Text("Semantic Matches")) {
                ForEach(searchResults.sorted(by: { $0.value > $1.value }), id: \.key) { result in
                    if let item = items.first(where: { $0.id == result.key }) {
                        SearchResultRow(item: item, score: result.value, color: clusterColors[abs(item.cluster) % clusterColors.count])
                    }
                }
            }
        } else if !items.isEmpty {
            Section(header: Text("Recent Index (\(items.count) items)")) {
                ForEach(items.prefix(100)) { item in
                    SearchResultRow(item: item, score: nil, color: clusterColors[abs(item.cluster) % clusterColors.count])
                }
            }
        } else {
            Section {
                Text("Index empty. Hit Scan Device.")
                    .foregroundColor(.gray)
                    .italic()
            }
        }
    }
    
    var loadingOverlay: some View {
        ZStack {
            Color.black.opacity(0.8).ignoresSafeArea()
            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.5)
                    .progressViewStyle(CircularProgressViewStyle(tint: .purple))
                
                Text(indexingStatus)
                    .font(.headline)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    
                Text(String(format: "%.0f%%", progress * 100))
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.7))
            }
            .padding(40)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 30))
            .shadow(color: .purple.opacity(0.4), radius: 30)
        }
        .transition(.opacity.combined(with: .scale))
    }
    
    // ─── Backend Integration ───
    private func initializeBackend() {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let modelPath = Bundle.main.path(forResource: "nomic-embed-text-v1.5-q4_k_m", ofType: "gguf") ?? "/tmp/model.gguf"
                let textEngine = try SlipStreamModel(path: modelPath)
                let unifiedEngine = LocalMotimodelEngine(textEngine: textEngine)
                
                let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let dbPath = docsURL.appendingPathComponent("phonemap.sqlite").path
                let d = try PhoneMapDatabase(databasePath: dbPath)
                
                let p = IngestionPipeline(db: d, embedder: unifiedEngine)
                
                DispatchQueue.main.async {
                    self.embedder = unifiedEngine
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
        guard let pipeline = pipeline, !isIndexing else { return }
        withAnimation { isIndexing = true }
        
        Task {
            await pipeline.startIndexing { prog, msg in
                DispatchQueue.main.async {
                    self.progress = prog
                    self.indexingStatus = msg
                }
            }
            
            DispatchQueue.main.async {
                self.items = self.db?.getAllItems() ?? []
            }
            
            var localItems = self.items
            await ProjectionLayer.clusterAndProject(items: &localItems, iterations: 150)
            
            let finalItems = localItems
            DispatchQueue.main.async {
                self.items = finalItems
                self.db?.updateItemsCache(self.items)
                withAnimation { self.isIndexing = false }
            }
        }
    }
    
    private func performSearch() async {
        guard let db = db, let embedder = embedder, !query.isEmpty else { return }
        
        do {
            let qVector = try await embedder.embed(text: query)
            let results = db.search(queryEmbedding: qVector, topK: 15)
            
            DispatchQueue.main.async {
                withAnimation {
                    self.searchResults.removeAll()
                    for r in results {
                        if r.score > 0.4 {
                            self.searchResults[r.item.id] = r.score
                        }
                    }
                }
            }
        } catch {
            print("Search error: \(error)")
        }
    }
}

// ─── Subviews ───

struct SearchResultRow: View {
    let item: PhoneItemMeta
    let score: Float?
    let color: Color
    
    @State private var isExpanded = false
    
    var iconName: String {
        switch item.type.lowercased() {
        case "photo": return "photo.fill"
        case "contact": return "person.crop.circle.fill"
        case "app": return "app.badge.fill"
        case "file": return "doc.text.fill"
        default: return "doc.fill"
        }
    }
    
    var body: some View {
        Button(action: {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                isExpanded.toggle()
            }
        }) {
            HStack(alignment: .top, spacing: 15) {
                Image(systemName: iconName)
                    .font(.title2)
                    .foregroundColor(color)
                    .frame(width: 30)
                
                VStack(alignment: .leading, spacing: 5) {
                    Text(item.type.capitalized)
                        .font(.caption)
                        .bold()
                        .foregroundColor(.secondary)
                    
                    Text(item.contentText)
                        .font(.subheadline)
                        .foregroundColor(.primary)
                        .lineLimit(isExpanded ? nil : 2)
                        .multilineTextAlignment(.leading)
                }
                
                Spacer()
                
                if let s = score {
                    Text(String(format: "%.0f%%", s * 100))
                        .font(.caption2)
                        .bold()
                        .padding(5)
                        .background(Color.green.opacity(0.2))
                        .foregroundColor(.green)
                        .cornerRadius(5)
                }
            }
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct VectorCanvasView: View {
    @Binding var items: [PhoneItemMeta]
    @Binding var searchResults: [String: Float]
    let clusterColors: [Color]
    
    @State private var selectedItem: PhoneItemMeta?
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            GeometryReader { geo in
                let scale = min(geo.size.width, geo.size.height) / 100.0
                
                Canvas { context, size in
                    context.translateBy(x: size.width / 2, y: size.height / 2)
                    
                    for item in items {
                        let rect = CGRect(x: item.x * scale, y: item.y * scale, width: 16, height: 16)
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
                    SpatialTapGesture().onEnded { value in
                        let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                        let tapX = (value.location.x - center.x) / scale
                        let tapY = (value.location.y - center.y) / scale
                        
                        if let tapped = items.first(where: { abs($0.x - tapX) < 10 && abs($0.y - tapY) < 10 }) {
                            withAnimation { selectedItem = tapped }
                        } else {
                            withAnimation { selectedItem = nil }
                        }
                    }
                )
            }
            
            if let selected = selectedItem {
                VStack {
                    Spacer()
                    SearchResultRow(item: selected, score: searchResults[selected.id], color: clusterColors[abs(selected.cluster) % clusterColors.count])
                        .padding()
                        .background(.ultraThinMaterial)
                        .cornerRadius(16)
                        .padding()
                }
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .navigationTitle("3D Vector Map")
        .navigationBarTitleDisplayMode(.inline)
    }
}
