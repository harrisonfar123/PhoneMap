import SwiftUI

@main
struct SlipStreamDemoApp: App {
    @AppStorage("setupComplete") private var isSetupComplete: Bool = false
    @AppStorage("modelBookmarkData") private var modelBookmarkData: Data = Data()
    @AppStorage("memoryBudget") private var memoryBudget: Int = 0
    
    @State private var selectedModelURL: URL? = nil
    @State private var memoryBudgetUInt: UInt64 = 0
    
    var body: some Scene {
        WindowGroup {
            if isSetupComplete {
                ContentView()
                    .onAppear {
                        memoryBudgetUInt = UInt64(memoryBudget)
                    }
                    .toolbar {
                        ToolbarItem(placement: .navigationBarLeading) {
                            Button(action: {
                                withAnimation { isSetupComplete = false }
                            }) {
                                Image(systemName: "gearshape.fill")
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
            } else {
                SetupView(
                    isSetupComplete: $isSetupComplete,
                    selectedModelURL: $selectedModelURL,
                    memoryBudget: $memoryBudgetUInt,
                    modelBookmarkData: $modelBookmarkData
                )
                .onChange(of: memoryBudgetUInt) { newValue in
                    memoryBudget = Int(newValue)
                }
                .onChange(of: selectedModelURL) { newURL in
                    // Bookmark is already saved inside SetupView
                }
            }
        }
    }
}
