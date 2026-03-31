import SwiftUI

struct PastChatsView: View {
    @Binding var sessions: [ChatSession]
    @Binding var currentSessionId: UUID
    @Binding var messages: [ChatMessage]
    @Binding var showHistory: Bool
    
    var body: some View {
        NavigationView {
            ZStack {
                Color(UIColor.systemGroupedBackground)
                    .ignoresSafeArea()
                
                if sessions.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "tray.fill")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary.opacity(0.5))
                        Text("No Past Chats")
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                } else {
                    List {
                        ForEach(sessions) { session in
                            Button(action: {
                                // Load this session
                                currentSessionId = session.id
                                messages = session.messages
                                showHistory = false
                            }) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(session.date.formatted(date: .abbreviated, time: .shortened))
                                        .font(.caption)
                                        .foregroundColor(.cyan)
                                        .fontWeight(.medium)
                                    
                                    Text(session.previewText)
                                        .font(.body)
                                        .foregroundColor(.primary)
                                        .lineLimit(2)
                                        .multilineTextAlignment(.leading)
                                    
                                    HStack {
                                        Image(systemName: "message.fill")
                                        Text("\(session.messages.count) messages")
                                    }
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                }
                                .padding(.vertical, 4)
                            }
                            .listRowBackground(Color(UIColor.secondarySystemGroupedBackground))
                        }
                        .onDelete { indexSet in
                            // Delete from past sessions list
                            sessions.remove(atOffsets: indexSet)
                            // If they delete the active one, clear chat
                            if !sessions.contains(where: { $0.id == currentSessionId }) {
                                messages.removeAll()
                                currentSessionId = UUID()
                            }
                        }
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: {
                        // Start New Chat
                        currentSessionId = UUID()
                        messages.removeAll()
                        showHistory = false
                    }) {
                        Image(systemName: "square.and.pencil")
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        showHistory = false
                    }) {
                        Text("Done")
                            .font(.headline)
                    }
                }
            }
        }
    }
}
