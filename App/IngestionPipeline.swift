import Foundation
import Photos
import Contacts
import SlipStream

public class IngestionPipeline {
    private let db: PhoneMapDatabase
    private let embedder: SlipStreamModel
    
    public init(db: PhoneMapDatabase, embedder: SlipStreamModel) {
        self.db = db
        self.embedder = embedder
    }
    
    /// Starts the indexing process on a background actor
    public func startIndexing(progress: @escaping (Double, String) -> Void) async {
        progress(0.1, "Extracting Contacts...")
        await indexContacts()
        
        progress(0.4, "Extracting Local Files...")
        await indexLocalNotes()
        
        progress(1.0, "Indexing Complete!")
    }
    
    private func indexContacts() async {
        let store = CNContactStore()
        let keysToFetch = [CNContactGivenNameKey, CNContactFamilyNameKey, CNContactNoteKey, CNContactJobTitleKey] as [CNKeyDescriptor]
        let request = CNContactFetchRequest(keysToFetch: keysToFetch)
        
        do {
            try store.enumerateContacts(with: request) { contact, stop in
                let text = "\(contact.givenName) \(contact.familyName). \(contact.jobTitle). \(contact.note)"
                if !text.trimmingCharacters(in: .whitespaces).isEmpty {
                    Task {
                        await self.processAndStore(id: contact.identifier, type: "contact", text: text)
                    }
                }
            }
        } catch {
            print("Failed to fetch contacts: \(error)")
        }
    }
    
    private func indexLocalNotes() async {
        // Fallback dummy data for the MVP to showcase the clustering immediately
        // since full sandbox filesystem access requires user picker intents.
        let dummyNotes = [
            "Meeting with engineering team about the new backend architecture constraints.",
            "Grocery list: Milk, Eggs, Bread, Bananas, Coffee.",
            "Flight tickets to Tokyo booked for October 12th. Confirmation number: XJ932K.",
            "Gym routine: Bench press 3x10, Squats 4x8, Deadlift 3x5.",
            "Idea for startup: AI-powered vector search for local on-device files.",
            "Call Sarah for her birthday tomorrow at 5 PM."
        ]
        
        for (i, note) in dummyNotes.enumerated() {
            let id = "note_\(i)"
            await processAndStore(id: id, type: "note", text: note)
        }
    }
    
    private func processAndStore(id: String, type: String, text: String) async {
        // Basic Chunker: in a full implementation we would split by semantic paragraphs.
        // For MVP, chunk up to 512 tokens.
        do {
            let vector = try await embedder.embed(prompt: text)
            
            // Assign random temporary coordinates and cluster. 
            // In a real setup, Projection layer updates these asynchronously.
            let x = Double.random(in: -100...100)
            let y = Double.random(in: -100...100)
            
            let item = PhoneItemMeta(
                id: id,
                type: type,
                contentText: text,
                x: x,
                y: y,
                cluster: Int.random(in: 0...4),
                embedding: vector
            )
            
            try db.insertItem(item)
            
        } catch {
            print("Failed to embed item \\(id): \\(error)")
        }
    }
}
