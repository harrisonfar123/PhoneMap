import Foundation
import Photos
import Contacts
import SlipStream

public class IngestionPipeline {
    private let db: PhoneMapDatabase
    private let embedder: MotimodelProvider
    
    public init(db: PhoneMapDatabase, embedder: MotimodelProvider) {
        self.db = db
        self.embedder = embedder
    }
    
    /// Starts the indexing process on a background actor
    public func startIndexing(progress: @escaping (Double, String) -> Void) async {
        do {
            try db.deleteAllItems()
        } catch {
            print("Failed to clear DB before re-indexing: \(error)")
        }
        
        progress(0.1, "Extracting Contacts...")
        await indexContacts()
        
        progress(0.4, "Extracting Semantic Pixels (Photos)...")
        await indexPhotos(limit: 50)
        
        progress(1.0, "Indexing Complete!")
    }
    
    private func indexContacts() async {
        let store = CNContactStore()
        let keysToFetch = [CNContactGivenNameKey, CNContactFamilyNameKey, CNContactJobTitleKey] as [CNKeyDescriptor]
        let request = CNContactFetchRequest(keysToFetch: keysToFetch)
        
        do {
            try store.enumerateContacts(with: request) { contact, stop in
                let text = "\(contact.givenName) \(contact.familyName). \(contact.jobTitle)"
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
    
    private func indexPhotos(limit: Int) async {
        var auth = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if auth == .notDetermined {
            auth = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        }
        
        guard auth == .authorized || auth == .limited else {
            print("Photo permission not granted.")
            return
        }
        
        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        fetchOptions.fetchLimit = limit
        
        let assets = PHAsset.fetchAssets(with: .image, options: fetchOptions)
        
        let manager = PHImageManager.default()
        let requestOptions = PHImageRequestOptions()
        requestOptions.isSynchronous = true // CRITICAL: Prevents duplicate continuation resumes
        requestOptions.deliveryMode = .highQualityFormat
        requestOptions.isNetworkAccessAllowed = true
        
        for i in 0..<assets.count {
            let asset = assets.object(at: i)
            
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                manager.requestImage(for: asset, targetSize: CGSize(width: 512, height: 512), contentMode: .aspectFill, options: requestOptions) { image, info in
                    
                    var safeCgImage = image?.cgImage
                    if safeCgImage == nil, let ci = image?.ciImage {
                        safeCgImage = CIContext().createCGImage(ci, from: ci.extent)
                    }
                    
                    guard let finalCgImg = safeCgImage else {
                        continuation.resume()
                        return
                    }
                    
                    Task {
                        await self.processAndStoreImage(id: asset.localIdentifier, type: "photo", image: finalCgImg)
                        continuation.resume()
                    }
                }
            }
        }
    }
    
    private func processAndStoreImage(id: String, type: String, image: CGImage) async {
        do {
            let tuple = try await embedder.embed(image: image)
            
            let x = Double.random(in: -100...100)
            let y = Double.random(in: -100...100)
            
            let item = PhoneItemMeta(
                id: id,
                type: type,
                contentText: tuple.description, // Maps Apple Vision description natively!
                x: x,
                y: y,
                cluster: Int.random(in: 0...4),
                embedding: tuple.vector
            )
            
            try db.insertItem(item)
            
        } catch {
            print("Failed to embed photo: \(error)")
        }
    }
    
    private func processAndStore(id: String, type: String, text: String) async {
        // Basic Chunker: in a full implementation we would split by semantic paragraphs.
        // For MVP, chunk up to 512 tokens.
        do {
            let vector = try await embedder.embed(text: text)
            
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
            print("Failed to embed item \(id): \(error)")
        }
    }
    
    private func indexMockApps() async {
        let mockApps = [
            ("com.apple.mobilesafari", "Safari - Web Browser. Surf the web, access bookmarks, read internet sites. Utility."),
            ("com.apple.Notes", "Apple Notes - Write down thoughts, save links, draw ideas. Editor."),
            ("com.apple.weather", "Weather - Check current forecast, temperature, radar, and rain. Widgets."),
            ("com.apple.MobileSMS", "Messages - Send text messages, SMS, and iMessage over internet. Communication."),
            ("com.apple.camera", "Camera - Take photos, record videos, portrait mode. Lens."),
            ("com.spotify.client", "Spotify - Play music, listen to podcasts, discover audio. Media."),
            ("com.burbn.instagram", "Instagram - Share photos, message friends, watch reels. Social Media."),
            ("com.apple.Health", "Health App - Track daily steps, heart rate, nutrition, sleep data. Wellness.")
        ]
        
        for app in mockApps {
            Task {
                await self.processAndStore(id: app.0, type: "app", text: app.1)
            }
        }
    }
    
    private func indexFiles() async {
        let fileManager = FileManager.default
        guard let docsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        
        do {
            let fileURLs = try fileManager.contentsOfDirectory(at: docsURL, includingPropertiesForKeys: nil)
            for url in fileURLs {
                let ext = url.pathExtension.lowercased()
                if ext == "txt" || ext == "csv" || ext == "json" {
                    if let fileContent = try? String(contentsOf: url, encoding: .utf8) {
                        let contentToEmbed = "File Name: \(url.lastPathComponent). Contents: \(fileContent)"
                        Task {
                            await self.processAndStore(id: url.absoluteString, type: "file", text: contentToEmbed)
                        }
                    }
                }
            }
        } catch {
            print("Failed to read App Documents: \(error)")
        }
    }
}
