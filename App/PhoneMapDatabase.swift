import Foundation
import Accelerate
import SQLite3

public struct PhoneItemMeta {
    public let id: String
    public let type: String
    public let contentText: String
    public let x: Double
    public let y: Double
    public let cluster: Int
    public let embedding: [Float]
}

public class PhoneMapDatabase {
    private var db: OpaquePointer?
    
    // In-memory cache for fast vector searches
    private var cachedItems: [PhoneItemMeta] = []
    
    public init(databasePath: String) throws {
        if sqlite3_open(databasePath, &db) != SQLITE_OK {
            throw NSError(domain: "PhoneMapDB", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to open DB"])
        }
        try initializeSchema()
        try loadCache()
    }
    
    deinit {
        if let db = db {
            sqlite3_close(db)
        }
    }
    
    private func initializeSchema() throws {
        let createTable = """
        CREATE TABLE IF NOT EXISTS phone_items (
            id TEXT PRIMARY KEY,
            type TEXT,
            content_text TEXT,
            x REAL,
            y REAL,
            cluster INTEGER,
            embedding BLOB
        );
        """
        if sqlite3_exec(db, createTable, nil, nil, nil) != SQLITE_OK {
            throw NSError(domain: "PhoneMapDB", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed creating table"])
        }
    }
    
    public func insertItem(_ item: PhoneItemMeta) throws {
        let insertSQL = "INSERT OR REPLACE INTO phone_items (id, type, content_text, x, y, cluster, embedding) VALUES (?, ?, ?, ?, ?, ?, ?);"
        var stmt: OpaquePointer?
        
        guard sqlite3_prepare_v2(db, insertSQL, -1, &stmt, nil) == SQLITE_OK else {
            throw NSError(domain: "PhoneMapDB", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to prepare insert"])
        }
        
        defer { sqlite3_finalize(stmt) }
        
        sqlite3_bind_text(stmt, 1, (item.id as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 2, (item.type as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 3, (item.contentText as NSString).utf8String, -1, nil)
        sqlite3_bind_double(stmt, 4, item.x)
        sqlite3_bind_double(stmt, 5, item.y)
        sqlite3_bind_int(stmt, 6, Int32(item.cluster))
        
        let embeddingData = Data(buffer: UnsafeBufferPointer(start: item.embedding, count: item.embedding.count))
        embeddingData.withUnsafeBytes { ptr in
            sqlite3_bind_blob(stmt, 7, ptr.baseAddress, Int32(ptr.count), nil)
        }
        
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw NSError(domain: "PhoneMapDB", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to step insert"])
        }
        
        cachedItems.append(item)
    }
    
    private func loadCache() throws {
        cachedItems.removeAll()
        
        let query = "SELECT id, type, content_text, x, y, cluster, embedding FROM phone_items;"
        var stmt: OpaquePointer?
        
        guard sqlite3_prepare_v2(db, query, -1, &stmt, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(stmt) }
        
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = String(cString: sqlite3_column_text(stmt, 0))
            let type = String(cString: sqlite3_column_text(stmt, 1))
            let contentText = String(cString: sqlite3_column_text(stmt, 2))
            let x = sqlite3_column_double(stmt, 3)
            let y = sqlite3_column_double(stmt, 4)
            let cluster = Int(sqlite3_column_int(stmt, 5))
            
            let blobBytes = sqlite3_column_blob(stmt, 6)
            let blobCount = Int(sqlite3_column_bytes(stmt, 6))
            var embedding: [Float] = []
            
            if blobCount > 0, let blobBytes = blobBytes {
                let count = blobCount / MemoryLayout<Float>.size
                let buffer = UnsafeBufferPointer<Float>(start: blobBytes.assumingMemoryBound(to: Float.self), count: count)
                embedding = Array(buffer)
            }
            
            cachedItems.append(PhoneItemMeta(id: id, type: type, contentText: contentText, x: x, y: y, cluster: cluster, embedding: embedding))
        }
    }
    
    /// Compute Cosine Similarity between vector A and B
    private func cosineSimilarity(a: [Float], b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        
        var dotProduct: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        
        var aMagSq: Float = 0
        vDSP_svesq(a, 1, &aMagSq, vDSP_Length(a.count))
        
        var bMagSq: Float = 0
        vDSP_svesq(b, 1, &bMagSq, vDSP_Length(b.count))
        
        let denominator = sqrt(aMagSq) * sqrt(bMagSq)
        if denominator == 0 { return 0 }
        
        return dotProduct / denominator
    }
    
    /// Find top K most similar items
    public func search(queryEmbedding: [Float], topK: Int = 10) -> [(item: PhoneItemMeta, score: Float)] {
        var scored: [(item: PhoneItemMeta, score: Float)] = []
        
        for item in cachedItems {
            let score = cosineSimilarity(a: queryEmbedding, b: item.embedding)
            scored.append((item, score))
        }
        
        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(topK))
    }
    
    public func getAllItems() -> [PhoneItemMeta] {
        return cachedItems
    }
    
    public func updateItemsCache(_ items: [PhoneItemMeta]) {
        self.cachedItems = items
        
        // Optionally write back projection updates to SQL.
        // For MVP speed, keep in memory.
    }
    
    public func deleteAllItems() throws {
        let deleteSQL = "DELETE FROM phone_items;"
        if sqlite3_exec(db, deleteSQL, nil, nil, nil) != SQLITE_OK {
            throw NSError(domain: "PhoneMapDB", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed deleting items"])
        }
        cachedItems.removeAll()
    }
}
