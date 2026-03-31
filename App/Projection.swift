import Foundation
import Accelerate

public class ProjectionLayer {
    
    /// Calculate the Cosine Similarity between a and b
    private static func cosineSimilarity(a: [Float], b: [Float]) -> Float {
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
    
    /// Projects 768-D embeddings into a 2D space using a fast Force-Directed Graph simulation.
    /// This acts as a UMAP fallback for the MVP visualization layer.
    public static func clusterAndProject(items: inout [PhoneItemMeta], iterations: Int = 100) async {
        guard items.count > 1 else { return }
        
        var positions = items.map { _ in SIMD2<Double>(Double.random(in: -50...50), Double.random(in: -50...50)) }
        var velocities = items.map { _ in SIMD2<Double>(0, 0) }
        
        let repelConstant = 200.0
        let attractConstant = 0.05
        let damping = 0.85
        let dt = 0.1
        
        // Pre-compute distance matrix to save cycles
        var similarityMatrix = [[Double]](repeating: [Double](repeating: 0.0, count: items.count), count: items.count)
        
        for i in 0..<items.count {
            for j in (i+1)..<items.count {
                let sim = Double(cosineSimilarity(a: items[i].embedding, b: items[j].embedding))
                similarityMatrix[i][j] = sim
                similarityMatrix[j][i] = sim
            }
        }
        
        for _ in 0..<iterations {
            var forces = items.map { _ in SIMD2<Double>(0, 0) }
            
            for i in 0..<items.count {
                for j in 0..<items.count {
                    if i == j { continue }
                    
                    let dir = positions[j] - positions[i]
                    let distSq = max(dir.x * dir.x + dir.y * dir.y, 0.1)
                    let dist = sqrt(distSq)
                    let normal = dir / dist
                    
                    // Repulsion (Coulomb's Law)
                    let repel = -(repelConstant / distSq)
                    forces[i] += normal * repel
                    
                    // Attraction based on Semantic Similarity (Hooke's Law)
                    // High cosine similarity means strong spring attraction.
                    let sim = similarityMatrix[i][j]
                    if sim > 0.5 {
                        // Exaggerate high similarities
                        let attract = attractConstant * (sim * sim) * dist
                        forces[i] += normal * attract
                    }
                }
            }
            
            // Apply bounds pulling to keep them centered
            for i in 0..<items.count {
                let p = positions[i]
                let dist = sqrt(p.x * p.x + p.y * p.y)
                if dist > 300 {
                    forces[i] += -(p / dist) * (dist - 300) * 0.1
                }
            }
            
            // Integrate
            for i in 0..<items.count {
                velocities[i] = (velocities[i] + forces[i] * dt) * damping
                positions[i] += velocities[i] * dt
            }
        }
        
        // Naive cluster assignment (k-means placeholder based on quadrants of space)
        for i in 0..<items.count {
            let p = positions[i]
            let c: Int
            if p.x > 0 && p.y > 0 { c = 0 }
            else if p.x < 0 && p.y > 0 { c = 1 }
            else if p.x < 0 && p.y < 0 { c = 2 }
            else { c = 3 }
            
            items[i] = PhoneItemMeta(
                id: items[i].id,
                type: items[i].type,
                contentText: items[i].contentText,
                x: positions[i].x,
                y: positions[i].y,
                cluster: c,
                embedding: items[i].embedding
            )
        }
    }
}
