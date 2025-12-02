import Foundation

/// Response model for image search
struct ImageSearchResponse: Decodable {
    let status: String
    let images: [String]
    let error: String?
}

/// Service for fetching location images from the backend
final class ImageSearchService {
    static let shared = ImageSearchService()
    
    private let session: URLSession
    private var imageCache: [String: [URL]] = [:]
    
    private init(session: URLSession = .shared) {
        self.session = session
    }
    
    /// Search for images of a location/landmark
    /// - Parameters:
    ///   - query: Search term (e.g., "Golden Gate Bridge")
    ///   - maxImages: Maximum number of images to return (default 3)
    /// - Returns: Array of image URLs
    func searchImages(query: String, maxImages: Int = 3) async throws -> [URL] {
        // Check cache first
        let cacheKey = "\(query.lowercased()):\(maxImages)"
        if let cached = imageCache[cacheKey] {
            return cached
        }
        
        let url = AppEnvironment.navigationBaseURL.appendingPathComponent("images")
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = [
            "query": query,
            "max_images": maxImages
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            return []
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let imageResponse = try decoder.decode(ImageSearchResponse.self, from: data)
        
        guard imageResponse.status == "ok" else {
            return []
        }
        
        // Convert strings to URLs
        let urls = imageResponse.images.compactMap { URL(string: $0) }
        
        // Cache the results
        imageCache[cacheKey] = urls
        
        return urls
    }
    
    /// Clear the image cache
    func clearCache() {
        imageCache.removeAll()
    }
}
