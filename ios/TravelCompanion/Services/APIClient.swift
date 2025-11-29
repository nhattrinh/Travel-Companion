import Foundation

final class APIClient {
    static let shared = APIClient()
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func get(path: String) async throws -> Data {
        let url = Environment.apiBaseURL.appendingPathComponent(path)
        let (data, _) = try await session.data(from: url)
        return data
    }
    
    /// POST translation frame for live camera translation
    func postTranslationFrame(imageBase64: String, targetLanguage: String, token: String) async throws -> LiveFrameResponse {
        let url = Environment.translationBaseURL.appendingPathComponent("live-frame")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = [
            "image_base64": imageBase64,
            "target_language": targetLanguage
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(LiveFrameResponse.self, from: data)
    }
    
    /// Save translation to history
    func saveTranslation(_ request: SaveTranslationRequest, token: String) async throws -> SavedTranslationResponse {
        let url = Environment.translationBaseURL.appendingPathComponent("save")
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        urlRequest.httpBody = try encoder.encode(request)
        
        let (data, response) = try await session.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(SavedTranslationResponse.self, from: data)
    }
    
    /// Fetch nearby POIs
    func fetchPOIs(latitude: Double, longitude: Double, radiusMeters: Int, token: String) async throws -> POIResponse {
        var components = URLComponents(url: Environment.navigationBaseURL.appendingPathComponent("pois"), resolvingAgainstBaseURL: true)!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(latitude)),
            URLQueryItem(name: "longitude", value: String(longitude)),
            URLQueryItem(name: "radius_m", value: String(radiusMeters))
        ]
        
        guard let url = components.url else {
            throw APIError.invalidResponse
        }
        
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(SavedTranslationResponse.self, from: data)
    }
    
    /// Fetch nearby POIs
    func fetchPOIs(latitude: Double, longitude: Double, radiusMeters: Int, token: String) async throws -> POIResponse {
        var components = URLComponents(url: Environment.navigationBaseURL.appendingPathComponent("pois"), resolvingAgainstBaseURL: true)!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(latitude)),
            URLQueryItem(name: "longitude", value: String(longitude)),
            URLQueryItem(name: "radius_m", value: String(radiusMeters))
        ]
        
        guard let url = components.url else {
            throw APIError.invalidResponse
        }
        
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(POIResponse.self, from: data)
    }
    
    /// Fetch phrase suggestions by context
    func fetchPhraseSuggestions(context: String, targetLanguage: String, token: String) async throws -> PhraseSuggestionResponse {
        var components = URLComponents(url: Environment.phrasebookBaseURL.appendingPathComponent("suggestions"), resolvingAgainstBaseURL: true)!
        components.queryItems = [
            URLQueryItem(name: "context", value: context),
            URLQueryItem(name: "target_language", value: targetLanguage),
            URLQueryItem(name: "limit", value: "20")
        ]
        
        guard let url = components.url else {
            throw APIError.invalidResponse
        }
        
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(PhraseSuggestionResponse.self, from: data)
    }
    
    /// Fetch user favorites
    func fetchFavorites(itemType: String, token: String) async throws -> FavoritesResponse {
        var components = URLComponents(url: Environment.favoritesBaseURL, resolvingAgainstBaseURL: true)!
        components.queryItems = [
            URLQueryItem(name: "item_type", value: itemType)
        ]
        
        guard let url = components.url else {
            throw APIError.invalidResponse
        }
        
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FavoritesResponse.self, from: data)
    }
    
    /// Add item to favorites
    func addFavorite(request: FavoriteRequest, token: String) async throws -> FavoriteActionResponse {
        let url = Environment.favoritesBaseURL
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        urlRequest.httpBody = try encoder.encode(request)
        
        let (data, response) = try await session.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 || httpResponse.statusCode == 201 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FavoriteActionResponse.self, from: data)
    }
    
    /// Remove item from favorites
    func removeFavorite(favoriteId: Int, token: String) async throws -> FavoriteActionResponse {
        let url = Environment.favoritesBaseURL.appendingPathComponent("\(favoriteId)")
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FavoriteActionResponse.self, from: data)
    }
    
    /// Fetch all trips
    func fetchTrips(token: String) async throws -> TripListResponse {
        let url = Environment.tripsBaseURL
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(TripListResponse.self, from: data)
    }
    
    /// Fetch trip summary with stats
    func fetchTripSummary(tripId: Int, token: String) async throws -> TripSummaryResponse {
        let url = Environment.tripsBaseURL.appendingPathComponent("\(tripId)/summary")
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(TripSummaryResponse.self, from: data)
    }
    
    /// Create new trip
    func createTrip(request: CreateTripRequest, token: String) async throws -> CreateTripResponse {
        let url = Environment.tripsBaseURL
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        urlRequest.httpBody = try encoder.encode(request)
        
        let (data, response) = try await session.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 || httpResponse.statusCode == 201 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(CreateTripResponse.self, from: data)
    }
    
    /// Fetch translation history
    func fetchTranslationHistory(tripId: Int?, limit: Int, token: String) async throws -> TranslationHistoryResponse {
        var components = URLComponents(url: Environment.translationBaseURL.appendingPathComponent("history"), resolvingAgainstBaseURL: true)!
        var queryItems = [URLQueryItem(name: "limit", value: String(limit))]
        if let tripId = tripId {
            queryItems.append(URLQueryItem(name: "trip_id", value: String(tripId)))
        }
        components.queryItems = queryItems
        
        guard let url = components.url else {
            throw APIError.invalidResponse
        }
        
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(TranslationHistoryResponse.self, from: data)
    }
    
    /// Delete translation from history
    func deleteTranslation(translationId: Int, token: String) async throws -> FavoriteActionResponse {
        let url = Environment.translationBaseURL.appendingPathComponent("\(translationId)")
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FavoriteActionResponse.self, from: data)
    }
    
    /// Delete user account and all associated data
    func deleteAccount(token: String) async throws -> FavoriteActionResponse {
        let url = Environment.authBaseURL.appendingPathComponent("account")
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FavoriteActionResponse.self, from: data)
    }
    
    /// Delete user account and all associated data
    func deleteAccount(token: String) async throws -> FavoriteActionResponse {
        let url = Environment.authBaseURL.appendingPathComponent("account")
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FavoriteActionResponse.self, from: data)
    }
}

enum APIError: LocalizedError {
    case invalidResponse
    case httpError(Int)
    
    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid server response"
        case .httpError(let code):
            return "HTTP error: \(code)"
        }
    }
}
