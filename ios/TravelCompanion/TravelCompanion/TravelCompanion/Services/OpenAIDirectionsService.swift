import Foundation
import CoreLocation

/// Service for getting walking directions from backend
/// The backend handles the OpenAI API call
final class DirectionsService {
    static let shared = DirectionsService()
    
    private let session: URLSession
    
    private init(session: URLSession = .shared) {
        self.session = session
    }
    
    /// Get walking directions between two locations via backend API
    func getWalkingDirections(from startLocation: String, to endLocation: String) async throws -> WalkingRoute {
        let url = AppEnvironment.navigationBaseURL.appendingPathComponent("directions")
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Build request body
        let requestBody = DirectionsRequest(
            startLocation: startLocation,
            endLocation: endLocation
        )
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        request.httpBody = try encoder.encode(requestBody)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw DirectionsError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            throw DirectionsError.httpError(httpResponse.statusCode)
        }
        
        // Decode the envelope response
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let envelopeResponse = try decoder.decode(DirectionsEnvelopeResponse.self, from: data)
        
        // Check for API errors
        guard envelopeResponse.status == "ok" else {
            throw DirectionsError.apiError(envelopeResponse.error ?? "Unknown error")
        }
        
        // Extract the route data
        guard let route = envelopeResponse.data else {
            throw DirectionsError.noContent
        }
        
        return route
    }
}

// MARK: - Request Models

struct DirectionsRequest: Encodable {
    let startLocation: String
    let endLocation: String
}

// MARK: - Response Models

struct DirectionsEnvelopeResponse: Decodable {
    let status: String
    let data: WalkingRoute?
    let error: String?
}

// MARK: - Errors

enum DirectionsError: LocalizedError {
    case invalidResponse
    case httpError(Int)
    case apiError(String)
    case noContent
    case invalidJSON
    
    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "Server error: \(code)"
        case .apiError(let message):
            return message
        case .noContent:
            return "No directions found"
        case .invalidJSON:
            return "Could not parse directions response"
        }
    }
}

// Keep the old name as a typealias for backward compatibility
typealias OpenAIDirectionsService = DirectionsService
