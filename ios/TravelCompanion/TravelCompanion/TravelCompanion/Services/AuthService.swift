import Foundation

final class AuthService {
    static let shared = AuthService()
    private init() {}

    private let client = APIClient.shared

    struct Envelope<T: Decodable>: Decodable { let status: String; let data: T?; let error: String? }
    struct TokenPayload: Decodable { let token: Tokens; let user: UserDTO? }
    struct Tokens: Decodable { 
        let access_token: String
        let refresh_token: String
        let token_type: String?  // Backend returns this field
    }

    enum AuthError: Error { case server(String); case decoding; case unauthorized }

    @discardableResult
    func register(email: String, password: String) async throws -> UserDTO {
        let body: [String: Any] = ["email": email, "password": password]
        let data = try await post(path: "/auth/register", json: body)
        let envelope = try JSONDecoder().decode(Envelope<TokenPayload>.self, from: data)
        guard envelope.status == "ok", let payload = envelope.data, let user = payload.user else { throw AuthError.server(envelope.error ?? "unknown") }
        persist(tokens: payload.token)
        return user
    }

    @discardableResult
    func login(email: String, password: String) async throws -> UserDTO {
        let body: [String: Any] = ["email": email, "password": password]
        let data = try await post(path: "/auth/login", json: body)
        
        // Debug: print raw response
        if let jsonString = String(data: data, encoding: .utf8) {
            print("Login response: \(jsonString)")
        }
        
        do {
            let envelope = try JSONDecoder().decode(Envelope<TokenPayload>.self, from: data)
            guard envelope.status == "ok", let payload = envelope.data, let user = payload.user else { 
                throw AuthError.server(envelope.error ?? "unknown") 
            }
            persist(tokens: payload.token)
            return user
        } catch {
            print("Decoding error: \(error)")
            throw error
        }
    }

    func refresh() async throws {
        guard let refresh = KeychainTokenStore.shared.refreshToken() else { throw AuthError.unauthorized }
        let body: [String: Any] = ["refresh_token": refresh]
        let data = try await post(path: "/auth/refresh", json: body)
        struct RefreshOnly: Decodable { let token: Tokens }
        let envelope = try JSONDecoder().decode(Envelope<RefreshOnly>.self, from: data)
        guard envelope.status == "ok", let payload = envelope.data else { throw AuthError.server(envelope.error ?? "unknown") }
        persist(tokens: payload.token)
    }

    private func persist(tokens: Tokens) {
        KeychainTokenStore.shared.save(access: tokens.access_token, refresh: tokens.refresh_token)
    }

    private func post(path: String, json: [String: Any]) async throws -> Data {
        let url = AppEnvironment.apiBaseURL.appendingPathComponent(path)
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: json)
        let (data, _) = try await URLSession.shared.data(for: req)
        return data
    }
    
    /// Get access token, refreshing if needed
    func getAccessToken() async throws -> String {
        guard let token = KeychainTokenStore.shared.accessToken() else {
            throw AuthError.unauthorized
        }
        // TODO: Add token expiration check and refresh logic
        return token
    }
}
