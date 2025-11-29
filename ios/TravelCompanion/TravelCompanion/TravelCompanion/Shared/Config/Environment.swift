import Foundation

/// Environment configuration placeholder.
/// Values will be loaded from bundled plist or remote config in later phases.
enum AppEnvironment {
    // Use localhost for simulator, use Mac's IP for physical device
    #if targetEnvironment(simulator)
    static let apiBaseURL: URL = URL(string: "http://localhost:8000")!
    #else
    // Replace with your Mac's local IP address for physical device testing
    static let apiBaseURL: URL = URL(string: "http://192.168.0.191:8000")!
    #endif
    
    static let authBaseURL: URL = apiBaseURL.appendingPathComponent("auth")
    static let translationBaseURL: URL = apiBaseURL.appendingPathComponent("translation")
    static let navigationBaseURL: URL = apiBaseURL.appendingPathComponent("navigation")
    static let phrasebookBaseURL: URL = apiBaseURL.appendingPathComponent("phrases")
    static let favoritesBaseURL: URL = apiBaseURL.appendingPathComponent("favorites")
    static let tripsBaseURL: URL = apiBaseURL.appendingPathComponent("trips")
}
