import Foundation

/// Environment configuration placeholder.
/// Values will be loaded from bundled plist or remote config in later phases.
enum Environment {
    static let apiBaseURL: URL = URL(string: "https://api.example.com")!
    static let authBaseURL: URL = apiBaseURL.appendingPathComponent("auth")
    static let translationBaseURL: URL = apiBaseURL.appendingPathComponent("translation")
    static let navigationBaseURL: URL = apiBaseURL.appendingPathComponent("navigation")
    static let phrasebookBaseURL: URL = apiBaseURL.appendingPathComponent("phrases")
    static let favoritesBaseURL: URL = apiBaseURL.appendingPathComponent("favorites")
    static let tripsBaseURL: URL = apiBaseURL.appendingPathComponent("trips")
}
