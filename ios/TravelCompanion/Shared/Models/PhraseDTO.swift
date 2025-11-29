import Foundation

/// Phrase with translations for different contexts
struct Phrase: Codable, Identifiable {
    let id: UUID
    let context: String
    let originalText: String
    let translations: [String: String]
    let isFavorite: Bool
    
    init(id: UUID = UUID(), context: String, originalText: String, translations: [String: String], isFavorite: Bool = false) {
        self.id = id
        self.context = context
        self.originalText = originalText
        self.translations = translations
        self.isFavorite = isFavorite
    }
    
    func translation(for language: String) -> String {
        translations[language] ?? originalText
    }
    
    var contextIcon: String {
        switch context.lowercased() {
        case "restaurant", "dining":
            return "fork.knife"
        case "transit", "transportation":
            return "tram.fill"
        case "hotel", "lodging":
            return "bed.double.fill"
        case "shopping":
            return "cart.fill"
        case "emergency":
            return "exclamationmark.triangle.fill"
        case "greeting":
            return "hand.wave.fill"
        default:
            return "text.bubble.fill"
        }
    }
    
    var contextColor: String {
        switch context.lowercased() {
        case "restaurant", "dining":
            return "orange"
        case "transit", "transportation":
            return "blue"
        case "hotel", "lodging":
            return "green"
        case "shopping":
            return "purple"
        case "emergency":
            return "red"
        case "greeting":
            return "yellow"
        default:
            return "gray"
        }
    }
}

/// API response for phrase suggestions
struct PhraseSuggestionResponse: Codable {
    let status: String
    let data: PhraseSuggestionData?
    let error: String?
}

struct PhraseSuggestionData: Codable {
    let phrases: [PhraseDTO]
    let context: String
}

struct PhraseDTO: Codable {
    let id: Int
    let context: String
    let originalText: String
    let translations: [String: String]
    
    func toPhrase(isFavorite: Bool = false) -> Phrase {
        Phrase(
            id: UUID(),
            context: context,
            originalText: originalText,
            translations: translations,
            isFavorite: isFavorite
        )
    }
}

/// API response for favorites
struct FavoritesResponse: Codable {
    let status: String
    let data: FavoritesData?
    let error: String?
}

struct FavoritesData: Codable {
    let favorites: [FavoriteDTO]
}

struct FavoriteDTO: Codable {
    let id: Int
    let itemType: String
    let itemId: Int
    let phrase: PhraseDTO?
    
    func toPhrase() -> Phrase? {
        guard let phrase = phrase else { return nil }
        return phrase.toPhrase(isFavorite: true)
    }
}

/// Request to add/remove favorite
struct FavoriteRequest: Codable {
    let itemType: String
    let itemId: Int
}

/// Response after adding favorite
struct FavoriteActionResponse: Codable {
    let status: String
    let data: FavoriteActionData?
    let error: String?
}

struct FavoriteActionData: Codable {
    let favoriteId: Int
}
