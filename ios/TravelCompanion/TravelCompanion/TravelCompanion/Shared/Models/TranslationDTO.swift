import Foundation

/// Translation segment with bounding box and text
struct TranslationSegment: Codable, Identifiable {
    let id: UUID
    let originalText: String
    let translatedText: String
    let confidence: Double
    let boundingBox: BoundingBox
    var itemType: String = "food"  // "food" or "price"
    var price: String? = nil  // Associated price for food items
    var photoURL: URL? = nil  // URL for food image
    
    init(
        id: UUID = UUID(),
        originalText: String,
        translatedText: String,
        confidence: Double,
        boundingBox: BoundingBox,
        itemType: String = "food",
        price: String? = nil,
        photoURL: URL? = nil
    ) {
        self.id = id
        self.originalText = originalText
        self.translatedText = translatedText
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.itemType = itemType
        self.price = price
        self.photoURL = photoURL
    }
}

/// Bounding box coordinates (normalized 0-1)
struct BoundingBox: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

/// API response for live frame translation
struct LiveFrameResponse: Codable {
    let status: String
    let data: LiveFrameData?
    let error: String?
}

struct LiveFrameData: Codable {
    let segments: [TranslationSegmentDTO]
    let sourceLanguage: String?
    let targetLanguage: String?
    let latencyMs: Double?
    
    // Map backend field name
    var detectedLanguage: String? { sourceLanguage }
    var processingTimeMs: Double? { latencyMs }
}

struct TranslationSegmentDTO: Codable {
    // Backend field names
    let text: String
    let translated: String
    let confidence: Double
    let x1: Int
    let y1: Int
    let x2: Int
    let y2: Int
    let itemType: String?
    let price: String?
    let photoUrl: String?
    
    enum CodingKeys: String, CodingKey {
        case text, translated, confidence, x1, y1, x2, y2
        case itemType = "item_type"
        case price
        case photoUrl = "photo_url"
    }
    
    func toSegment() -> TranslationSegment {
        // Calculate width and height from coordinates
        let width = Double(x2 - x1)
        let height = Double(y2 - y1)
        
        return TranslationSegment(
            originalText: text,
            translatedText: translated,
            confidence: confidence,
            boundingBox: BoundingBox(
                x: Double(x1),
                y: Double(y1),
                width: width,
                height: height
            ),
            itemType: itemType ?? "food",
            price: price,
            photoURL: photoUrl.flatMap { URL(string: $0) }
        )
    }
}

struct BoundingBoxDTO: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

/// Request to save translation
struct SaveTranslationRequest: Codable {
    let originalText: String
    let translatedText: String
    let sourceLanguage: String
    let targetLanguage: String
    let confidence: Double?
}

/// Saved translation response
struct SavedTranslationResponse: Codable {
    let status: String
    let data: SavedTranslationData?
    let error: String?
}

struct SavedTranslationData: Codable {
    let id: Int
    let originalText: String
    let translatedText: String
    let timestamp: String
}

/// Text translation response
struct TextTranslationResponse: Codable {
    let status: String
    let data: TextTranslationData?
    let error: String?
}

struct TextTranslationData: Codable {
    let originalText: String
    let translatedText: String
    let sourceLanguage: String
    let targetLanguage: String
    let confidence: Double
}
