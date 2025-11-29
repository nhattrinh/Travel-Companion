import Foundation

/// Translation segment with bounding box and text
struct TranslationSegment: Codable, Identifiable {
    let id: UUID
    let originalText: String
    let translatedText: String
    let confidence: Double
    let boundingBox: BoundingBox
    
    init(id: UUID = UUID(), originalText: String, translatedText: String, confidence: Double, boundingBox: BoundingBox) {
        self.id = id
        self.originalText = originalText
        self.translatedText = translatedText
        self.confidence = confidence
        self.boundingBox = boundingBox
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
    let detectedLanguage: String?
    let processingTimeMs: Double?
}

struct TranslationSegmentDTO: Codable {
    let originalText: String
    let translatedText: String
    let confidence: Double
    let boundingBox: BoundingBoxDTO
    
    func toSegment() -> TranslationSegment {
        TranslationSegment(
            originalText: originalText,
            translatedText: translatedText,
            confidence: confidence,
            boundingBox: BoundingBox(
                x: boundingBox.x,
                y: boundingBox.y,
                width: boundingBox.width,
                height: boundingBox.height
            )
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
