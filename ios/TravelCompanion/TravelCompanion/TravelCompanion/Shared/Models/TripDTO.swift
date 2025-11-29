import Foundation

/// Trip with date range and analytics
struct Trip: Codable, Identifiable {
    let id: Int
    let name: String
    let startDate: Date
    let endDate: Date
    let isActive: Bool
    let stats: TripStats?
    
    var dateRange: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return "\(formatter.string(from: startDate)) - \(formatter.string(from: endDate))"
    }
    
    var duration: String {
        let calendar = Calendar.current
        let components = calendar.dateComponents([.day], from: startDate, to: endDate)
        let days = (components.day ?? 0) + 1
        return "\(days) day\(days == 1 ? "" : "s")"
    }
    
    var statusIcon: String {
        isActive ? "airplane.circle.fill" : "checkmark.circle.fill"
    }
    
    var statusColor: String {
        isActive ? "blue" : "green"
    }
}

/// Trip statistics
struct TripStats: Codable {
    let translationCount: Int
    let favoriteCount: Int
    let uniqueContexts: Int
    let topLanguage: String?
    
    var translationCountText: String {
        "\(translationCount) translation\(translationCount == 1 ? "" : "s")"
    }
    
    var favoriteCountText: String {
        "\(favoriteCount) favorite\(favoriteCount == 1 ? "" : "s")"
    }
}

/// API response for trip list
struct TripListResponse: Codable {
    let status: String
    let data: TripListData?
    let error: String?
}

struct TripListData: Codable {
    let trips: [TripDTO]
    let activeTrip: TripDTO?
}

struct TripDTO: Codable {
    let id: Int
    let name: String
    let startDate: String
    let endDate: String
    let isActive: Bool
    let createdAt: String
    
    func toTrip(stats: TripStats? = nil) -> Trip {
        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withFullDate, .withDashSeparatorInDate]
        
        return Trip(
            id: id,
            name: name,
            startDate: dateFormatter.date(from: startDate) ?? Date(),
            endDate: dateFormatter.date(from: endDate) ?? Date(),
            isActive: isActive,
            stats: stats
        )
    }
}

/// API response for trip summary
struct TripSummaryResponse: Codable {
    let status: String
    let data: TripSummaryData?
    let error: String?
}

struct TripSummaryData: Codable {
    let trip: TripDTO
    let stats: TripStats
}

/// Request to create trip
struct CreateTripRequest: Codable {
    let name: String
    let startDate: String
    let endDate: String
}

/// Response after creating trip
struct CreateTripResponse: Codable {
    let status: String
    let data: CreateTripData?
    let error: String?
}

struct CreateTripData: Codable {
    let tripId: Int
    let trip: TripDTO
}

/// Translation history entry
struct TranslationHistory: Codable, Identifiable {
    let id: Int
    let originalText: String
    let translatedText: String
    let sourceLanguage: String
    let targetLanguage: String
    let confidence: Double
    let imageUrl: String?
    let createdAt: Date
    let tripId: Int?
    
    var confidenceText: String {
        String(format: "%.0f%%", confidence * 100)
    }
    
    var timeAgo: String {
        let interval = Date().timeIntervalSince(createdAt)
        
        if interval < 60 {
            return "Just now"
        } else if interval < 3600 {
            let minutes = Int(interval / 60)
            return "\(minutes)m ago"
        } else if interval < 86400 {
            let hours = Int(interval / 3600)
            return "\(hours)h ago"
        } else {
            let days = Int(interval / 86400)
            return "\(days)d ago"
        }
    }
}

/// API response for translation history
struct TranslationHistoryResponse: Codable {
    let status: String
    let data: TranslationHistoryData?
    let error: String?
}

struct TranslationHistoryData: Codable {
    let translations: [TranslationHistoryDTO]
    let total: Int
}

struct TranslationHistoryDTO: Codable {
    let id: Int
    let originalText: String
    let translatedText: String
    let sourceLanguage: String
    let targetLanguage: String
    let confidence: Double
    let imageUrl: String?
    let createdAt: String
    let tripId: Int?
    
    func toTranslationHistory() -> TranslationHistory {
        let dateFormatter = ISO8601DateFormatter()
        
        return TranslationHistory(
            id: id,
            originalText: originalText,
            translatedText: translatedText,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            confidence: confidence,
            imageUrl: imageUrl,
            createdAt: dateFormatter.date(from: createdAt) ?? Date(),
            tripId: tripId
        )
    }
}
