import Foundation
import CoreLocation

/// Point of Interest with location and context information
struct POI: Codable, Identifiable {
    let id: UUID
    let name: String
    let category: String
    let latitude: Double
    let longitude: Double
    let etiquetteNotes: String
    let distanceMeters: Double
    
    var coordinate: CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }
    
    var distanceFormatted: String {
        if distanceMeters < 1000 {
            return String(format: "%.0f m", distanceMeters)
        } else {
            return String(format: "%.1f km", distanceMeters / 1000)
        }
    }
    
    var categoryIcon: String {
        switch category.lowercased() {
        case "restaurant":
            return "fork.knife"
        case "transit":
            return "tram.fill"
        case "temple", "shrine":
            return "building.columns"
        case "lodging", "hotel":
            return "bed.double.fill"
        case "shopping":
            return "cart.fill"
        case "park":
            return "leaf.fill"
        default:
            return "mappin.circle.fill"
        }
    }
    
    init(id: UUID = UUID(), name: String, category: String, latitude: Double, longitude: Double, etiquetteNotes: String, distanceMeters: Double) {
        self.id = id
        self.name = name
        self.category = category
        self.latitude = latitude
        self.longitude = longitude
        self.etiquetteNotes = etiquetteNotes
        self.distanceMeters = distanceMeters
    }
}

/// API response for POI search
struct POIResponse: Codable {
    let status: String
    let data: POIData?
    let error: String?
}

struct POIData: Codable {
    let pois: [POIDTO]
    let userLocation: LocationDTO?
}

struct POIDTO: Codable {
    let name: String
    let category: String
    let latitude: Double
    let longitude: Double
    let etiquetteNotes: String
    let distanceM: Double
    
    func toPOI() -> POI {
        POI(
            name: name,
            category: category,
            latitude: latitude,
            longitude: longitude,
            etiquetteNotes: etiquetteNotes,
            distanceMeters: distanceM
        )
    }
}

struct LocationDTO: Codable {
    let latitude: Double
    let longitude: Double
}
