import SwiftUI
import Combine

@MainActor
class TripOverviewViewModel: ObservableObject {
    @Published var activeTrip: TripDetail?
    @Published var tripSummary: TripSummary?
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    
    private let apiClient: APIClient
    
    init(apiClient: APIClient = .shared) {
        self.apiClient = apiClient
    }
    
    func loadActiveTrip() async {
        isLoading = true
        errorMessage = nil
        
        do {
            let response: Envelope<TripDetail?> = try await apiClient.request(
                endpoint: "/trips/active",
                method: "GET"
            )
            
            if response.status == "ok" {
                activeTrip = response.data ?? nil
                
                // Load summary if trip exists
                if let trip = activeTrip {
                    await loadTripSummary(tripId: trip.id)
                }
            } else if let error = response.error {
                errorMessage = error
            }
        } catch {
            errorMessage = "Failed to load active trip: \(error.localizedDescription)"
        }
        
        isLoading = false
    }
    
    func loadTripSummary(tripId: Int) async {
        do {
            let response: Envelope<TripSummary> = try await apiClient.request(
                endpoint: "/trips/\(tripId)/summary",
                method: "GET"
            )
            
            if response.status == "ok", let data = response.data {
                tripSummary = data
            }
        } catch {
            errorMessage = "Failed to load trip summary: \(error.localizedDescription)"
        }
    }
    
    func createTrip(destination: String, startDate: Date, endDate: Date?) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let tripData: [String: Any] = [
                "destination": destination,
                "start_date": ISO8601DateFormatter().string(from: startDate),
                "end_date": endDate.map { ISO8601DateFormatter().string(from: $0) } as Any
            ]
            
            let response: Envelope<TripDetail> = try await apiClient.request(
                endpoint: "/trips",
                method: "POST",
                body: tripData
            )
            
            if response.status == "ok", let data = response.data {
                activeTrip = data
                await loadTripSummary(tripId: data.id)
            } else if let error = response.error {
                errorMessage = error
            }
        } catch {
            errorMessage = "Failed to create trip: \(error.localizedDescription)"
        }
        
        isLoading = false
    }
    
    func completeTrip(tripId: Int) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let response: Envelope<TripDetail> = try await apiClient.request(
                endpoint: "/trips/\(tripId)/complete",
                method: "POST"
            )
            
            if response.status == "ok", let data = response.data {
                activeTrip = data.status == "active" ? data : nil
            } else if let error = response.error {
                errorMessage = error
            }
        } catch {
            errorMessage = "Failed to complete trip: \(error.localizedDescription)"
        }
        
        isLoading = false
    }
}

// MARK: - Models
struct TripDetail: Codable, Identifiable {
    let id: Int
    let user_id: Int
    let destination: String
    let start_date: String
    let end_date: String?
    let status: String
    let metadata: [String: String]?
    let created_at: String
    let updated_at: String
    
    var startDateFormatted: String {
        guard let date = ISO8601DateFormatter().date(from: start_date) else {
            return start_date
        }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
    
    var endDateFormatted: String? {
        guard let endDateStr = end_date,
              let date = ISO8601DateFormatter().date(from: endDateStr) else {
            return nil
        }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
}

struct TripSummary: Codable {
    let trip: TripDetail
    let translation_count: Int
    let favorite_count: Int
    let recent_translations: [RecentTranslation]
}

struct RecentTranslation: Codable, Identifiable {
    let id: Int
    let source_text: String
    let target_text: String
    let source_language: String
    let target_language: String
    let created_at: String
    
    var createdAtFormatted: String {
        guard let date = ISO8601DateFormatter().date(from: created_at) else {
            return created_at
        }
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}
