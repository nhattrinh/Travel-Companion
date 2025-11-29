import Foundation
import Combine

@MainActor
final class TripOverviewViewModel: ObservableObject {
    @Published var trips: [Trip] = []
    @Published var activeTrip: Trip?
    @Published var recentTranslations: [TranslationHistory] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var showingCreateSheet = false
    
    private let apiClient: APIClient
    private let authService: AuthService
    
    init(apiClient: APIClient = .shared, authService: AuthService = .shared) {
        self.apiClient = apiClient
        self.authService = authService
    }
    
    /// Fetch all trips
    func fetchTrips() async {
        isLoading = true
        errorMessage = nil
        
        do {
            guard let token = try await authService.getAccessToken() else {
                throw TripError.unauthorized
            }
            
            let response = try await apiClient.fetchTrips(token: token)
            
            guard response.status == "ok", let data = response.data else {
                throw TripError.apiError(response.error ?? "Unknown error")
            }
            
            trips = data.trips.map { $0.toTrip() }
            activeTrip = data.activeTrip?.toTrip()
            
            // Fetch stats for active trip
            if let active = activeTrip {
                await fetchTripStats(tripId: active.id)
            }
            
        } catch {
            errorMessage = error.localizedDescription
            trips = []
        }
        
        isLoading = false
    }
    
    /// Fetch trip statistics
    func fetchTripStats(tripId: Int) async {
        do {
            guard let token = try await authService.getAccessToken() else {
                throw TripError.unauthorized
            }
            
            let response = try await apiClient.fetchTripSummary(tripId: tripId, token: token)
            
            guard response.status == "ok", let data = response.data else {
                throw TripError.apiError(response.error ?? "Unknown error")
            }
            
            // Update active trip with stats
            if let index = trips.firstIndex(where: { $0.id == tripId }) {
                trips[index] = data.trip.toTrip(stats: data.stats)
            }
            
            if activeTrip?.id == tripId {
                activeTrip = data.trip.toTrip(stats: data.stats)
            }
            
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    /// Fetch recent translations
    func fetchRecentTranslations(limit: Int = 20) async {
        do {
            guard let token = try await authService.getAccessToken() else {
                throw TripError.unauthorized
            }
            
            let response = try await apiClient.fetchTranslationHistory(
                tripId: activeTrip?.id,
                limit: limit,
                token: token
            )
            
            guard response.status == "ok", let data = response.data else {
                throw TripError.apiError(response.error ?? "Unknown error")
            }
            
            recentTranslations = data.translations.map { $0.toTranslationHistory() }
            
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    /// Create new trip
    func createTrip(name: String, startDate: Date, endDate: Date) async {
        isLoading = true
        errorMessage = nil
        
        do {
            guard let token = try await authService.getAccessToken() else {
                throw TripError.unauthorized
            }
            
            let dateFormatter = ISO8601DateFormatter()
            dateFormatter.formatOptions = [.withFullDate, .withDashSeparatorInDate]
            
            let request = CreateTripRequest(
                name: name,
                startDate: dateFormatter.string(from: startDate),
                endDate: dateFormatter.string(from: endDate)
            )
            
            let response = try await apiClient.createTrip(request: request, token: token)
            
            guard response.status == "ok", let data = response.data else {
                throw TripError.apiError(response.error ?? "Failed to create trip")
            }
            
            // Reload trips to get updated list
            await fetchTrips()
            
            showingCreateSheet = false
            
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    /// Delete translation from history
    func deleteTranslation(_ translation: TranslationHistory) async {
        do {
            guard let token = try await authService.getAccessToken() else {
                throw TripError.unauthorized
            }
            
            _ = try await apiClient.deleteTranslation(
                translationId: translation.id,
                token: token
            )
            
            // Remove from local list
            recentTranslations.removeAll { $0.id == translation.id }
            
            // Refresh stats if in active trip
            if let activeTrip = activeTrip {
                await fetchTripStats(tripId: activeTrip.id)
            }
            
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    /// Copy translation to clipboard
    func copyTranslation(_ translation: TranslationHistory) {
        UIPasteboard.general.string = translation.translatedText
    }
    
    /// Refresh all data
    func refresh() async {
        await fetchTrips()
        await fetchRecentTranslations()
    }
}

enum TripError: LocalizedError {
    case unauthorized
    case apiError(String)
    
    var errorDescription: String? {
        switch self {
        case .unauthorized:
            return "Authentication required"
        case .apiError(let message):
            return message
        }
    }
}
