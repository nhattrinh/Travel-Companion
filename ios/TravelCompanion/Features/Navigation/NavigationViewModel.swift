import Foundation
import CoreLocation
import Combine
import Network

@MainActor
final class NavigationViewModel: ObservableObject {
    @Published var pois: [POI] = []
    @Published var userLocation: CLLocationCoordinate2D?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var selectedPOI: POI?
    @Published var isOffline = false
    
    private let apiClient: APIClient
    private let authService: AuthService
    private let locationManager = LocationManager()
    private let networkMonitor = NWPathMonitor()
    private let monitorQueue = DispatchQueue(label: "NetworkMonitor")
    private var cancellables = Set<AnyCancellable>()
    
    init(apiClient: APIClient = .shared, authService: AuthService = .shared) {
        self.apiClient = apiClient
        self.authService = authService
        setupLocationUpdates()
        setupNetworkMonitoring()
    }
    
    private func setupLocationUpdates() {
        locationManager.$currentLocation
            .compactMap { $0 }
            .sink { [weak self] location in
                self?.userLocation = location.coordinate
            }
            .store(in: &cancellables)
    }
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor in
                self?.isOffline = path.status != .satisfied
            }
        }
        networkMonitor.start(queue: monitorQueue)
    }
    
    func requestLocationPermission() async {
        await locationManager.requestPermission()
    }
    
    func fetchNearbyPOIs(radiusMeters: Int = 1000) async {
        guard !isOffline else {
            errorMessage = "You are offline. POI search unavailable."
            return
        }
        
        guard let location = userLocation else {
            errorMessage = "Location not available. Please enable location services."
            return
        }
        
        isLoading = true
        errorMessage = nil
        
        do {
            guard let token = try await authService.getAccessToken() else {
                throw NavigationError.unauthorized
            }
            
            let response = try await apiClient.fetchPOIs(
                latitude: location.latitude,
                longitude: location.longitude,
                radiusMeters: radiusMeters,
                token: token
            )
            
            guard response.status == "ok", let data = response.data else {
                throw NavigationError.apiError(response.error ?? "Unknown error")
            }
            
            pois = data.pois.map { $0.toPOI() }
            
        } catch {
            errorMessage = error.localizedDescription
            pois = []
        }
        
        isLoading = false
    }
    
    func selectPOI(_ poi: POI) {
        selectedPOI = poi
    }
    
    func clearSelection() {
        selectedPOI = nil
    }
    
    func refresh() async {
        await fetchNearbyPOIs()
    }
    
    deinit {
        networkMonitor.cancel()
    }
}

@MainActor
final class LocationManager: NSObject, ObservableObject {
    @Published var currentLocation: CLLocation?
    @Published var authorizationStatus: CLAuthorizationStatus = .notDetermined
    
    private let manager = CLLocationManager()
    
    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = 10
        authorizationStatus = manager.authorizationStatus
    }
    
    func requestPermission() async {
        if authorizationStatus == .notDetermined {
            manager.requestWhenInUseAuthorization()
        }
    }
    
    func startUpdating() {
        manager.startUpdatingLocation()
    }
    
    func stopUpdating() {
        manager.stopUpdatingLocation()
    }
}

extension LocationManager: CLLocationManagerDelegate {
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        Task { @MainActor in
            self.currentLocation = location
        }
    }
    
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            self.authorizationStatus = manager.authorizationStatus
            if self.authorizationStatus == .authorizedWhenInUse || self.authorizationStatus == .authorizedAlways {
                self.startUpdating()
            }
        }
    }
    
    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error.localizedDescription)")
    }
}

enum NavigationError: LocalizedError {
    case unauthorized
    case apiError(String)
    case locationUnavailable
    
    var errorDescription: String? {
        switch self {
        case .unauthorized:
            return "Authentication required"
        case .apiError(let message):
            return message
        case .locationUnavailable:
            return "Location services unavailable"
        }
    }
}
