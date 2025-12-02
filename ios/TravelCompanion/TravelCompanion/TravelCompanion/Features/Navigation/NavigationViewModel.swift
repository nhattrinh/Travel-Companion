import Foundation
import CoreLocation
import Combine
import Network
import UIKit

@MainActor
final class NavigationViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var userLocation: CLLocationCoordinate2D?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var isOffline = false
    
    // Search state
    @Published var destinationQuery: String = ""
    
    // Directions state
    @Published var currentRoute: WalkingRoute?
    @Published var isLoadingDirections = false
    @Published var directionsError: String?
    
    // Drawer state
    @Published var showDirectionsDrawer = false
    @Published var drawerHeight: DrawerHeight = .collapsed
    
    // MARK: - Private Properties
    private let directionsService: DirectionsService
    private let locationManager = LocationManager()
    private let networkMonitor = NWPathMonitor()
    private let monitorQueue = DispatchQueue(label: "NetworkMonitor")
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    nonisolated init(directionsService: DirectionsService = DirectionsService.shared) {
        self.directionsService = directionsService
        // Setup must be done on MainActor
        Task { @MainActor [weak self] in
            self?.setupLocationUpdates()
            self?.setupNetworkMonitoring()
        }
    }
    
    // MARK: - Setup Methods
    private func setupLocationUpdates() {
        locationManager.$currentLocation
            .compactMap { $0 }
            .receive(on: DispatchQueue.main)
            .sink { [weak self] location in
                guard let self else { return }
                self.userLocation = location.coordinate
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
    
    // MARK: - Location Methods
    func requestLocationPermission() async {
        await locationManager.requestPermission()
    }
    
    // MARK: - Directions Methods
    
    /// Get walking directions from current location to destination
    func getDirections() async {
        guard !destinationQuery.isEmpty else {
            directionsError = "Please enter a destination"
            return
        }
        
        guard !isOffline else {
            directionsError = "You are offline. Directions unavailable."
            return
        }
        
        // Use current location or a placeholder
        let startLocation: String
        if let location = userLocation {
            // Use coordinates as start location
            startLocation = "Current location (\(String(format: "%.4f", location.latitude)), \(String(format: "%.4f", location.longitude)))"
        } else {
            startLocation = "Current location"
        }
        
        isLoadingDirections = true
        directionsError = nil
        
        do {
            let route = try await directionsService.getWalkingDirections(
                from: startLocation,
                to: destinationQuery
            )
            currentRoute = route
            showDirectionsDrawer = true
            drawerHeight = .partial
        } catch {
            directionsError = error.localizedDescription
            currentRoute = nil
        }
        
        isLoadingDirections = false
    }
    
    /// Clear current route and hide drawer
    func clearRoute() {
        currentRoute = nil
        showDirectionsDrawer = false
        drawerHeight = .collapsed
        destinationQuery = ""
        directionsError = nil
    }
    
    /// Toggle drawer between collapsed and expanded
    func toggleDrawer() {
        switch drawerHeight {
        case .collapsed:
            drawerHeight = .partial
        case .partial:
            drawerHeight = .expanded
        case .expanded:
            drawerHeight = .partial
        }
    }
    
    deinit {
        networkMonitor.cancel()
    }
}

// MARK: - Drawer Height Enum
enum DrawerHeight: Equatable {
    case collapsed
    case partial
    case expanded
    
    var height: CGFloat {
        let screenHeight = UIScreen.main.bounds.height
        switch self {
        case .collapsed: return 0
        case .partial: return 300
        case .expanded: return screenHeight * 0.75
        }
    }
}

// MARK: - Location Manager
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

// MARK: - Navigation Errors
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
