import SwiftUI
import Combine
import MapKit

struct MapView: View {
    @StateObject private var viewModel = NavigationViewModel()
    @State private var region = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 35.6762, longitude: 139.6503),
        span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
    )
    @FocusState private var isSearchFocused: Bool
    
    var body: some View {
        ZStack(alignment: .bottom) {
            // Map with route overlay
            mapWithRouteOverlay
                .ignoresSafeArea()
            
            // Top controls overlay
            VStack {
                // Offline banner
                if viewModel.isOffline {
                    offlineBanner
                }
                
                // Search bar and Get Directions button
                searchSection
                
                Spacer()
            }
            
            // Floating controls
            floatingControls
            
            // Loading overlay
            if viewModel.isLoadingDirections {
                loadingOverlay
            }
            
            // Error message
            if let error = viewModel.directionsError {
                errorBanner(error)
            }
            
            // Directions drawer
            if viewModel.showDirectionsDrawer, let route = viewModel.currentRoute {
                DirectionsDrawer(
                    route: route,
                    drawerHeight: $viewModel.drawerHeight,
                    onClose: { viewModel.clearRoute() }
                )
                .transition(.move(edge: .bottom))
            }
        }
        .task {
            await viewModel.requestLocationPermission()
        }
        .onReceive(viewModel.$userLocation) { newLocation in
            if let location = newLocation, viewModel.currentRoute == nil {
                withAnimation {
                    region.center = location
                }
            }
        }
        .onReceive(viewModel.$currentRoute) { route in
            if let route = route, !route.routeCoordinates.isEmpty {
                // Adjust map to show the entire route
                adjustMapToShowRoute(route)
            }
        }
    }
    
    // MARK: - Map with Route Overlay
    private var mapWithRouteOverlay: some View {
        Map(coordinateRegion: $region, annotationItems: routeAnnotations) { annotation in
            MapAnnotation(coordinate: annotation.coordinate) {
                RouteMarker(annotation: annotation)
            }
        }
    }
    
    // MARK: - Route Annotations
    private var routeAnnotations: [RouteAnnotation] {
        guard let route = viewModel.currentRoute else { return [] }
        return route.waypoints.compactMap { waypoint -> RouteAnnotation? in
            guard let coord = waypoint.coordinate else { return nil }
            return RouteAnnotation(
                id: waypoint.id,
                coordinate: coord,
                name: waypoint.name,
                type: waypoint.type,
                icon: waypoint.icon
            )
        }
    }
    
    // MARK: - Search Section
    private var searchSection: some View {
        VStack(spacing: 12) {
            // Destination search field
            HStack(spacing: 12) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    
                    TextField("Where do you want to go?", text: $viewModel.destinationQuery)
                        .textFieldStyle(.plain)
                        .focused($isSearchFocused)
                        .submitLabel(.search)
                        .onSubmit {
                            if !viewModel.destinationQuery.isEmpty {
                                Task {
                                    await viewModel.getDirections()
                                }
                            }
                        }
                    
                    if !viewModel.destinationQuery.isEmpty {
                        Button {
                            viewModel.destinationQuery = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(12)
                .background(.ultraThinMaterial)
                .cornerRadius(12)
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
            }
            
            // Get Directions button - only show when there's text
            if !viewModel.destinationQuery.isEmpty {
                Button {
                    isSearchFocused = false
                    Task {
                        await viewModel.getDirections()
                    }
                } label: {
                    HStack {
                        Image(systemName: "figure.walk")
                        Text("Get Walking Directions")
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.blue)
                    .cornerRadius(12)
                    .shadow(color: .blue.opacity(0.3), radius: 8, x: 0, y: 4)
                }
                .disabled(viewModel.isLoadingDirections || viewModel.isOffline)
                .transition(.opacity.combined(with: .scale(scale: 0.95)))
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
        .animation(.easeInOut(duration: 0.2), value: viewModel.destinationQuery.isEmpty)
    }
    
    // MARK: - Floating Controls
    private var floatingControls: some View {
        VStack {
            HStack {
                Spacer()
                VStack(spacing: 12) {
                    // Center on user button
                    Button {
                        centerOnUser()
                    } label: {
                        Image(systemName: "location.fill")
                            .foregroundColor(.white)
                            .padding(12)
                            .background(.blue)
                            .clipShape(Circle())
                            .shadow(radius: 4)
                    }
                    
                    // Clear route button (if route exists)
                    if viewModel.currentRoute != nil {
                        Button {
                            withAnimation {
                                viewModel.clearRoute()
                            }
                        } label: {
                            Image(systemName: "xmark")
                                .foregroundColor(.white)
                                .padding(12)
                                .background(.red)
                                .clipShape(Circle())
                                .shadow(radius: 4)
                        }
                    }
                }
                .padding()
            }
            Spacer()
        }
        .padding(.top, 120) // Below search bar
    }
    
    // MARK: - Offline Banner
    private var offlineBanner: some View {
        HStack {
            Image(systemName: "wifi.slash")
            Text("You are offline. Directions unavailable.")
                .font(.caption)
        }
        .foregroundColor(.white)
        .padding()
        .frame(maxWidth: .infinity)
        .background(.red.opacity(0.9))
        .transition(.move(edge: .top))
    }
    
    // MARK: - Loading Overlay
    private var loadingOverlay: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)
            Text("Getting walking directions...")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(24)
        .background(.ultraThinMaterial)
        .cornerRadius(16)
        .shadow(radius: 10)
    }
    
    // MARK: - Error Banner
    private func errorBanner(_ message: String) -> some View {
        VStack {
            Spacer()
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.yellow)
                Text(message)
                    .font(.caption)
                Spacer()
                Button {
                    viewModel.directionsError = nil
                } label: {
                    Image(systemName: "xmark")
                        .font(.caption)
                }
            }
            .foregroundColor(.white)
            .padding()
            .background(.red.opacity(0.9))
            .cornerRadius(12)
            .padding()
            .padding(.bottom, viewModel.showDirectionsDrawer ? viewModel.drawerHeight.height : 0)
        }
    }
    
    // MARK: - Helper Methods
    private func centerOnUser() {
        if let location = viewModel.userLocation {
            withAnimation {
                region.center = location
            }
        }
    }
    
    private func adjustMapToShowRoute(_ route: WalkingRoute) {
        let coordinates = route.routeCoordinates
        guard !coordinates.isEmpty else { return }
        
        let lats = coordinates.map { $0.latitude }
        let lngs = coordinates.map { $0.longitude }
        
        let minLat = lats.min() ?? 0
        let maxLat = lats.max() ?? 0
        let minLng = lngs.min() ?? 0
        let maxLng = lngs.max() ?? 0
        
        let center = CLLocationCoordinate2D(
            latitude: (minLat + maxLat) / 2,
            longitude: (minLng + maxLng) / 2
        )
        
        let span = MKCoordinateSpan(
            latitudeDelta: max((maxLat - minLat) * 1.5, 0.01),
            longitudeDelta: max((maxLng - minLng) * 1.5, 0.01)
        )
        
        withAnimation {
            region = MKCoordinateRegion(center: center, span: span)
        }
    }
}

// MARK: - Route Annotation Model
struct RouteAnnotation: Identifiable {
    let id: String
    let coordinate: CLLocationCoordinate2D
    let name: String
    let type: WaypointType
    let icon: String
}

// MARK: - Route Marker View
struct RouteMarker: View {
    let annotation: RouteAnnotation
    
    var body: some View {
        VStack(spacing: 0) {
            Image(systemName: annotation.icon)
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(.white)
                .padding(10)
                .background(markerColor)
                .clipShape(Circle())
                .shadow(radius: 4)
            
            Triangle()
                .fill(markerColor)
                .frame(width: 12, height: 8)
                .offset(y: -2)
        }
    }
    
    private var markerColor: Color {
        switch annotation.type {
        case .start: return .green
        case .intermediate: return .blue
        case .end: return .red
        }
    }
}

// MARK: - Triangle Shape
struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        Path { path in
            path.move(to: CGPoint(x: rect.midX, y: rect.maxY))
            path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
            path.closeSubpath()
        }
    }
}

// MARK: - Corner Radius Extension
extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners
    
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}

#Preview {
    MapView()
}
