import SwiftUI
import MapKit

struct MapView: View {
    @StateObject private var viewModel = NavigationViewModel()
    @State private var region = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 35.6762, longitude: 139.6503),
        span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)
    )
    @State private var showingRadiusSelector = false
    @State private var selectedRadius = 1000
    
    var body: some View {
        ZStack {
            Map(coordinateRegion: $region, annotationItems: viewModel.pois) { poi in
                MapAnnotation(coordinate: poi.coordinate) {
                    POIMarker(poi: poi) {
                        viewModel.selectPOI(poi)
                    }
                }
            }
            .ignoresSafeArea()
            
            VStack {
                if viewModel.isOffline {
                    HStack {
                        Image(systemName: "wifi.slash")
                        Text("You are offline. Reconnect to search POIs.")
                            .font(.caption)
                    }
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(.red.opacity(0.9))
                    .transition(.move(edge: .top))
                }
                
                Spacer()
                
                if !viewModel.pois.isEmpty {
                    VStack(spacing: 0) {
                        HStack {
                            Text("\(viewModel.pois.count) Places Nearby")
                                .font(.headline)
                            Spacer()
                            Button {
                                showingRadiusSelector = true
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "slider.horizontal.3")
                                    Text("\(selectedRadius)m")
                                        .font(.caption)
                                }
                            }
                        }
                        .padding()
                        
                        Divider()
                        
                        ScrollView {
                            LazyVStack(spacing: 0) {
                                ForEach(viewModel.pois) { poi in
                                    POIListRow(poi: poi) {
                                        viewModel.selectPOI(poi)
                                    }
                                    Divider()
                                }
                            }
                        }
                        .frame(maxHeight: 300)
                    }
                    .background(.ultraThinMaterial)
                    .cornerRadius(16, corners: [.topLeft, .topRight])
                    .shadow(radius: 10)
                }
            }
            
            VStack {
                HStack {
                    Spacer()
                    VStack(spacing: 12) {
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
                        
                        Button {
                            Task { await viewModel.refresh() }
                        } label: {
                            Image(systemName: "arrow.clockwise")
                                .foregroundColor(.white)
                                .padding(12)
                                .background(.green)
                                .clipShape(Circle())
                                .shadow(radius: 4)
                        }
                        .disabled(viewModel.isOffline || viewModel.isLoading)
                    }
                    .padding()
                }
                Spacer()
            }
            
            if viewModel.isLoading {
                ProgressView("Searching nearby...")
                    .padding()
                    .background(.ultraThinMaterial)
                    .cornerRadius(12)
            }
            
            if let error = viewModel.errorMessage {
                VStack {
                    Spacer()
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding()
                        .background(.red.opacity(0.8))
                        .cornerRadius(8)
                        .padding()
                }
            }
        }
        .sheet(item: $viewModel.selectedPOI) { poi in
            POIDetailView(poi: poi, viewModel: viewModel)
        }
        .confirmationDialog("Search Radius", isPresented: $showingRadiusSelector, titleVisibility: .visible) {
            Button("500 meters") {
                selectedRadius = 500
                Task { await viewModel.fetchNearbyPOIs(radiusMeters: 500) }
            }
            Button("1 kilometer (Default)") {
                selectedRadius = 1000
                Task { await viewModel.fetchNearbyPOIs(radiusMeters: 1000) }
            }
            Button("2 kilometers") {
                selectedRadius = 2000
                Task { await viewModel.fetchNearbyPOIs(radiusMeters: 2000) }
            }
            Button("5 kilometers") {
                selectedRadius = 5000
                Task { await viewModel.fetchNearbyPOIs(radiusMeters: 5000) }
            }
        }
        .task {
            await viewModel.requestLocationPermission()
            await viewModel.fetchNearbyPOIs()
        }
        .onChange(of: viewModel.userLocation) { newLocation in
            if let location = newLocation {
                withAnimation {
                    region.center = location
                }
            }
        }
    }
    
    private func centerOnUser() {
        if let location = viewModel.userLocation {
            withAnimation {
                region.center = location
            }
        }
    }
}

struct POIMarker: View {
    let poi: POI
    let onTap: () -> Void
    
    var body: some View {
        VStack(spacing: 0) {
            Image(systemName: poi.categoryIcon)
                .font(.system(size: 12))
                .foregroundColor(.white)
                .padding(8)
                .background(categoryColor)
                .clipShape(Circle())
            Triangle()
                .fill(categoryColor)
                .frame(width: 10, height: 6)
                .offset(y: -1)
        }
        .onTapGesture { onTap() }
    }
    
    private var categoryColor: Color {
        switch poi.category.lowercased() {
        case "restaurant": return .orange
        case "transit": return .blue
        case "temple", "shrine": return .purple
        case "lodging", "hotel": return .green
        default: return .red
        }
    }
}

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

struct POIListRow: View {
    let poi: POI
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                Image(systemName: poi.categoryIcon)
                    .font(.title3)
                    .foregroundColor(.blue)
                    .frame(width: 40)
                VStack(alignment: .leading, spacing: 4) {
                    Text(poi.name)
                        .font(.headline)
                        .foregroundColor(.primary)
                    Text(poi.category.capitalized)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Text(poi.distanceFormatted)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
        }
        .buttonStyle(.plain)
    }
}

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
