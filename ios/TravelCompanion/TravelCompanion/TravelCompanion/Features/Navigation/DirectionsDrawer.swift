import SwiftUI

/// A slide-up drawer that displays walking directions
struct DirectionsDrawer: View {
    let route: WalkingRoute
    @Binding var drawerHeight: DrawerHeight
    let onClose: () -> Void
    
    @State private var dragOffset: CGFloat = 0
    @GestureState private var isDragging = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Handle bar
            handleBar
            
            // Header with route info
            routeHeader
            
            Divider()
            
            // Scrollable directions content
            if drawerHeight != .collapsed {
                directionsContent
            }
        }
        .frame(height: max(0, drawerHeight.height + dragOffset))
        .background(.ultraThinMaterial)
        .cornerRadius(20, corners: [.topLeft, .topRight])
        .shadow(color: .black.opacity(0.15), radius: 10, x: 0, y: -5)
        .gesture(dragGesture)
        .animation(.interactiveSpring(response: 0.4, dampingFraction: 0.8), value: drawerHeight)
    }
    
    // MARK: - Handle Bar
    private var handleBar: some View {
        VStack(spacing: 8) {
            Capsule()
                .fill(Color.secondary.opacity(0.4))
                .frame(width: 40, height: 5)
                .padding(.top, 8)
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .onTapGesture {
            toggleDrawerHeight()
        }
    }
    
    // MARK: - Route Header
    private var routeHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(route.routeName)
                        .font(.headline)
                        .lineLimit(2)
                    
                    HStack(spacing: 16) {
                        Label(route.totalDistanceFormatted, systemImage: "ruler")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Label(route.totalDurationFormatted, systemImage: "clock")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                Button {
                    onClose()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundColor(.secondary)
                }
            }
            
            // Safety note
            if !route.safetyNote.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "info.circle.fill")
                        .foregroundColor(.orange)
                        .font(.caption)
                    Text(route.safetyNote)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(8)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
    }
    
    // MARK: - Directions Content
    private var directionsContent: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(route.directions) { segment in
                    DirectionSegmentRow(segment: segment, waypoints: route.waypoints)
                    
                    if segment.id != route.directions.last?.id {
                        Divider()
                            .padding(.leading, 48)
                    }
                }
                
                // Route notes at the bottom
                if !route.notes.isEmpty {
                    routeNotes
                }
            }
            .padding(.bottom, 20)
        }
    }
    
    // MARK: - Route Notes
    private var routeNotes: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "note.text")
                    .foregroundColor(.blue)
                Text("Notes")
                    .font(.subheadline)
                    .fontWeight(.semibold)
            }
            
            Text(route.notes)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
        .padding(.horizontal)
        .padding(.top, 12)
    }
    
    // MARK: - Drag Gesture
    private var dragGesture: some Gesture {
        DragGesture()
            .updating($isDragging) { _, state, _ in
                state = true
            }
            .onChanged { value in
                let newOffset = -value.translation.height
                dragOffset = newOffset
            }
            .onEnded { value in
                let velocity = -value.predictedEndTranslation.height
                let threshold: CGFloat = 100
                
                withAnimation(.interactiveSpring(response: 0.4, dampingFraction: 0.8)) {
                    if velocity > threshold {
                        // Swipe up - expand
                        expandDrawer()
                    } else if velocity < -threshold {
                        // Swipe down - collapse or close
                        collapseDrawer()
                    } else {
                        // Snap to nearest position based on current drag
                        snapToNearestPosition()
                    }
                    dragOffset = 0
                }
            }
    }
    
    // MARK: - Helper Methods
    private func toggleDrawerHeight() {
        withAnimation(.interactiveSpring(response: 0.4, dampingFraction: 0.8)) {
            switch drawerHeight {
            case .collapsed:
                drawerHeight = .partial
            case .partial:
                drawerHeight = .expanded
            case .expanded:
                drawerHeight = .partial
            }
        }
    }
    
    private func expandDrawer() {
        switch drawerHeight {
        case .collapsed:
            drawerHeight = .partial
        case .partial:
            drawerHeight = .expanded
        case .expanded:
            break
        }
    }
    
    private func collapseDrawer() {
        switch drawerHeight {
        case .collapsed:
            break
        case .partial:
            drawerHeight = .collapsed
            onClose()
        case .expanded:
            drawerHeight = .partial
        }
    }
    
    private func snapToNearestPosition() {
        let currentHeight = drawerHeight.height + dragOffset
        let positions: [(DrawerHeight, CGFloat)] = [
            (.collapsed, DrawerHeight.collapsed.height),
            (.partial, DrawerHeight.partial.height),
            (.expanded, DrawerHeight.expanded.height)
        ]
        
        let nearest = positions.min(by: { abs($0.1 - currentHeight) < abs($1.1 - currentHeight) })
        
        if let nearest = nearest {
            if nearest.0 == .collapsed {
                onClose()
            }
            drawerHeight = nearest.0
        }
    }
}

// MARK: - Direction Segment Row
struct DirectionSegmentRow: View {
    let segment: DirectionSegment
    let waypoints: [Waypoint]
    
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Segment header
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(alignment: .top, spacing: 12) {
                    // Segment number circle
                    ZStack {
                        Circle()
                            .fill(Color.blue)
                            .frame(width: 32, height: 32)
                        
                        Text("\(segment.segmentIndex)")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                    }
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text(segment.summary)
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                            .multilineTextAlignment(.leading)
                        
                        HStack(spacing: 12) {
                            if !segment.distanceFormatted.isEmpty {
                                Label(segment.distanceFormatted, systemImage: "arrow.forward")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            if !segment.durationFormatted.isEmpty {
                                Label(segment.durationFormatted, systemImage: "clock")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    Spacer()
                    
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
            }
            .buttonStyle(.plain)
            
            // Expanded steps
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(segment.steps.enumerated()), id: \.offset) { index, step in
                        HStack(alignment: .top, spacing: 12) {
                            // Step indicator
                            Circle()
                                .fill(Color.blue.opacity(0.3))
                                .frame(width: 8, height: 8)
                                .padding(.top, 5)
                            
                            Text(step)
                                .font(.caption)
                                .foregroundColor(.primary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    
                    // Checkpoint info
                    HStack(spacing: 8) {
                        Image(systemName: "flag.fill")
                            .foregroundColor(.green)
                            .font(.caption)
                        
                        Text("Checkpoint: \(segment.checkpoint.name)")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.green)
                    }
                    .padding(.top, 4)
                }
                .padding(.leading, 44)
                .padding(.trailing)
                .padding(.bottom, 12)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}

// MARK: - Preview
#Preview {
    VStack {
        Spacer()
        DirectionsDrawer(
            route: WalkingRoute(
                routeName: "Tokyo Skytree to Tokyo Tower",
                startLocationQuery: "Tokyo Skytree",
                endLocationQuery: "Tokyo Tower",
                notes: "This is a scenic route through central Tokyo.",
                totalDistanceKm: 8.5,
                estimatedDurationMin: 105,
                safetyNote: "This is a long walk. Consider bringing water and checking the weather.",
                waypoints: [
                    Waypoint(id: "start", name: "Tokyo Skytree", lat: 35.7101, lng: 139.8107, description: "Starting point", type: .start),
                    Waypoint(id: "w1", name: "Senso-ji Temple", lat: 35.7148, lng: 139.7967, description: "Famous temple", type: .intermediate),
                    Waypoint(id: "end", name: "Tokyo Tower", lat: 35.6586, lng: 139.7454, description: "Destination", type: .end)
                ],
                directions: [
                    DirectionSegment(
                        segmentIndex: 1,
                        fromId: "start",
                        toId: "w1",
                        summary: "Walk from Tokyo Skytree to Senso-ji Temple",
                        distanceEstimateKm: 1.2,
                        estimatedDurationMin: 15,
                        steps: [
                            "Exit Tokyo Skytree and head west",
                            "Turn left on Kototoi-dori Avenue",
                            "Continue straight for 800 meters",
                            "You will see Senso-ji Temple on your right"
                        ],
                        checkpoint: Checkpoint(name: "Senso-ji Temple", lat: 35.7148, lng: 139.7967)
                    )
                ]
            ),
            drawerHeight: .constant(.partial),
            onClose: {}
        )
    }
}
