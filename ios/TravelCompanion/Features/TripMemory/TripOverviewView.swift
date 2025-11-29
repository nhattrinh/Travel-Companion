import SwiftUI

struct TripOverviewView: View {
    @StateObject private var viewModel = TripOverviewViewModel()
    @State private var selectedTab = 0
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Active trip card
                if let activeTrip = viewModel.activeTrip {
                    ActiveTripCard(trip: activeTrip)
                        .padding()
                } else {
                    NoActiveTripBanner {
                        viewModel.showingCreateSheet = true
                    }
                    .padding()
                }
                
                // Tab selector
                Picker("View", selection: $selectedTab) {
                    Text("Stats").tag(0)
                    Text("History").tag(1)
                    Text("All Trips").tag(2)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .padding(.bottom)
                
                // Content
                if viewModel.isLoading && viewModel.trips.isEmpty {
                    Spacer()
                    ProgressView("Loading trips...")
                    Spacer()
                } else {
                    switch selectedTab {
                    case 0:
                        statsView
                    case 1:
                        historyView
                    case 2:
                        allTripsView
                    default:
                        EmptyView()
                    }
                }
                
                // Error banner
                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(.red.opacity(0.8))
                }
            }
            .navigationTitle("My Trips")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        viewModel.showingCreateSheet = true
                    } label: {
                        Image(systemName: "plus.circle.fill")
                    }
                }
            }
            .sheet(isPresented: $viewModel.showingCreateSheet) {
                CreateTripSheet(viewModel: viewModel)
            }
            .refreshable {
                await viewModel.refresh()
            }
            .task {
                await viewModel.fetchTrips()
                await viewModel.fetchRecentTranslations()
            }
        }
    }
    
    private var statsView: some View {
        ScrollView {
            VStack(spacing: 20) {
                if let activeTrip = viewModel.activeTrip, let stats = activeTrip.stats {
                    // Stats cards
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 16) {
                        StatCard(
                            icon: "text.bubble.fill",
                            title: "Translations",
                            value: "\(stats.translationCount)",
                            color: .blue
                        )
                        
                        StatCard(
                            icon: "star.fill",
                            title: "Favorites",
                            value: "\(stats.favoriteCount)",
                            color: .yellow
                        )
                        
                        StatCard(
                            icon: "tag.fill",
                            title: "Contexts",
                            value: "\(stats.uniqueContexts)",
                            color: .purple
                        )
                        
                        StatCard(
                            icon: "globe",
                            title: "Top Language",
                            value: stats.topLanguage?.uppercased() ?? "N/A",
                            color: .green
                        )
                    }
                    .padding()
                    
                    // Recent activity summary
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Recent Activity")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        if !viewModel.recentTranslations.isEmpty {
                            ForEach(viewModel.recentTranslations.prefix(3)) { translation in
                                RecentActivityRow(translation: translation)
                                Divider()
                            }
                        } else {
                            Text("No recent translations")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .padding()
                        }
                    }
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    .padding(.horizontal)
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "chart.bar")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("No active trip")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Create a trip to start tracking your journey")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 60)
                }
            }
        }
    }
    
    private var historyView: some View {
        RecentTranslationsView(viewModel: viewModel)
    }
    
    private var allTripsView: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                if viewModel.trips.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "airplane")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("No trips yet")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Create your first trip to get started")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 60)
                } else {
                    ForEach(viewModel.trips) { trip in
                        TripRow(trip: trip) {
                            Task {
                                await viewModel.fetchTripStats(tripId: trip.id)
                            }
                        }
                        Divider()
                    }
                }
            }
        }
    }
}

struct ActiveTripCard: View {
    let trip: Trip
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: trip.statusIcon)
                    .font(.title2)
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(trip.name)
                        .font(.headline)
                    Text(trip.dateRange)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(trip.isActive ? "Active" : "Completed")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(trip.isActive ? .blue : .green)
                    Text(trip.duration)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            if let stats = trip.stats {
                Divider()
                HStack(spacing: 20) {
                    Label("\(stats.translationCount)", systemImage: "text.bubble")
                        .font(.caption)
                    Label("\(stats.favoriteCount)", systemImage: "star")
                        .font(.caption)
                    if let language = stats.topLanguage {
                        Label(language.uppercased(), systemImage: "globe")
                            .font(.caption)
                    }
                }
                .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.blue.opacity(0.3), lineWidth: 1)
        )
    }
}

struct NoActiveTripBanner: View {
    let onCreate: () -> Void
    
    var body: some View {
        Button(action: onCreate) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("No active trip")
                        .font(.headline)
                    Text("Tap to create a new trip")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Image(systemName: "plus.circle.fill")
                    .font(.title2)
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
    }
}

struct StatCard: View {
    let icon: String
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title)
                .foregroundColor(color)
            
            Text(value)
                .font(.system(size: 32, weight: .bold))
                .foregroundColor(.primary)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct RecentActivityRow: View {
    let translation: TranslationHistory
    
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(translation.translatedText)
                    .font(.body)
                    .lineLimit(1)
                
                Text(translation.originalText)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(translation.timeAgo)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                Text(translation.confidenceText)
                    .font(.caption2)
                    .foregroundColor(.blue)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}

struct TripRow: View {
    let trip: Trip
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                Image(systemName: trip.statusIcon)
                    .font(.title2)
                    .foregroundColor(trip.isActive ? .blue : .gray)
                    .frame(width: 40)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(trip.name)
                        .font(.headline)
                    
                    Text(trip.dateRange)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if let stats = trip.stats {
                        HStack(spacing: 12) {
                            Label("\(stats.translationCount)", systemImage: "text.bubble")
                            Label("\(stats.favoriteCount)", systemImage: "star")
                        }
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(trip.isActive ? "Active" : "Completed")
                        .font(.caption)
                        .foregroundColor(trip.isActive ? .blue : .green)
                    
                    Text(trip.duration)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
        }
        .buttonStyle(.plain)
    }
}

struct CreateTripSheet: View {
    @ObservedObject var viewModel: TripOverviewViewModel
    @Environment(\.dismiss) var dismiss
    
    @State private var tripName = ""
    @State private var startDate = Date()
    @State private var endDate = Date().addingTimeInterval(7 * 24 * 60 * 60) // 7 days later
    
    var body: some View {
        NavigationView {
            Form {
                Section("Trip Details") {
                    TextField("Trip Name", text: $tripName)
                    
                    DatePicker("Start Date", selection: $startDate, displayedComponents: .date)
                    
                    DatePicker("End Date", selection: $endDate, in: startDate..., displayedComponents: .date)
                }
                
                Section {
                    Button {
                        Task {
                            await viewModel.createTrip(
                                name: tripName,
                                startDate: startDate,
                                endDate: endDate
                            )
                        }
                    } label: {
                        HStack {
                            Spacer()
                            if viewModel.isLoading {
                                ProgressView()
                            } else {
                                Text("Create Trip")
                                    .fontWeight(.semibold)
                            }
                            Spacer()
                        }
                    }
                    .disabled(tripName.isEmpty || viewModel.isLoading)
                }
            }
            .navigationTitle("New Trip")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}
