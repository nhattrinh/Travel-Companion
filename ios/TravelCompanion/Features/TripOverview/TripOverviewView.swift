import SwiftUI

struct TripOverviewView: View {
    @StateObject private var viewModel = TripOverviewViewModel()
    @State private var showCreateTrip = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    if viewModel.isLoading {
                        ProgressView("Loading trip...")
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else if let errorMessage = viewModel.errorMessage {
                        ErrorBanner(message: errorMessage)
                    } else if let trip = viewModel.activeTrip {
                        activeTripCard(trip: trip)
                        
                        if let summary = viewModel.tripSummary {
                            statsSection(summary: summary)
                            recentTranslationsSection(summary: summary)
                        }
                    } else {
                        noActiveTripView
                    }
                }
                .padding()
            }
            .navigationTitle("Trip Overview")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if viewModel.activeTrip == nil {
                        Button(action: { showCreateTrip = true }) {
                            Image(systemName: "plus.circle")
                        }
                    }
                }
            }
            .sheet(isPresented: $showCreateTrip) {
                CreateTripSheet(viewModel: viewModel, isPresented: $showCreateTrip)
            }
            .task {
                await viewModel.loadActiveTrip()
            }
        }
    }
    
    // MARK: - Active Trip Card
    private func activeTripCard(trip: TripDetail) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(trip.destination)
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    HStack {
                        Image(systemName: "calendar")
                            .font(.caption)
                        Text(trip.startDateFormatted)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        if let endDate = trip.endDateFormatted {
                            Text("→ \(endDate)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                
                Spacer()
                
                StatusBadge(status: trip.status)
            }
            
            if trip.status == "active" {
                Button(action: {
                    Task {
                        await viewModel.completeTrip(tripId: trip.id)
                    }
                }) {
                    Text("Complete Trip")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(Color.blue)
                        .cornerRadius(8)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
    
    // MARK: - Stats Section
    private func statsSection(summary: TripSummary) -> some View {
        HStack(spacing: 20) {
            StatCard(
                icon: "text.bubble",
                title: "Translations",
                count: summary.translation_count,
                color: .blue
            )
            
            StatCard(
                icon: "star.fill",
                title: "Favorites",
                count: summary.favorite_count,
                color: .yellow
            )
        }
    }
    
    // MARK: - Recent Translations
    private func recentTranslationsSection(summary: TripSummary) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Translations")
                .font(.headline)
            
            if summary.recent_translations.isEmpty {
                Text("No translations yet")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(summary.recent_translations) { translation in
                    TranslationRowView(translation: translation)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
    
    // MARK: - No Active Trip
    private var noActiveTripView: some View {
        VStack(spacing: 20) {
            Image(systemName: "globe.americas")
                .font(.system(size: 64))
                .foregroundColor(.secondary)
            
            Text("No Active Trip")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Create a trip to start tracking translations and favorites")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            
            Button(action: { showCreateTrip = true }) {
                Text("Create Trip")
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 12)
                    .background(Color.blue)
                    .cornerRadius(8)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Supporting Views
struct StatusBadge: View {
    let status: String
    
    var color: Color {
        switch status {
        case "active": return .green
        case "completed": return .gray
        case "archived": return .orange
        default: return .secondary
        }
    }
    
    var body: some View {
        Text(status.capitalized)
            .font(.caption)
            .fontWeight(.semibold)
            .foregroundColor(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(color)
            .cornerRadius(12)
    }
}

struct StatCard: View {
    let icon: String
    let title: String
    let count: Int
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title)
                .foregroundColor(color)
            
            Text("\(count)")
                .font(.title2)
                .fontWeight(.bold)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}

struct TranslationRowView: View {
    let translation: RecentTranslation
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(translation.source_text)
                    .font(.body)
                    .lineLimit(1)
                Spacer()
                Text(translation.createdAtFormatted)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(translation.target_text)
                .font(.subheadline)
                .foregroundColor(.blue)
                .lineLimit(1)
            
            Text("\(translation.source_language.uppercased()) → \(translation.target_language.uppercased())")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(8)
    }
}

struct ErrorBanner: View {
    let message: String
    
    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle")
            Text(message)
                .font(.subheadline)
            Spacer()
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(8)
    }
}

struct CreateTripSheet: View {
    @ObservedObject var viewModel: TripOverviewViewModel
    @Binding var isPresented: Bool
    
    @State private var destination = ""
    @State private var startDate = Date()
    @State private var endDate = Date().addingTimeInterval(7 * 24 * 60 * 60) // 7 days from now
    @State private var includeEndDate = true
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Trip Details")) {
                    TextField("Destination", text: $destination)
                    
                    DatePicker("Start Date", selection: $startDate, displayedComponents: .date)
                    
                    Toggle("Include End Date", isOn: $includeEndDate)
                    
                    if includeEndDate {
                        DatePicker("End Date", selection: $endDate, displayedComponents: .date)
                    }
                }
            }
            .navigationTitle("Create Trip")
            .navigationBarItems(
                leading: Button("Cancel") {
                    isPresented = false
                },
                trailing: Button("Create") {
                    Task {
                        await viewModel.createTrip(
                            destination: destination,
                            startDate: startDate,
                            endDate: includeEndDate ? endDate : nil
                        )
                        isPresented = false
                    }
                }
                .disabled(destination.isEmpty)
            )
        }
    }
}

#Preview {
    TripOverviewView()
}
