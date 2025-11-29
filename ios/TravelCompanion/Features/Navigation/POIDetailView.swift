import SwiftUI
import MapKit

struct POIDetailView: View {
    let poi: POI
    @ObservedObject var viewModel: NavigationViewModel
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    Map(coordinateRegion: .constant(
                        MKCoordinateRegion(
                            center: poi.coordinate,
                            span: MKCoordinateSpan(latitudeDelta: 0.005, longitudeDelta: 0.005)
                        )
                    ), annotationItems: [poi]) { poi in
                        MapAnnotation(coordinate: poi.coordinate) {
                            Image(systemName: poi.categoryIcon)
                                .foregroundColor(.white)
                                .padding(12)
                                .background(.red)
                                .clipShape(Circle())
                        }
                    }
                    .frame(height: 200)
                    .cornerRadius(12)
                    .padding(.horizontal)
                    
                    VStack(alignment: .leading, spacing: 16) {
                        HStack {
                            Image(systemName: poi.categoryIcon)
                            Text(poi.category.capitalized)
                        }
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.blue)
                        .cornerRadius(8)
                        
                        HStack {
                            Image(systemName: "location.fill")
                                .foregroundColor(.secondary)
                            Text(poi.distanceFormatted)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Text("away")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        Divider()
                        
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Image(systemName: "info.circle.fill")
                                    .foregroundColor(.blue)
                                Text("Cultural Etiquette")
                                    .font(.headline)
                            }
                            
                            if !poi.etiquetteNotes.isEmpty {
                                ForEach(etiquetteRules, id: \.self) { rule in
                                    HStack(alignment: .top, spacing: 8) {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(.green)
                                            .font(.caption)
                                        Text(rule)
                                            .font(.subheadline)
                                            .fixedSize(horizontal: false, vertical: true)
                                    }
                                }
                            } else {
                                Text("No specific etiquette notes for this location.")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                        
                        if poi.category.lowercased() == "restaurant" {
                            ContextHintCard(
                                icon: "fork.knife",
                                title: "Dining Tips",
                                hints: [
                                    "Say 'itadakimasu' before eating",
                                    "Slurping noodles is acceptable",
                                    "Don't leave chopsticks standing in rice"
                                ]
                            )
                        } else if poi.category.lowercased() == "transit" {
                            ContextHintCard(
                                icon: "tram.fill",
                                title: "Transit Tips",
                                hints: [
                                    "Queue in orderly lines",
                                    "Keep voice low on trains",
                                    "No phone calls in trains"
                                ]
                            )
                        }
                    }
                    .padding(.horizontal)
                    
                    Spacer(minLength: 20)
                    
                    VStack(spacing: 12) {
                        Button {
                            openInMaps()
                        } label: {
                            Label("Get Directions", systemImage: "arrow.triangle.turn.up.right.circle.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button {
                            shareLocation()
                        } label: {
                            Label("Share Location", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                                .padding()
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                }
            }
            .navigationTitle(poi.name)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }
    
    private var etiquetteRules: [String] {
        poi.etiquetteNotes
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
    
    private func openInMaps() {
        let mapItem = MKMapItem(placemark: MKPlacemark(coordinate: poi.coordinate))
        mapItem.name = poi.name
        mapItem.openInMaps(launchOptions: [
            MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeWalking
        ])
    }
    
    private func shareLocation() {
        let text = "\(poi.name) - \(poi.category)"
        let url = URL(string: "https://maps.apple.com/?ll=\(poi.latitude),\(poi.longitude)&q=\(poi.name.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")")!
        let activityVC = UIActivityViewController(activityItems: [text, url], applicationActivities: nil)
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootVC = windowScene.windows.first?.rootViewController {
            rootVC.present(activityVC, animated: true)
        }
    }
}

struct ContextHintCard: View {
    let icon: String
    let title: String
    let hints: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.orange)
                Text(title)
                    .font(.headline)
            }
            ForEach(hints, id: \.self) { hint in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "lightbulb.fill")
                        .foregroundColor(.orange)
                        .font(.caption)
                    Text(hint)
                        .font(.subheadline)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
    }
}
