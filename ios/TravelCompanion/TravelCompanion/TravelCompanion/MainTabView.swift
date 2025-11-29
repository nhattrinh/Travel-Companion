import SwiftUI
import Combine

struct MainTabView: View {
    @EnvironmentObject var authState: AuthState
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Translation (Camera)
            CameraView()
                .tabItem {
                    Label("Translate", systemImage: "camera.fill")
                }
                .tag(0)
            
            // Navigation (Map)
            MapView()
                .tabItem {
                    Label("Navigate", systemImage: "map.fill")
                }
                .tag(1)
            
            // Phrasebook
            PhraseListView()
                .tabItem {
                    Label("Phrases", systemImage: "text.bubble.fill")
                }
                .tag(2)
            
            // Trip Memory
            TripOverviewView()
                .tabItem {
                    Label("Trips", systemImage: "airplane.circle.fill")
                }
                .tag(3)
            
            // Settings
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape.fill")
                }
                .tag(4)
        }
        .accentColor(.blue)
    }
}

#Preview {
    MainTabView()
        .environmentObject(AuthState())
}
