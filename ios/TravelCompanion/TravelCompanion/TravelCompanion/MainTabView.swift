import SwiftUI
import Combine

struct MainTabView: View {
    @EnvironmentObject var authState: AuthState
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Menu (Camera options)
            MenuView()
                .tabItem {
                    Label("Visualize", systemImage: "square.grid.2x2.fill")
                }
                .tag(0)
            
            // Navigation (Map)
            MapView()
                .tabItem {
                    Label("Navigate", systemImage: "map.fill")
                }
                .tag(1)
            
            // Live Translate (Voice)
            LiveTranslateView()
                .tabItem {
                    Label("Translate", systemImage: "waveform.circle.fill")
                }
                .tag(2)
            
            // Settings
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape.fill")
                }
                .tag(3)
        }
        .accentColor(.blue)
    }
}

#Preview {
    MainTabView()
        .environmentObject(AuthState())
}
