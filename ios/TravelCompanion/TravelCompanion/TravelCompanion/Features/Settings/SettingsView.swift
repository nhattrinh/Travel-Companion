import SwiftUI
import Combine

struct SettingsView: View {
    @EnvironmentObject var authState: AuthState
    @StateObject private var viewModel = SettingsViewModel()
    @State private var showingLogoutConfirmation = false
    @State private var showingDeleteAccountConfirmation = false
    
    var body: some View {
        NavigationView {
            List {
                // User Profile Section
                Section {
                    HStack {
                        Image(systemName: "person.circle.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.blue)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Travel User")
                                .font(.headline)
                            Text("user@example.com")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.leading, 8)
                    }
                    .padding(.vertical, 8)
                }
                
                // Language Preferences
                Section("Preferences") {
                    Picker("Default Translation Language", selection: $viewModel.defaultLanguage) {
                        ForEach(viewModel.availableLanguages, id: \.0) { code, name in
                            Text(name).tag(code)
                        }
                    }
                    
                    Toggle("Auto-save Translations", isOn: $viewModel.autoSaveTranslations)
                    
                    Toggle("Offline Mode", isOn: $viewModel.offlineMode)
                        .disabled(true) // Placeholder for future feature
                }
                
                // App Settings
                Section("App") {
                    NavigationLink {
                        NotificationSettingsView()
                    } label: {
                        Label("Notifications", systemImage: "bell.fill")
                    }
                    
                    NavigationLink {
                        CacheManagementView()
                    } label: {
                        HStack {
                            Label("Storage & Cache", systemImage: "internaldrive")
                            Spacer()
                            Text(viewModel.cacheSize)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    NavigationLink {
                        AboutView()
                    } label: {
                        Label("About", systemImage: "info.circle")
                    }
                }
                
                // Privacy & Security
                Section("Privacy & Security") {
                    NavigationLink {
                        PrivacySettingsView()
                    } label: {
                        Label("Privacy Policy", systemImage: "hand.raised.fill")
                    }
                    
                    Button {
                        showingDeleteAccountConfirmation = true
                    } label: {
                        Label("Delete Account", systemImage: "trash.fill")
                            .foregroundColor(.red)
                    }
                }
                
                // Account Actions
                Section {
                    Button {
                        showingLogoutConfirmation = true
                    } label: {
                        HStack {
                            Spacer()
                            Text("Log Out")
                                .foregroundColor(.red)
                            Spacer()
                        }
                    }
                }
                
                // App Version
                Section {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    .font(.caption)
                }
            }
            .navigationTitle("Settings")
            .confirmationDialog("Log Out", isPresented: $showingLogoutConfirmation, titleVisibility: .visible) {
                Button("Log Out", role: .destructive) {
                    authState.logout()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Are you sure you want to log out?")
            }
            .confirmationDialog("Delete Account", isPresented: $showingDeleteAccountConfirmation, titleVisibility: .visible) {
                Button("Delete Account", role: .destructive) {
                    Task {
                        await viewModel.deleteAccount()
                        authState.logout()
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This action cannot be undone. All your data will be permanently deleted.")
            }
        }
    }
}

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var defaultLanguage: String = "ja"
    @Published var autoSaveTranslations: Bool = true
    @Published var offlineMode: Bool = false
    @Published var cacheSize: String = "12.4 MB"
    
    let availableLanguages = [
        ("en", "English"),
        ("ja", "Japanese"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("zh", "Chinese"),
        ("ko", "Korean")
    ]
    
    private let apiClient: APIClient
    private let authService: AuthService
    
    init(apiClient: APIClient = .shared, authService: AuthService = .shared) {
        self.apiClient = apiClient
        self.authService = authService
        loadPreferences()
    }
    
    func loadPreferences() {
        // Load from UserDefaults or API
        defaultLanguage = UserDefaults.standard.string(forKey: "defaultLanguage") ?? "ja"
        autoSaveTranslations = UserDefaults.standard.bool(forKey: "autoSaveTranslations")
    }
    
    func savePreferences() {
        UserDefaults.standard.set(defaultLanguage, forKey: "defaultLanguage")
        UserDefaults.standard.set(autoSaveTranslations, forKey: "autoSaveTranslations")
    }
    
    func deleteAccount() async {
        do {
            let token = try await authService.getAccessToken()
            
            _ = try await apiClient.deleteAccount(token: token)
            
        } catch {
            print("Delete account error: \\(error)")
        }
    }
}

// MARK: - Supporting Views

struct NotificationSettingsView: View {
    @State private var translationNotifications = true
    @State private var tripReminders = true
    @State private var dailyTips = false
    
    var body: some View {
        List {
            Toggle("Translation Saved", isOn: $translationNotifications)
            Toggle("Trip Reminders", isOn: $tripReminders)
            Toggle("Daily Travel Tips", isOn: $dailyTips)
        }
        .navigationTitle("Notifications")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct CacheManagementView: View {
    @State private var showingClearConfirmation = false
    
    var body: some View {
        List {
            Section {
                HStack {
                    Text("Cached Translations")
                    Spacer()
                    Text("8.2 MB")
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Text("Cached Images")
                    Spacer()
                    Text("4.2 MB")
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Text("Total")
                        .fontWeight(.semibold)
                    Spacer()
                    Text("12.4 MB")
                        .fontWeight(.semibold)
                }
            }
            
            Section {
                Button {
                    showingClearConfirmation = true
                } label: {
                    HStack {
                        Spacer()
                        Text("Clear Cache")
                            .foregroundColor(.red)
                        Spacer()
                    }
                }
            }
        }
        .navigationTitle("Storage & Cache")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog("Clear Cache", isPresented: $showingClearConfirmation) {
            Button("Clear Cache", role: .destructive) {
                // Clear cache logic
            }
        } message: {
            Text("This will clear all cached data. You may need to re-download some content.")
        }
    }
}

struct PrivacySettingsView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Privacy Policy")
                    .font(.title2)
                    .fontWeight(.bold)
                    .padding(.bottom, 8)
                
                Text("Your privacy is important to us. This app collects the following data:")
                    .font(.body)
                
                VStack(alignment: .leading, spacing: 12) {
                    PrivacyItem(
                        icon: "camera.fill",
                        title: "Camera Access",
                        description: "Used for real-time translation of signs and menus"
                    )
                    
                    PrivacyItem(
                        icon: "location.fill",
                        title: "Location",
                        description: "Used to find nearby points of interest and provide contextual recommendations"
                    )
                    
                    PrivacyItem(
                        icon: "text.bubble.fill",
                        title: "Translation History",
                        description: "Stored to provide trip memory features. Can be deleted anytime"
                    )
                }
                .padding(.vertical)
                
                Text("Data Retention")
                    .font(.headline)
                    .padding(.top)
                
                Text("Translation history is retained for 30 days. You can delete your data at any time from Settings.")
                    .font(.body)
                    .foregroundColor(.secondary)
                
                Spacer()
            }
            .padding()
        }
        .navigationTitle("Privacy")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct PrivacyItem: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct AboutView: View {
    var body: some View {
        List {
            Section {
                HStack {
                    Text("Version")
                    Spacer()
                    Text("1.0.0")
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Text("Build")
                    Spacer()
                    Text("2025.11.15")
                        .foregroundColor(.secondary)
                }
            }
            
            Section("Credits") {
                Text("Travel Companion")
                    .font(.headline)
                Text("Making international travel accessible through AI-powered translation and context-aware assistance")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Section("Support") {
                Link("Website", destination: URL(string: "https://travelcompanion.example.com")!)
                Link("Contact Support", destination: URL(string: "mailto:support@travelcompanion.example.com")!)
                Link("Terms of Service", destination: URL(string: "https://travelcompanion.example.com/terms")!)
            }
        }
        .navigationTitle("About")
        .navigationBarTitleDisplayMode(.inline)
    }
}
