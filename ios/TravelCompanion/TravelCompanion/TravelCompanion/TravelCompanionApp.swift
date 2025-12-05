//
//  TravelCompanionApp.swift
//  TravelCompanion
//
//  Created by Nhat Trinh on 11/28/25.
//

import SwiftUI
import Combine

@main
struct TravelCompanionApp: App {
    @StateObject private var authState = AuthState()
    @State private var showOnboarding = !UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
    @State private var showWelcome = true
    
    var body: some Scene {
        WindowGroup {
            Group {
                if showOnboarding {
                    OnboardingView(showOnboarding: $showOnboarding)
                } else if showWelcome && !authState.isAuthenticated {
                    WelcomeView(showWelcome: $showWelcome)
                } else if authState.isAuthenticated {
                    MainTabView()
                        .environmentObject(authState)
                } else {
                    LoginView()
                        .environmentObject(authState)
                }
            }
        }
    }
}

final class AuthState: ObservableObject {
    @Published var isAuthenticated: Bool = KeychainTokenStore.shared.accessToken() != nil
    @Published var currentUser: UserDTO?
    
    private let userDefaultsKey = "currentUser"
    
    init() {
        // Load cached user from UserDefaults
        if let data = UserDefaults.standard.data(forKey: userDefaultsKey),
           let user = try? JSONDecoder().decode(UserDTO.self, from: data) {
            self.currentUser = user
        } else if isAuthenticated {
            // Token exists but no cached user - fetch from backend
            Task { @MainActor in
                await self.fetchUserProfile()
            }
        }
    }
    
    /// Fetch user profile from backend and cache it
    @MainActor
    func fetchUserProfile() async {
        do {
            let user = try await AuthService.shared.fetchCurrentUser()
            self.currentUser = user
            // Cache user in UserDefaults
            if let data = try? JSONEncoder().encode(user) {
                UserDefaults.standard.set(data, forKey: userDefaultsKey)
            }
        } catch {
            print("Failed to fetch user profile: \(error)")
            // If unauthorized, clear auth state
            if case AuthService.AuthError.unauthorized = error {
                self.logout()
            }
        }
    }
    
    func login(user: UserDTO? = nil) {
        if let user = user {
            self.currentUser = user
            // Cache user in UserDefaults
            if let data = try? JSONEncoder().encode(user) {
                UserDefaults.standard.set(data, forKey: userDefaultsKey)
            }
        }
        isAuthenticated = true
    }
    
    func logout() {
        KeychainTokenStore.shared.clear()
        UserDefaults.standard.removeObject(forKey: userDefaultsKey)
        currentUser = nil
        isAuthenticated = false
    }
}
