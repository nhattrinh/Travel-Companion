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
    
    var body: some Scene {
        WindowGroup {
            Group {
                if showOnboarding {
                    OnboardingView(showOnboarding: $showOnboarding)
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
    
    func login() {
        isAuthenticated = true
    }
    
    func logout() {
        KeychainTokenStore.shared.clear()
        isAuthenticated = false
    }
}
