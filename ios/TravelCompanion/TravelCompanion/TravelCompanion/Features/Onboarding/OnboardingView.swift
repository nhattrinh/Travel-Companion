import SwiftUI

struct OnboardingView: View {
    @Binding var showOnboarding: Bool
    @State private var currentPage = 0
    
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                colors: [.blue.opacity(0.3), .purple.opacity(0.3)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack {
                // Page indicator
                HStack(spacing: 8) {
                    ForEach(0..<onboardingPages.count, id: \.self) { index in
                        Circle()
                            .fill(currentPage == index ? Color.blue : Color.gray.opacity(0.3))
                            .frame(width: 8, height: 8)
                    }
                }
                .padding(.top, 40)
                
                // Paged content
                TabView(selection: $currentPage) {
                    ForEach(Array(onboardingPages.enumerated()), id: \.offset) { index, page in
                        OnboardingPageView(page: page)
                            .tag(index)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .animation(.easeInOut, value: currentPage)
                
                // Action buttons
                HStack(spacing: 16) {
                    if currentPage > 0 {
                        Button {
                            withAnimation {
                                currentPage -= 1
                            }
                        } label: {
                            Text("Back")
                                .foregroundColor(.blue)
                                .padding()
                        }
                    }
                    
                    Spacer()
                    
                    Button {
                        if currentPage < onboardingPages.count - 1 {
                            withAnimation {
                                currentPage += 1
                            }
                        } else {
                            completeOnboarding()
                        }
                    } label: {
                        Text(currentPage == onboardingPages.count - 1 ? "Get Started" : "Next")
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 40)
            }
        }
    }
    
    private func completeOnboarding() {
        UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
        withAnimation {
            showOnboarding = false
        }
    }
}

struct OnboardingPageView: View {
    let page: OnboardingPage
    
    var body: some View {
        VStack(spacing: 30) {
            Spacer()
            
            // Icon
            Image(systemName: page.icon)
                .font(.system(size: 100))
                .foregroundColor(.blue)
                .padding(.bottom, 20)
            
            // Title
            Text(page.title)
                .font(.system(size: 32, weight: .bold))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            // Description
            Text(page.description)
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal, 40)
            
            Spacer()
        }
        .padding()
    }
}

struct OnboardingPage {
    let icon: String
    let title: String
    let description: String
}

let onboardingPages = [
    OnboardingPage(
        icon: "camera.fill",
        title: "Instant Translation",
        description: "Point your camera at signs, menus, or text to get real-time translations with visual overlays"
    ),
    OnboardingPage(
        icon: "map.fill",
        title: "Smart Navigation",
        description: "Discover nearby places with context-aware etiquette notes and cultural tips"
    ),
    OnboardingPage(
        icon: "text.bubble.fill",
        title: "Context Phrasebook",
        description: "Get relevant phrase suggestions based on your location and situation"
    ),
    OnboardingPage(
        icon: "airplane.circle.fill",
        title: "Trip Memory",
        description: "Track your journey with automatic translation history and favorite places"
    ),
    OnboardingPage(
        icon: "hand.raised.fill",
        title: "Privacy First",
        description: "Your data is stored securely and automatically deleted after 30 days"
    )
]
