import SwiftUI

struct WelcomeView: View {
    @Binding var showWelcome: Bool
    
    var body: some View {
        ZStack {
            // Background
            Color(.systemBackground)
                .ignoresSafeArea()
            
            VStack {
                Spacer()
                
                // Logo centered
                LogoView()
                    .frame(width: 300, height: 80)
                
                Spacer()
                
                // Get Started button at bottom
                Button(action: {
                    withAnimation {
                        showWelcome = false
                    }
                }) {
                    Text("Get Started")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 50)
            }
        }
    }
}

// MARK: - Logo View (SVG recreation)
struct LogoView: View {
    var body: some View {
        HStack(spacing: 12) {
            // Speech bubbles icon
            ZStack {
                // White speech bubble (back)
                SpeechBubble(pointingLeft: true)
                    .fill(Color.white)
                    .overlay(
                        SpeechBubble(pointingLeft: true)
                            .stroke(Color.black, lineWidth: 3)
                    )
                    .frame(width: 50, height: 45)
                    .offset(x: -15, y: 0)
                
                // Question mark in white bubble
                VStack(spacing: 2) {
                    Circle()
                        .fill(Color.black)
                        .frame(width: 8, height: 8)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.black)
                        .frame(width: 3, height: 12)
                }
                .offset(x: -15, y: -5)
                
                // Blue speech bubble (front)
                SpeechBubble(pointingLeft: false)
                    .fill(Color(red: 0, green: 122/255, blue: 1)) // #007AFF
                    .frame(width: 50, height: 45)
                    .offset(x: 15, y: 0)
                
                // Sound waves in red bubble
                HStack(spacing: 3) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.white)
                        .frame(width: 3, height: 18)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.white)
                        .frame(width: 3, height: 24)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.white)
                        .frame(width: 3, height: 14)
                }
                .offset(x: 15, y: -5)
            }
            .frame(width: 80, height: 60)
            
            // Text
            Text("Travel Companion")
                .font(.system(size: 28, weight: .bold, design: .default))
                .foregroundColor(.primary)
        }
    }
}

// MARK: - Speech Bubble Shape
struct SpeechBubble: Shape {
    let pointingLeft: Bool
    
    func path(in rect: CGRect) -> Path {
        var path = Path()
        
        let cornerRadius: CGFloat = 10
        let tailSize: CGFloat = 12
        
        // Main bubble body
        let bubbleRect = CGRect(
            x: 0,
            y: 0,
            width: rect.width,
            height: rect.height - tailSize
        )
        
        path.addRoundedRect(in: bubbleRect, cornerSize: CGSize(width: cornerRadius, height: cornerRadius))
        
        // Tail
        if pointingLeft {
            path.move(to: CGPoint(x: rect.width * 0.25, y: rect.height - tailSize))
            path.addLine(to: CGPoint(x: rect.width * 0.15, y: rect.height))
            path.addLine(to: CGPoint(x: rect.width * 0.35, y: rect.height - tailSize))
        } else {
            path.move(to: CGPoint(x: rect.width * 0.65, y: rect.height - tailSize))
            path.addLine(to: CGPoint(x: rect.width * 0.85, y: rect.height))
            path.addLine(to: CGPoint(x: rect.width * 0.75, y: rect.height - tailSize))
        }
        path.closeSubpath()
        
        return path
    }
}

#Preview {
    WelcomeView(showWelcome: .constant(true))
}
