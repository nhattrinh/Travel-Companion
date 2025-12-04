import SwiftUI
import Combine

// MARK: - Status Banner
enum BannerStyle {
    case success, error, info
    
    var backgroundColor: Color {
        switch self {
        case .success: return Color.green
        case .error: return Color.red
        case .info: return Color.blue
        }
    }
    
    var icon: String {
        switch self {
        case .success: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        case .info: return "info.circle.fill"
        }
    }
}

struct StatusBanner: View {
    let message: String
    let style: BannerStyle
    let onDismiss: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: style.icon)
                .font(.title3)
            
            Text(message)
                .font(.subheadline)
                .multilineTextAlignment(.leading)
            
            Spacer()
            
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.caption)
                    .fontWeight(.bold)
            }
        }
        .foregroundColor(.white)
        .padding()
        .background(style.backgroundColor)
        .cornerRadius(10)
        .shadow(color: style.backgroundColor.opacity(0.4), radius: 8, x: 0, y: 4)
        .padding(.horizontal)
    }
}

// MARK: - Login View
struct LoginView: View {
    @EnvironmentObject var authState: AuthState
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var loading = false
    @State private var user: UserDTO? = nil
    
    // Status banner states
    @State private var showBanner = false
    @State private var bannerMessage = ""
    @State private var bannerStyle: BannerStyle = .info
    @State private var statusMessage = ""

    var body: some View {
        ZStack(alignment: .top) {
            VStack(spacing: 20) {
                Text("Login")
                    .font(.title2)
                    .bold()

                TextField("Email", text: $email)
                    .textContentType(.emailAddress)
                    .keyboardType(.emailAddress)
                    .autocapitalization(.none)
                    .padding()
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.3)))

                SecureField("Password", text: $password)
                    .padding()
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.3)))
                
                // Password requirement hint
                if !password.isEmpty && password.count < 8 {
                    HStack {
                        Image(systemName: "exclamationmark.circle")
                            .foregroundColor(.orange)
                        Text("Password must be at least 8 characters (\(password.count)/8)")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                // Status message (inline)
                if !statusMessage.isEmpty {
                    HStack {
                        Image(systemName: "info.circle")
                            .foregroundColor(.secondary)
                        Text(statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .transition(.opacity)
                }

                // Login button with loading state
                Button(action: submit) {
                    HStack {
                        if loading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        }
                        Text(loading ? "Signing in..." : "Login")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(loginButtonDisabled ? Color.blue.opacity(0.5) : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .disabled(loginButtonDisabled)

                // Register button
                Button(action: register) {
                    HStack {
                        if loading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        }
                        Text(loading ? "Creating account..." : "Register")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(registerButtonDisabled ? Color.green.opacity(0.5) : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .disabled(registerButtonDisabled)

                if let user = user {
                    HStack {
                        Image(systemName: "person.circle.fill")
                            .foregroundColor(.green)
                        Text("Logged in as \(user.email)")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 8)
                }

                Spacer()
            }
            .padding(24)
            .padding(.top, showBanner ? 70 : 0)
            
            // Toast banner overlay
            if showBanner {
                StatusBanner(
                    message: bannerMessage,
                    style: bannerStyle,
                    onDismiss: dismissBanner
                )
                .transition(.move(edge: .top).combined(with: .opacity))
                .zIndex(1)
                .padding(.top, 8)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: showBanner)
        .animation(.easeInOut(duration: 0.2), value: statusMessage)
    }
    
    private var loginButtonDisabled: Bool {
        loading || email.isEmpty || password.count < 8
    }
    
    private var registerButtonDisabled: Bool {
        loading || email.isEmpty || password.count < 8
    }

    private func submit() {
        showStatus("Connecting to server...")
        runTask(action: "login") { 
            try await AuthService.shared.login(email: email, password: password) 
        }
    }

    private func register() {
        showStatus("Creating your account...")
        runTask(action: "register") { 
            try await AuthService.shared.register(email: email, password: password) 
        }
    }
    
    private func showStatus(_ message: String) {
        withAnimation {
            statusMessage = message
        }
    }
    
    private func showBanner(message: String, style: BannerStyle, autoDismiss: Bool = true) {
        bannerMessage = message
        bannerStyle = style
        withAnimation {
            showBanner = true
            statusMessage = ""
        }
        
        if autoDismiss {
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                dismissBanner()
            }
        }
    }
    
    private func dismissBanner() {
        withAnimation {
            showBanner = false
        }
    }

    private func runTask(action: String, _ op: @escaping () async throws -> UserDTO) {
        loading = true
        dismissBanner()
        
        Task {
            do {
                let u = try await op()
                await MainActor.run {
                    self.user = u
                    self.loading = false
                    self.showBanner(
                        message: action == "login" ? "Welcome back, \(u.email)!" : "Account created successfully!",
                        style: .success
                    )
                    
                    // Delay auth state change slightly so user sees success message
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                        self.authState.login(user: u)
                    }
                }
            } catch let authError as AuthService.AuthError {
                await MainActor.run {
                    self.loading = false
                    let errorMessage = formatAuthError(authError, action: action)
                    self.showBanner(message: errorMessage, style: .error, autoDismiss: false)
                }
            } catch let urlError as URLError {
                await MainActor.run {
                    self.loading = false
                    let errorMessage = formatURLError(urlError)
                    self.showBanner(message: errorMessage, style: .error, autoDismiss: false)
                }
            } catch {
                await MainActor.run {
                    self.loading = false
                    self.showBanner(
                        message: "Something went wrong. Please try again.",
                        style: .error,
                        autoDismiss: false
                    )
                }
            }
        }
    }
    
    private func formatAuthError(_ error: AuthService.AuthError, action: String) -> String {
        switch error {
        case .server(let message):
            // Make server messages more user-friendly
            if message.lowercased().contains("invalid") || message.lowercased().contains("credentials") {
                return "Invalid email or password. Please try again."
            } else if message.lowercased().contains("exist") {
                return action == "register" 
                    ? "An account with this email already exists." 
                    : "No account found with this email."
            } else if message.lowercased().contains("password") {
                return "Password doesn't meet requirements."
            }
            return message
        case .decoding:
            return "Unable to process server response. Please try again."
        case .unauthorized:
            return "Session expired. Please log in again."
        }
    }
    
    private func formatURLError(_ error: URLError) -> String {
        switch error.code {
        case .notConnectedToInternet:
            return "No internet connection. Please check your network."
        case .timedOut:
            return "Request timed out. Please try again."
        case .cannotConnectToHost, .cannotFindHost:
            return "Cannot connect to server. Please try again later."
        default:
            return "Network error. Please check your connection."
        }
    }
}

#Preview {
    LoginView()
        .environmentObject(AuthState())
}
