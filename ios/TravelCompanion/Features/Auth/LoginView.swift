import SwiftUI

struct LoginView: View {
    @EnvironmentObject var authState: AuthState
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var loading = false
    @State private var error: String? = nil
    @State private var user: UserDTO? = nil

    var body: some View {
        VStack(spacing: 20) {
            Text("Travel Companion Login")
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

            if let error = error {
                Text(error).foregroundColor(.red).font(.caption)
            }

            if loading { ProgressView() }

            Button(action: submit) {
                Text("Login")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }.disabled(loading || email.isEmpty || password.count < 8)

            Button(action: register) {
                Text("Register")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }.disabled(loading || email.isEmpty || password.count < 8)

            if let user = user {
                Text("Logged in as \(user.email)").font(.footnote)
            }

            Spacer()
        }
        .padding(24)
    }

    private func submit() {
        runTask { try await AuthService.shared.login(email: email, password: password) }
    }

    private func register() {
        runTask { try await AuthService.shared.register(email: email, password: password) }
    }

    private func runTask(_ op: @escaping () async throws -> UserDTO) {
        loading = true; error = nil
        Task {
            do {
                let u = try await op()
                await MainActor.run {
                    self.user = u
                    self.authState.login()  // Trigger authentication state
                }
            } catch {
                await MainActor.run { self.error = (error as? AuthService.AuthError).map(String.init(describing:)) ?? "Unexpected error" }
            }
            await MainActor.run { loading = false }
        }
    }
}

#Preview {
    LoginView()
}
