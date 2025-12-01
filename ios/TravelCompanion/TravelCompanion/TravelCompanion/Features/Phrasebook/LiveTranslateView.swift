import SwiftUI
import Speech
import AVFoundation

// MARK: - Translation Card Model
struct TranslationCard: Identifiable, Equatable {
    let id = UUID()
    let originalText: String
    let translatedText: String
    let sourceLanguage: String
    let targetLanguage: String
    let timestamp: Date
    
    static func == (lhs: TranslationCard, rhs: TranslationCard) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - Live Translate View
struct LiveTranslateView: View {
    @StateObject private var viewModel = LiveTranslateViewModel()
    @State private var selectedTargetLanguage = "en"
    
    let availableLanguages = [
        ("en", "English"),
        ("ko", "Korean"),
        ("vi", "Vietnamese"),
        ("ja", "Japanese"),
        ("zh", "Chinese"),
        ("es", "Spanish")
    ]
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background
                Color(.systemGroupedBackground)
                    .ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Main content
                    if viewModel.translations.isEmpty && !viewModel.isListening {
                        emptyStateView
                    } else {
                        translationsList
                        
                        // Bottom controls when translations exist
                        controlsView
                    }
                }
            }
            .navigationTitle("Live Translate")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !viewModel.translations.isEmpty {
                        Button("Clear") {
                            withAnimation {
                                viewModel.clearTranslations()
                            }
                        }
                    }
                }
            }
            .alert("Microphone Permission Required", isPresented: $viewModel.showPermissionAlert) {
                Button("Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Please enable microphone access in Settings to use Live Translate.")
            }
        }
        .onChange(of: selectedTargetLanguage) { newValue in
            viewModel.targetLanguage = newValue
        }
    }
    
    // MARK: - Language Selector
    private var languageSelector: some View {
        HStack {
            Text("Translate to:")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Picker("Language", selection: $selectedTargetLanguage) {
                ForEach(availableLanguages, id: \.0) { code, name in
                    Text(name).tag(code)
                }
            }
            .pickerStyle(.menu)
            .tint(.blue)
        }
    }
    
    // MARK: - Translate Button
    private var translateButton: some View {
        Button(action: {
            if viewModel.isListening {
                viewModel.stopListening()
            } else {
                viewModel.startListening()
            }
        }) {
            HStack(spacing: 12) {
                if viewModel.isListening {
                    PulsatingCircle()
                    Text("Listening...")
                } else {
                    Image(systemName: "mic.fill")
                        .font(.title2)
                    Text("Start Translate")
                }
            }
            .font(.headline)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(viewModel.isListening ? Color.red : Color.blue)
            .cornerRadius(16)
        }
        .padding(.horizontal)
        .disabled(viewModel.isProcessing)
    }
    
    // MARK: - Empty State
    private var emptyStateView: some View {
        VStack(spacing: 24) {
            Spacer()
            
            Image(systemName: "waveform.circle")
                .font(.system(size: 80))
                .foregroundColor(.blue.opacity(0.6))
            
            VStack(spacing: 8) {
                Text("Start Speaking")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Tap the button and speak in any language")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            
            // Language selector moved here
            languageSelector
                .padding(.top, 8)
            
            // Centered translate button
            translateButton
                .padding(.top, 16)
            
            Spacer()
        }
        .padding()
    }
    
    // MARK: - Translations List
    private var translationsList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.translations) { card in
                        TranslationCardView(card: card)
                            .id(card.id)
                            .transition(.asymmetric(
                                insertion: .move(edge: .bottom).combined(with: .opacity),
                                removal: .opacity
                            ))
                    }
                    
                    // Live transcription preview
                    if viewModel.isListening && !viewModel.currentTranscription.isEmpty {
                        LiveTranscriptionView(text: viewModel.currentTranscription)
                            .id("live-transcription")
                    }
                }
                .padding()
            }
            .onChange(of: viewModel.translations.count) { _ in
                if let lastCard = viewModel.translations.last {
                    withAnimation(.easeOut(duration: 0.3)) {
                        proxy.scrollTo(lastCard.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: viewModel.currentTranscription) { _ in
                if viewModel.isListening {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo("live-transcription", anchor: .bottom)
                    }
                }
            }
        }
    }
    
    // MARK: - Controls (for when translations exist)
    private var controlsView: some View {
        VStack(spacing: 16) {
            // Error message
            if let error = viewModel.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal)
            }
            
            // Language selector and button when there are translations
            languageSelector
            
            translateButton
        }
        .padding(.vertical, 16)
        .background(Color(.systemBackground))
    }
}

// MARK: - Translation Card View
struct TranslationCardView: View {
    let card: TranslationCard
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Original text
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(languageName(for: card.sourceLanguage))
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(timeString(from: card.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                Text(card.originalText)
                    .font(.body)
            }
            
            Divider()
            
            // Translated text
            VStack(alignment: .leading, spacing: 4) {
                Text(languageName(for: card.targetLanguage))
                    .font(.caption)
                    .foregroundColor(.blue)
                
                Text(card.translatedText)
                    .font(.body)
                    .fontWeight(.medium)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
    }
    
    private func languageName(for code: String) -> String {
        let names = [
            "en": "English",
            "ko": "Korean",
            "vi": "Vietnamese",
            "ja": "Japanese",
            "zh": "Chinese",
            "es": "Spanish"
        ]
        return names[code] ?? code.uppercased()
    }
    
    private func timeString(from date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Live Transcription View
struct LiveTranscriptionView: View {
    let text: String
    
    var body: some View {
        HStack {
            PulsatingCircle()
                .frame(width: 12, height: 12)
            
            Text(text)
                .font(.body)
                .foregroundColor(.secondary)
                .italic()
            
            Spacer()
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Pulsating Circle
struct PulsatingCircle: View {
    @State private var isAnimating = false
    
    var body: some View {
        Circle()
            .fill(Color.red)
            .frame(width: 12, height: 12)
            .scaleEffect(isAnimating ? 1.3 : 1.0)
            .opacity(isAnimating ? 0.5 : 1.0)
            .animation(
                Animation.easeInOut(duration: 0.8)
                    .repeatForever(autoreverses: true),
                value: isAnimating
            )
            .onAppear {
                isAnimating = true
            }
    }
}

// MARK: - Preview
#Preview {
    LiveTranslateView()
}
