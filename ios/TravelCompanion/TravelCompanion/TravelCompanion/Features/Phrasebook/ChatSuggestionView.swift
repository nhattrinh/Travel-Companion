import SwiftUI
import UIKit
import Combine

/// Context-aware chat-style phrase suggestions
struct ChatSuggestionView: View {
    @StateObject private var viewModel = PhrasebookViewModel()
    @State private var selectedPhrase: Phrase?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Context header
                contextHeader
                
                Divider()
                
                // Phrase suggestions as chat bubbles
                ScrollView {
                    LazyVStack(spacing: 16) {
                        if viewModel.isLoading {
                            ProgressView()
                                .padding()
                        } else if viewModel.suggestedPhrases.isEmpty {
                            emptyState
                        } else {
                            ForEach(viewModel.suggestedPhrases) { phrase in
                                ChatBubble(
                                    phrase: phrase,
                                    targetLanguage: viewModel.selectedLanguage,
                                    onTap: { selectedPhrase = phrase }
                                )
                            }
                        }
                    }
                    .padding()
                }
                
                // Quick context switcher
                quickContextBar
            }
            .navigationTitle("Quick Phrases")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        ForEach(viewModel.availableLanguages, id: \.0) { code, name in
                            Button {
                                viewModel.selectedLanguage = code
                                Task { await viewModel.fetchSuggestions() }
                            } label: {
                                HStack {
                                    Text(name)
                                    if viewModel.selectedLanguage == code {
                                        Image(systemName: "checkmark")
                                    }
                                }
                            }
                        }
                    } label: {
                        Image(systemName: "globe")
                    }
                }
            }
            .sheet(item: $selectedPhrase) { phrase in
                PhraseDetailSheet(
                    phrase: phrase,
                    targetLanguage: viewModel.selectedLanguage,
                    viewModel: viewModel
                )
            }
            .task {
                await viewModel.fetchSuggestions()
            }
        }
    }
    
    private var contextHeader: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: iconForContext(viewModel.selectedContext))
                    .font(.title2)
                    .foregroundColor(.blue)
                Text(viewModel.selectedContext.capitalized)
                    .font(.headline)
            }
            
            Text("Tap a phrase to see details and copy")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color(.systemGray6))
    }
    
    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "text.bubble")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            Text("No suggestions for this context")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("Try selecting a different context below")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.top, 60)
    }
    
    private var quickContextBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(viewModel.availableContexts, id: \.self) { context in
                    QuickContextButton(
                        context: context,
                        isSelected: viewModel.selectedContext == context
                    ) {
                        viewModel.changeContext(context)
                    }
                }
            }
            .padding()
        }
        .background(.ultraThinMaterial)
        .shadow(color: .black.opacity(0.1), radius: 5, y: -2)
    }
    
    private func iconForContext(_ context: String) -> String {
        switch context.lowercased() {
        case "restaurant": return "fork.knife"
        case "transit": return "tram.fill"
        case "hotel": return "bed.double.fill"
        case "shopping": return "cart.fill"
        case "emergency": return "exclamationmark.triangle.fill"
        case "greeting": return "hand.wave.fill"
        default: return "text.bubble.fill"
        }
    }
}

struct ChatBubble: View {
    let phrase: Phrase
    let targetLanguage: String
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                // Translation (prominent)
                Text(phrase.translation(for: targetLanguage))
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                // Original text (smaller)
                Text(phrase.originalText)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                // Context badge
                HStack(spacing: 4) {
                    Image(systemName: phrase.contextIcon)
                        .font(.caption2)
                    Text(phrase.context.capitalized)
                        .font(.caption2)
                }
                .foregroundColor(.blue)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color.blue.opacity(0.1))
            .cornerRadius(16)
        }
        .buttonStyle(.plain)
    }
}

struct QuickContextButton: View {
    let context: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: iconForContext)
                    .font(.title2)
                Text(context.capitalized)
                    .font(.caption)
            }
            .foregroundColor(isSelected ? .white : .primary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(isSelected ? Color.blue : Color(.systemGray6))
            .cornerRadius(12)
        }
    }
    
    private var iconForContext: String {
        switch context.lowercased() {
        case "restaurant": return "fork.knife"
        case "transit": return "tram.fill"
        case "hotel": return "bed.double.fill"
        case "shopping": return "cart.fill"
        case "emergency": return "exclamationmark.triangle.fill"
        case "greeting": return "hand.wave.fill"
        default: return "text.bubble.fill"
        }
    }
}

struct PhraseDetailSheet: View {
    let phrase: Phrase
    let targetLanguage: String
    @ObservedObject var viewModel: PhrasebookViewModel
    @Environment(\.dismiss) var dismiss
    @State private var showCopiedFeedback = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Translation (large)
                VStack(spacing: 12) {
                    Text(phrase.translation(for: targetLanguage))
                        .font(.system(size: 36, weight: .bold))
                        .multilineTextAlignment(.center)
                        .foregroundColor(.blue)
                    
                    Text(phrase.originalText)
                        .font(.title3)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding()
                
                Divider()
                
                // Context info
                HStack {
                    Image(systemName: phrase.contextIcon)
                        .font(.title2)
                        .foregroundColor(.blue)
                    Text("For \(phrase.context.capitalized) situations")
                        .font(.headline)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color(.systemGray6))
                .cornerRadius(12)
                .padding(.horizontal)
                
                Spacer()
                
                // Actions
                VStack(spacing: 12) {
                    Button {
                        viewModel.copyPhrase(phrase)
                        withAnimation {
                            showCopiedFeedback = true
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                            withAnimation {
                                showCopiedFeedback = false
                            }
                            dismiss()
                        }
                    } label: {
                        HStack {
                            Image(systemName: showCopiedFeedback ? "checkmark" : "doc.on.doc")
                            Text(showCopiedFeedback ? "Copied!" : "Copy to Clipboard")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(showCopiedFeedback)
                    
                    Button {
                        Task {
                            await viewModel.toggleFavorite(phrase)
                        }
                    } label: {
                        HStack {
                            Image(systemName: phrase.isFavorite ? "star.fill" : "star")
                            Text(phrase.isFavorite ? "Remove from Favorites" : "Add to Favorites")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                    }
                    .buttonStyle(.bordered)
                    
                    Button {
                        sharePhrase()
                    } label: {
                        HStack {
                            Image(systemName: "square.and.arrow.up")
                            Text("Share")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
            }
            .navigationTitle("Phrase Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }
    
    private func sharePhrase() {
        let text = "\(phrase.translation(for: targetLanguage))\n(\(phrase.originalText))"
        let activityVC = UIActivityViewController(activityItems: [text], applicationActivities: nil)
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootVC = windowScene.windows.first?.rootViewController {
            rootVC.present(activityVC, animated: true)
        }
    }
}
