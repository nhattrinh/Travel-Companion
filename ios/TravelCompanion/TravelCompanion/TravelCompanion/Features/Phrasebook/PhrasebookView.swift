import SwiftUI
import Combine

struct PhrasebookView: View {
    @StateObject private var viewModel = PhrasebookViewModel()
    @State private var selectedTab = 0
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Tab selector
                Picker("Mode", selection: $selectedTab) {
                    Text("Suggestions").tag(0)
                    Text("Favorites").tag(1)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                
                if selectedTab == 0 {
                    suggestionView
                } else {
                    favoritesView
                }
            }
            .navigationTitle("Phrasebook")
            .task {
                await viewModel.fetchSuggestions()
                await viewModel.fetchFavorites()
            }
        }
    }
    
    // MARK: - Suggestion View
    private var suggestionView: some View {
        VStack(spacing: 0) {
            // Context picker
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(viewModel.availableContexts, id: \.self) { context in
                        Button(action: {
                            viewModel.selectedContext = context
                            Task {
                                await viewModel.fetchSuggestions()
                            }
                        }) {
                            Text(context.capitalized)
                                .font(.subheadline)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(
                                    viewModel.selectedContext == context
                                        ? Color.blue
                                        : Color.gray.opacity(0.2)
                                )
                                .foregroundColor(
                                    viewModel.selectedContext == context
                                        ? .white
                                        : .primary
                                )
                                .cornerRadius(20)
                        }
                    }
                }
                .padding(.horizontal)
            }
            .padding(.vertical, 8)
            
            Divider()
            
            // Phrases list
            if viewModel.isLoading {
                ProgressView("Loading phrases...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let error = viewModel.errorMessage {
                VStack(spacing: 16) {
                    Text("Error")
                        .font(.headline)
                    Text(error)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Button("Retry") {
                        Task {
                            await viewModel.fetchSuggestions()
                        }
                    }
                    .buttonStyle(.bordered)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.suggestedPhrases.isEmpty {
                Text("No phrases available")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(viewModel.suggestedPhrases) { phrase in
                        PhraseRowView(
                            phrase: phrase,
                            targetLanguage: viewModel.selectedLanguage,
                            isFavorite: viewModel.favoritePhrases.contains { $0.id == phrase.id },
                            onToggleFavorite: {
                                Task {
                                    await viewModel.toggleFavorite(phrase)
                                }
                            }
                        )
                    }
                }
                .listStyle(PlainListStyle())
            }
        }
    }
    
    // MARK: - Favorites View
    private var favoritesView: some View {
        Group {
            if viewModel.favoritePhrases.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "star.slash")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("No Favorites")
                        .font(.headline)
                    Text("Tap the star icon on phrases to save them here")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(viewModel.favoritePhrases) { phrase in
                        FavoritePhraseRowView(
                            phrase: phrase,
                            targetLanguage: viewModel.selectedLanguage,
                            onRemove: {
                                Task {
                                    await viewModel.toggleFavorite(phrase)
                                }
                            }
                        )
                    }
                }
                .listStyle(PlainListStyle())
            }
        }
    }
}

// MARK: - Phrase Row
struct PhraseRowView: View {
    let phrase: Phrase
    let targetLanguage: String
    let isFavorite: Bool
    let onToggleFavorite: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(phrase.originalText)
                    .font(.body)
                Text(phrase.translation(for: targetLanguage))
                    .font(.headline)
            }
            
            Spacer()
            
            Button(action: onToggleFavorite) {
                Image(systemName: isFavorite ? "star.fill" : "star")
                    .foregroundColor(isFavorite ? .yellow : .gray)
                    .font(.system(size: 20))
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding(.vertical, 8)
    }
}

// MARK: - Favorite Phrase Row
struct FavoritePhraseRowView: View {
    let phrase: Phrase
    let targetLanguage: String
    let onRemove: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(phrase.originalText)
                    .font(.body)
                Text(phrase.translation(for: targetLanguage))
                    .font(.headline)
            }
            
            Spacer()
            
            Button(action: onRemove) {
                Image(systemName: "star.fill")
                    .foregroundColor(.yellow)
                    .font(.system(size: 20))
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding(.vertical, 8)
    }
}

#Preview {
    PhrasebookView()
}
