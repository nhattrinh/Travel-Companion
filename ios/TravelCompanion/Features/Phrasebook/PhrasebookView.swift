import SwiftUI

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
                await viewModel.loadSuggestions()
                await viewModel.loadFavorites()
            }
        }
    }
    
    // MARK: - Suggestion View
    private var suggestionView: some View {
        VStack(spacing: 0) {
            // Context picker
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(viewModel.contextOptions, id: \.self) { context in
                        Button(action: {
                            viewModel.context = context
                            Task {
                                await viewModel.loadSuggestions()
                            }
                        }) {
                            Text(context.capitalized)
                                .font(.subheadline)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(
                                    viewModel.context == context
                                        ? Color.blue
                                        : Color.gray.opacity(0.2)
                                )
                                .foregroundColor(
                                    viewModel.context == context
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
                            await viewModel.loadSuggestions()
                        }
                    }
                    .buttonStyle(.bordered)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.suggestions.isEmpty {
                Text("No phrases available")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(viewModel.suggestions) { phrase in
                        PhraseRowView(
                            phrase: phrase,
                            isFavorite: viewModel.favorites.contains { $0.id == phrase.id },
                            onToggleFavorite: {
                                Task {
                                    await viewModel.toggleFavorite(phraseId: phrase.id)
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
            if viewModel.favorites.isEmpty {
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
                    ForEach(viewModel.favorites) { phrase in
                        FavoritePhraseRowView(
                            phrase: phrase,
                            onRemove: {
                                Task {
                                    await viewModel.toggleFavorite(phraseId: phrase.id)
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
    let phrase: PhraseSuggestion
    let isFavorite: Bool
    let onToggleFavorite: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(phrase.canonical_text)
                    .font(.body)
                Text(phrase.translation)
                    .font(.headline)
                if let phonetic = phrase.phonetic {
                    Text(phonetic)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
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
    let phrase: PhraseDetail
    let onRemove: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(phrase.canonical_text)
                    .font(.body)
                if let translation = phrase.translations.values.first {
                    Text(translation)
                        .font(.headline)
                }
                if let phonetic = phrase.phonetic {
                    Text(phonetic)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
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
