import SwiftUI

struct PhraseListView: View {
    @StateObject private var viewModel = PhrasebookViewModel()
    @State private var selectedTab = 0
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Context selector
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(viewModel.availableContexts, id: \.self) { context in
                            ContextButton(
                                context: context,
                                isSelected: viewModel.selectedContext == context
                            ) {
                                viewModel.changeContext(context)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 12)
                }
                .background(Color(.systemGray6))
                
                // Search bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    TextField("Search phrases...", text: $viewModel.searchQuery)
                        .textFieldStyle(.plain)
                    if !viewModel.searchQuery.isEmpty {
                        Button {
                            viewModel.searchQuery = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                
                // Tab selector
                Picker("View", selection: $selectedTab) {
                    Text("Suggestions").tag(0)
                    Text("Favorites (\(viewModel.favoritePhrases.count))").tag(1)
                }
                .pickerStyle(.segmented)
                .padding()
                
                // Content
                if viewModel.isLoading {
                    Spacer()
                    ProgressView("Loading phrases...")
                    Spacer()
                } else if selectedTab == 0 {
                    suggestionsList
                } else {
                    favoritesList
                }
                
                // Error banner
                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(.red.opacity(0.8))
                }
            }
            .navigationTitle("Phrasebook")
            .navigationBarTitleDisplayMode(.large)
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
                        HStack {
                            Text(languageName)
                            Image(systemName: "globe")
                        }
                    }
                }
            }
            .task {
                await viewModel.fetchSuggestions()
                await viewModel.fetchFavorites()
            }
        }
    }
    
    private var suggestionsList: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                if viewModel.filteredSuggestedPhrases.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "text.bubble")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No phrases found")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Try a different context or search term")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 60)
                } else {
                    ForEach(viewModel.filteredSuggestedPhrases) { phrase in
                        PhraseRow(
                            phrase: phrase,
                            targetLanguage: viewModel.selectedLanguage,
                            onCopy: { viewModel.copyPhrase(phrase) },
                            onToggleFavorite: {
                                Task { await viewModel.toggleFavorite(phrase) }
                            }
                        )
                        Divider()
                    }
                }
            }
        }
    }
    
    private var favoritesList: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                if viewModel.filteredFavoritePhrases.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "star")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No favorites yet")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Tap the star icon to save phrases")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 60)
                } else {
                    ForEach(viewModel.filteredFavoritePhrases) { phrase in
                        PhraseRow(
                            phrase: phrase,
                            targetLanguage: viewModel.selectedLanguage,
                            onCopy: { viewModel.copyPhrase(phrase) },
                            onToggleFavorite: {
                                Task { await viewModel.toggleFavorite(phrase) }
                            }
                        )
                        Divider()
                    }
                }
            }
        }
    }
    
    private var languageName: String {
        viewModel.availableLanguages.first { $0.0 == viewModel.selectedLanguage }?.1 ?? "Language"
    }
}

struct ContextButton: View {
    let context: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: iconForContext)
                    .font(.caption)
                Text(context.capitalized)
                    .font(.subheadline)
                    .fontWeight(isSelected ? .semibold : .regular)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(isSelected ? Color.blue : Color(.systemGray5))
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(20)
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

struct PhraseRow: View {
    let phrase: Phrase
    let targetLanguage: String
    let onCopy: () -> Void
    let onToggleFavorite: () -> Void
    
    @State private var showCopiedFeedback = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 8) {
                    // Original text
                    Text(phrase.originalText)
                        .font(.body)
                        .foregroundColor(.primary)
                    
                    // Translation
                    HStack {
                        Text(phrase.translation(for: targetLanguage))
                            .font(.title3)
                            .fontWeight(.semibold)
                            .foregroundColor(.blue)
                        
                        if showCopiedFeedback {
                            HStack(spacing: 4) {
                                Image(systemName: "checkmark")
                                Text("Copied")
                            }
                            .font(.caption)
                            .foregroundColor(.green)
                            .transition(.opacity)
                        }
                    }
                }
                
                Spacer()
                
                // Actions
                VStack(spacing: 12) {
                    Button(action: {
                        onCopy()
                        withAnimation {
                            showCopiedFeedback = true
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                            withAnimation {
                                showCopiedFeedback = false
                            }
                        }
                    }) {
                        Image(systemName: "doc.on.doc")
                            .font(.title3)
                            .foregroundColor(.blue)
                    }
                    
                    Button(action: onToggleFavorite) {
                        Image(systemName: phrase.isFavorite ? "star.fill" : "star")
                            .font(.title3)
                            .foregroundColor(phrase.isFavorite ? .yellow : .gray)
                    }
                }
            }
            
            // Context badge
            HStack(spacing: 4) {
                Image(systemName: phrase.contextIcon)
                    .font(.caption2)
                Text(phrase.context.capitalized)
                    .font(.caption)
            }
            .foregroundColor(.secondary)
        }
        .padding()
    }
}
