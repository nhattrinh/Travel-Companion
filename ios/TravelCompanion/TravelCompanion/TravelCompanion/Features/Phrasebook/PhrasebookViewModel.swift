import Foundation
import Combine
import UIKit

@MainActor
final class PhrasebookViewModel: ObservableObject {
    @Published var suggestedPhrases: [Phrase] = []
    @Published var favoritePhrases: [Phrase] = []
    @Published var selectedContext: String = "restaurant"
    @Published var selectedLanguage: String = "ja"
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var searchQuery: String = ""
    
    private let apiClient: APIClient
    private let authService: AuthService
    private var cancellables = Set<AnyCancellable>()
    
    let availableContexts = [
        "restaurant",
        "transit",
        "hotel",
        "shopping",
        "emergency",
        "greeting"
    ]
    
    let availableLanguages = [
        ("en", "English"),
        ("ko", "Korean"),
        ("vi", "Vietnamese")
    ]
    
    init(apiClient: APIClient = .shared, authService: AuthService = .shared) {
        self.apiClient = apiClient
        self.authService = authService
        setupSearchDebounce()
    }
    
    private func setupSearchDebounce() {
        $searchQuery
            .debounce(for: .milliseconds(300), scheduler: RunLoop.main)
            .removeDuplicates()
            .sink { [weak self] _ in
                Task {
                    await self?.fetchSuggestions()
                }
            }
            .store(in: &cancellables)
    }
    
    func fetchSuggestions() async {
        isLoading = true
        errorMessage = nil
        
        do {
            let token = try await authService.getAccessToken()
            
            let response = try await apiClient.fetchPhraseSuggestions(
                context: selectedContext,
                targetLanguage: selectedLanguage,
                token: token
            )
            
            guard response.status == "ok", let data = response.data else {
                throw PhrasebookError.apiError(response.error ?? "Unknown error")
            }
            
            suggestedPhrases = data.phrases.map { $0.toPhrase() }
            
        } catch {
            errorMessage = error.localizedDescription
            suggestedPhrases = []
        }
        
        isLoading = false
    }
    
    func fetchFavorites() async {
        do {
            let token = try await authService.getAccessToken()
            
            let response = try await apiClient.fetchFavorites(
                itemType: "phrase",
                token: token
            )
            
            guard response.status == "ok", let data = response.data else {
                throw PhrasebookError.apiError(response.error ?? "Unknown error")
            }
            
            favoritePhrases = data.favorites.compactMap { $0.toPhrase() }
            
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func toggleFavorite(_ phrase: Phrase) async {
        if phrase.isFavorite {
            await removeFavorite(phrase)
        } else {
            await addFavorite(phrase)
        }
    }
    
    private func addFavorite(_ phrase: Phrase) async {
        do {
            let token = try await authService.getAccessToken()
            
            let request = FavoriteRequest(itemType: "phrase", itemId: 1)
            
            let response = try await apiClient.addFavorite(
                request: request,
                token: token
            )
            
            guard response.status == "ok" else {
                throw PhrasebookError.apiError(response.error ?? "Failed to add favorite")
            }
            
            if let index = suggestedPhrases.firstIndex(where: { $0.id == phrase.id }) {
                suggestedPhrases[index] = Phrase(
                    id: phrase.id,
                    context: phrase.context,
                    originalText: phrase.originalText,
                    translations: phrase.translations,
                    isFavorite: true
                )
            }
            
            await fetchFavorites()
            
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    private func removeFavorite(_ phrase: Phrase) async {
        do {
            let token = try await authService.getAccessToken()
            
            let favoriteId = 1
            
            let response = try await apiClient.removeFavorite(
                favoriteId: favoriteId,
                token: token
            )
            
            guard response.status == "ok" else {
                throw PhrasebookError.apiError(response.error ?? "Failed to remove favorite")
            }
            
            if let index = suggestedPhrases.firstIndex(where: { $0.id == phrase.id }) {
                suggestedPhrases[index] = Phrase(
                    id: phrase.id,
                    context: phrase.context,
                    originalText: phrase.originalText,
                    translations: phrase.translations,
                    isFavorite: false
                )
            }
            
            favoritePhrases.removeAll { $0.id == phrase.id }
            
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func changeContext(_ context: String) {
        selectedContext = context
        Task {
            await fetchSuggestions()
        }
    }
    
    func copyPhrase(_ phrase: Phrase) {
        let translation = phrase.translation(for: selectedLanguage)
        UIPasteboard.general.string = translation
    }
    
    var filteredSuggestedPhrases: [Phrase] {
        if searchQuery.isEmpty {
            return suggestedPhrases
        }
        return suggestedPhrases.filter {
            $0.originalText.localizedCaseInsensitiveContains(searchQuery) ||
            $0.translation(for: selectedLanguage).localizedCaseInsensitiveContains(searchQuery)
        }
    }
    
    var filteredFavoritePhrases: [Phrase] {
        if searchQuery.isEmpty {
            return favoritePhrases
        }
        return favoritePhrases.filter {
            $0.originalText.localizedCaseInsensitiveContains(searchQuery) ||
            $0.translation(for: selectedLanguage).localizedCaseInsensitiveContains(searchQuery)
        }
    }
}

enum PhrasebookError: LocalizedError {
    case unauthorized
    case apiError(String)
    
    var errorDescription: String? {
        switch self {
        case .unauthorized:
            return "Authentication required"
        case .apiError(let message):
            return message
        }
    }
}
