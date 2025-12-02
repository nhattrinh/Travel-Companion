import SwiftUI
import Combine
import PhotosUI
import UIKit

/// View for translating menu images from photo library
struct StaticCaptureView: View {
    @StateObject private var viewModel = TranslationViewModel()
    @State private var selectedImage: UIImage?
    @State private var showingImagePicker = false
    @Environment(\.dismiss) private var dismiss
    
    // Optional initial image passed from MenuView
    var initialImage: UIImage?
    var onDismiss: (() -> Void)?
    
    init(image: UIImage? = nil, onDismiss: (() -> Void)? = nil) {
        self.initialImage = image
        self.onDismiss = onDismiss
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color(.systemBackground)
                    .ignoresSafeArea()
                
                if let image = selectedImage {
                    // Main content with menu image and items
                    menuTranslationContent(image: image)
                } else {
                    // Empty state
                    emptyStateView
                }
                
                // Processing overlay
                if viewModel.isProcessing {
                    processingOverlay
                }
            }
            .navigationTitle("Menu Translation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        handleDismiss()
                    } label: {
                        Image(systemName: "xmark")
                    }
                }
                
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showingImagePicker = true
                    } label: {
                        Image(systemName: "photo.badge.plus")
                    }
                    .disabled(viewModel.isProcessing)
                }
            }
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(selectedImage: $selectedImage) {
                    viewModel.clearSegments()
                }
            }
            .onAppear {
                if let image = initialImage, selectedImage == nil {
                    selectedImage = image
                    // Auto-translate on appear - only if we have an initial image
                    Task { @MainActor in
                        try? await Task.sleep(nanoseconds: 100_000_000) // Small delay to let UI settle
                        await viewModel.translateFrame(image)
                    }
                }
            }
            .onChange(of: selectedImage) { oldImage, newImage in
                // Only trigger if this is a NEW image selection (not the initial load)
                if let image = newImage, oldImage != nil {
                    Task {
                        await viewModel.translateFrame(image)
                    }
                }
            }
        }
    }
    
    // MARK: - Menu Translation Content
    @ViewBuilder
    private func menuTranslationContent(image: UIImage) -> some View {
        ScrollView {
            VStack(spacing: 0) {
                // Menu image at top
                menuImageSection(image: image)
                
                // Divider
                Rectangle()
                    .fill(Color(.separator))
                    .frame(height: 1)
                
                // Menu items list - show only food items
                let foodItems = viewModel.translationSegments.filter { $0.itemType == "food" }
                if !foodItems.isEmpty {
                    menuItemsList
                } else if !viewModel.isProcessing {
                    noItemsFound
                }
                
                // Error message
                if let error = viewModel.errorMessage {
                    errorBanner(message: error)
                }
            }
        }
    }
    
    // MARK: - Menu Image Section
    @ViewBuilder
    private func menuImageSection(image: UIImage) -> some View {
        let foodItemCount = viewModel.translationSegments.filter { $0.itemType == "food" }.count
        
        VStack(spacing: 12) {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 300)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
            
            // Info bar
            if let lang = viewModel.detectedLanguage {
                HStack {
                    Label("Detected: \(languageName(for: lang))", systemImage: "globe")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("\(foodItemCount) items found")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 4)
            }
        }
        .padding()
    }
    
    // MARK: - Menu Items List
    @ViewBuilder
    private var menuItemsList: some View {
        LazyVStack(spacing: 16) {
            // Filter to only show food items, not prices
            ForEach(viewModel.translationSegments.filter { $0.itemType == "food" }) { segment in
                MenuItemCard(segment: segment)
            }
        }
        .padding()
    }
    
    // MARK: - No Items Found
    @ViewBuilder
    private var noItemsFound: some View {
        VStack(spacing: 12) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            
            Text("No menu items detected")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Try uploading a clearer image of the menu")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(40)
    }
    
    // MARK: - Empty State
    @ViewBuilder
    private var emptyStateView: some View {
        VStack(spacing: 24) {
            Image(systemName: "menucard")
                .font(.system(size: 80))
                .foregroundColor(.secondary)
            
            Text("Upload a Menu Photo")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Take or select a photo of a menu to see translated items with pictures")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            
            Button {
                showingImagePicker = true
            } label: {
                Label("Choose Photo", systemImage: "photo.fill")
                    .frame(maxWidth: 200)
                    .padding()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
    
    // MARK: - Processing Overlay
    @ViewBuilder
    private var processingOverlay: some View {
        ZStack {
            Color.black.opacity(0.3)
                .ignoresSafeArea()
            
            VStack(spacing: 16) {
                ProgressView()
                    .scaleEffect(1.5)
                    .tint(.white)
                
                Text("Translating menu...")
                    .font(.headline)
                    .foregroundColor(.white)
                
                Text("Finding items and translations")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            .padding(30)
            .background(.ultraThinMaterial)
            .cornerRadius(16)
        }
    }
    
    // MARK: - Error Banner
    @ViewBuilder
    private func errorBanner(message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
            
            Text(message)
                .font(.caption)
                .foregroundColor(.primary)
            
            Spacer()
            
            Button {
                viewModel.errorMessage = nil
            } label: {
                Image(systemName: "xmark")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(12)
        .padding()
    }
    
    // MARK: - Helper Methods
    private func handleDismiss() {
        if let onDismiss = onDismiss {
            onDismiss()
        } else {
            dismiss()
        }
    }
    
    private func languageName(for code: String) -> String {
        switch code.lowercased() {
        case "ko", "korean": return "Korean"
        case "vi", "vn", "vietnamese": return "Vietnamese"
        case "en", "english": return "English"
        default: return code.uppercased()
        }
    }
}

// MARK: - Menu Item Card
struct MenuItemCard: View {
    let segment: TranslationSegment
    
    @State private var imageUrls: [URL] = []
    @State private var isLoadingImages = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Image carousel for food item
            foodImageSection
            
            // Text content
            VStack(alignment: .leading, spacing: 8) {
                // Original text (Korean, Vietnamese, etc.)
                Text(segment.originalText)
                    .font(.headline)
                    .foregroundColor(.primary)
                    .lineLimit(2)
                
                // Translated text (English)
                Text(segment.translatedText)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                // Price and confidence row
                HStack {
                    // Show price if available
                    if let price = segment.price {
                        Text(price)
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.green)
                    }
                    
                    Spacer()
                    
                    // Confidence indicator
                    HStack(spacing: 4) {
                        Circle()
                            .fill(confidenceColor)
                            .frame(width: 8, height: 8)
                        
                        Text(confidenceText)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(.horizontal, 4)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
        .onAppear {
            loadImages()
        }
    }
    
    // MARK: - Food Image Section
    @ViewBuilder
    private var foodImageSection: some View {
        if isLoadingImages {
            gradientPlaceholder
                .overlay(ProgressView().tint(.white))
        } else if !imageUrls.isEmpty {
            // Horizontal scrolling carousel (max 3 images)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(imageUrls.prefix(3), id: \.absoluteString) { url in
                        AsyncImage(url: url) { phase in
                            switch phase {
                            case .empty:
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color.gray.opacity(0.3))
                                    .frame(width: 140, height: 100)
                                    .overlay(ProgressView().tint(.white))
                            case .success(let image):
                                image
                                    .resizable()
                                    .aspectRatio(contentMode: .fill)
                                    .frame(width: 140, height: 100)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                            case .failure:
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color.gray.opacity(0.3))
                                    .frame(width: 140, height: 100)
                                    .overlay(
                                        Image(systemName: "photo")
                                            .foregroundColor(.white.opacity(0.5))
                                    )
                            @unknown default:
                                EmptyView()
                            }
                        }
                    }
                }
            }
            .frame(height: 100)
        } else {
            gradientPlaceholder
        }
    }
    
    // MARK: - Load Images
    private func loadImages() {
        // Use translated text (English) for image search
        guard !segment.translatedText.isEmpty else { return }
        isLoadingImages = true
        
        Task {
            do {
                // Search for food + dish name for better results
                let searchQuery = "\(segment.translatedText) food dish"
                let urls = try await ImageSearchService.shared.searchImages(query: searchQuery)
                await MainActor.run {
                    imageUrls = urls
                    isLoadingImages = false
                }
            } catch {
                print("Failed to load images for \(segment.translatedText): \(error)")
                await MainActor.run {
                    isLoadingImages = false
                }
            }
        }
    }
    
    private var confidenceColor: Color {
        if segment.confidence >= 0.8 {
            return .green
        } else if segment.confidence >= 0.5 {
            return .orange
        } else {
            return .red
        }
    }
    
    private var confidenceText: String {
        let percentage = Int(segment.confidence * 100)
        return "\(percentage)% confidence"
    }
    
    /// Gradient placeholder with food icon
    @ViewBuilder
    private var gradientPlaceholder: some View {
        ZStack {
            LinearGradient(
                colors: gradientColors,
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .frame(height: 100)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            
            Image(systemName: foodIcon)
                .font(.system(size: 36))
                .foregroundColor(.white.opacity(0.9))
        }
    }
    
    /// Generate consistent gradient colors based on food name
    private var gradientColors: [Color] {
        let hash = segment.translatedText.hashValue
        let colorPairs: [[Color]] = [
            [Color.orange, Color.red],
            [Color.green, Color.teal],
            [Color.blue, Color.purple],
            [Color.pink, Color.orange],
            [Color.yellow, Color.orange],
            [Color.teal, Color.blue],
            [Color.purple, Color.pink],
            [Color.indigo, Color.blue]
        ]
        let index = abs(hash) % colorPairs.count
        return colorPairs[index]
    }
    
    /// Get appropriate food icon based on translated text
    private var foodIcon: String {
        let text = segment.translatedText.lowercased()
        if text.contains("soup") || text.contains("stew") || text.contains("guk") {
            return "cup.and.saucer.fill"
        } else if text.contains("rice") || text.contains("bap") {
            return "takeoutbag.and.cup.and.straw.fill"
        } else if text.contains("noodle") || text.contains("ramen") {
            return "takeoutbag.and.cup.and.straw.fill"
        } else if text.contains("meat") || text.contains("pork") || text.contains("beef") || text.contains("chicken") {
            return "flame.fill"
        } else if text.contains("fish") || text.contains("seafood") {
            return "fish.fill"
        } else if text.contains("drink") || text.contains("tea") || text.contains("coffee") {
            return "cup.and.saucer.fill"
        } else if text.contains("dessert") || text.contains("cake") || text.contains("sweet") {
            return "birthday.cake.fill"
        } else {
            return "fork.knife"
        }
    }
}

// MARK: - Image Picker
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    let onImageSelected: () -> Void
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // No updates needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
                parent.onImageSelected()
            }
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Preview
#Preview {
    StaticCaptureView()
}
