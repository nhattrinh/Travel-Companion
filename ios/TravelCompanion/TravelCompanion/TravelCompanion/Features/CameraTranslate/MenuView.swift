import SwiftUI
import PhotosUI

struct MenuView: View {
    @State private var showingCamera = false
    @State private var showingPhotoPicker = false
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var showingStaticCapture = false
    @State private var isLoadingImage = false
    
    var body: some View {
        NavigationStack {
            ZStack {
                // Background - matches other pages
                Color(.systemBackground)
                    .ignoresSafeArea()
                
                VStack(alignment: .leading, spacing: 40) {
                    // Subtitle
                    Text("Select an option to begin")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .padding(.horizontal)
                    
                    Spacer()
                    
                    // Options
                    VStack(spacing: 20) {
                        // Camera Roll Option
                        PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                            MenuOptionButton(
                                icon: "photo.on.rectangle",
                                title: "Camera Roll",
                                subtitle: "Select a photo of a menu",
                            )
                        }
                        .disabled(isLoadingImage)
                        
                        // Live Camera Option
                        Button {
                            showingCamera = true
                        } label: {
                            MenuOptionButton(
                                icon: "camera.fill",
                                title: "Live Camera",
                                subtitle: "Visualize menu in real-time"
                            )
                        }
                        .disabled(isLoadingImage)
                    }
                    .padding(.horizontal, 30)
                    
                    Spacer()
                    Spacer()
                }
                
                // Loading overlay
                if isLoadingImage {
                    Color.black.opacity(0.3)
                        .ignoresSafeArea()
                    ProgressView("Loading image...")
                        .padding()
                        .background(.ultraThinMaterial)
                        .cornerRadius(12)
                }
            }
            .onChange(of: selectedPhotoItem) { _, newItem in
                guard let newItem = newItem else { return }
                isLoadingImage = true
                
                Task {
                    do {
                        if let data = try await newItem.loadTransferable(type: Data.self),
                           let originalImage = UIImage(data: data) {
                            // Resize image to prevent memory issues
                            let resizedImage = resizeImageIfNeeded(originalImage, maxDimension: 1920)
                            await MainActor.run {
                                selectedImage = resizedImage
                                isLoadingImage = false
                                showingStaticCapture = true
                            }
                        } else {
                            await MainActor.run {
                                isLoadingImage = false
                            }
                        }
                    } catch {
                        print("Error loading image: \(error)")
                        await MainActor.run {
                            isLoadingImage = false
                        }
                    }
                }
            }
            .fullScreenCover(isPresented: $showingCamera) {
                CameraView()
            }
            .fullScreenCover(isPresented: $showingStaticCapture) {
                if let image = selectedImage {
                    StaticCaptureView(image: image) {
                        showingStaticCapture = false
                        selectedImage = nil
                        selectedPhotoItem = nil
                    }
                }
            }
            .navigationTitle("Menu Visualizer")
        }
    }
    
    /// Resize image if it exceeds the maximum dimension to prevent memory issues
    private func resizeImageIfNeeded(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
        let size = image.size
        
        // Check if resize is needed
        guard size.width > maxDimension || size.height > maxDimension else {
            return image
        }
        
        // Calculate new size maintaining aspect ratio
        let ratio = min(maxDimension / size.width, maxDimension / size.height)
        let newSize = CGSize(width: size.width * ratio, height: size.height * ratio)
        
        // Create resized image
        let renderer = UIGraphicsImageRenderer(size: newSize)
        let resizedImage = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: newSize))
        }
        
        return resizedImage
    }
}

struct MenuOptionButton: View {
    let icon: String
    let title: String
    let subtitle: String
    
    var body: some View {
        HStack(spacing: 20) {
            // Icon
            Image(systemName: icon)
                .font(.system(size: 28))
                .foregroundColor(.blue)
                .frame(width: 50, height: 50)
                .background(Color.blue.opacity(0.2))
                .cornerRadius(12)
            
            // Text
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.black)
                
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            
            Spacer()
            
            // Chevron
            Image(systemName: "chevron.right")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.gray)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 18)
        .background(Color.white.opacity(0.1))
        .cornerRadius(16)
    }
}

#Preview {
    MenuView()
}
