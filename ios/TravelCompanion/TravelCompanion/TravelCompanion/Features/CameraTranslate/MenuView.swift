import SwiftUI
import PhotosUI

struct MenuView: View {
    @State private var showingCamera = false
    @State private var showingPhotoPicker = false
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var showingStaticCapture = false
    
    var body: some View {
        NavigationStack {
            ZStack {
                // Background
                Color.black
                    .ignoresSafeArea()
                
                VStack(spacing: 40) {
                    Spacer()
                    
                    // Title
                    Text("Menu")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    
                    Text("Choose how to translate")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                    
                    Spacer()
                    
                    // Options
                    VStack(spacing: 20) {
                        // Camera Roll Option
                        PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                            MenuOptionButton(
                                icon: "photo.on.rectangle",
                                title: "Camera Roll",
                                subtitle: "Select a photo to translate"
                            )
                        }
                        
                        // Live Camera Option
                        Button {
                            showingCamera = true
                        } label: {
                            MenuOptionButton(
                                icon: "camera.fill",
                                title: "Live Camera",
                                subtitle: "Translate in real-time"
                            )
                        }
                    }
                    .padding(.horizontal, 30)
                    
                    Spacer()
                    Spacer()
                }
            }
            .onChange(of: selectedPhotoItem) { _, newItem in
                Task {
                    if let data = try? await newItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        selectedImage = image
                        showingStaticCapture = true
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
        }
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
                    .foregroundColor(.white)
                
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
