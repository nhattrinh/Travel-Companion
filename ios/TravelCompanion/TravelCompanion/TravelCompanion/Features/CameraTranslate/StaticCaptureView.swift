import SwiftUI
import Combine
import PhotosUI
import UIKit

/// View for translating static images from photo library
struct StaticCaptureView: View {
    @StateObject private var viewModel = TranslationViewModel()
    @State private var selectedImage: UIImage?
    @State private var showingImagePicker = false
    @State private var selectedSegment: TranslationSegment?
    
    var body: some View {
        NavigationView {
            VStack {
                if let image = selectedImage {
                    // Image with overlay
                    GeometryReader { geometry in
                        ZStack {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                            
                            ForEach(viewModel.translationSegments) { segment in
                                OverlaySegmentView(
                                    segment: segment,
                                    geometry: geometry,
                                    onTap: {
                                        selectedSegment = segment
                                    }
                                )
                            }
                        }
                    }
                    .frame(maxHeight: 500)
                    
                    // Translate button
                    if viewModel.translationSegments.isEmpty && !viewModel.isProcessing {
                        Button {
                            Task {
                                await viewModel.translateFrame(image)
                            }
                        } label: {
                            Label("Translate Image", systemImage: "textformat.alt")
                                .frame(maxWidth: .infinity)
                                .padding()
                        }
                        .buttonStyle(.borderedProminent)
                        .padding()
                    }
                    
                    // Results info
                    if !viewModel.translationSegments.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Found \(viewModel.translationSegments.count) text segment(s)")
                                .font(.headline)
                            
                            if let lang = viewModel.detectedLanguage {
                                Text("Detected language: \(lang.uppercased())")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Text("Tap a segment to view details")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                        .padding()
                    }
                    
                    Spacer()
                    
                } else {
                    // Empty state
                    VStack(spacing: 24) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 80))
                            .foregroundColor(.secondary)
                        
                        Text("Select a Photo to Translate")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("Choose an image containing text you'd like to translate")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        
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
                
                // Processing indicator
                if viewModel.isProcessing {
                    ProgressView("Translating...")
                        .padding()
                }
                
                // Error message
                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                        .padding()
                }
            }
            .navigationTitle("Image Translation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showingImagePicker = true
                    } label: {
                        Image(systemName: "photo.badge.plus")
                    }
                    .disabled(viewModel.isProcessing)
                }
                
                if selectedImage != nil {
                    ToolbarItem(placement: .cancellationAction) {
                        Button {
                            selectedImage = nil
                            viewModel.clearSegments()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                        }
                    }
                }
            }
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(selectedImage: $selectedImage) {
                    viewModel.clearSegments()
                }
            }
            .sheet(item: $selectedSegment) { segment in
                SegmentDetailSheet(segment: segment, viewModel: viewModel)
            }
        }
    }
}

/// UIKit Image Picker wrapper
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
