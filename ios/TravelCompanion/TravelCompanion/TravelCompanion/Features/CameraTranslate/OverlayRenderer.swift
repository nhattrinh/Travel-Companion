import SwiftUI
import Combine
import UIKit

/// Renders translation segment overlay on camera preview
struct OverlaySegmentView: View {
    let segment: TranslationSegment
    let geometry: GeometryProxy
    let onTap: () -> Void
    
    private var frame: CGRect {
        CGRect(
            x: segment.boundingBox.x * geometry.size.width,
            y: segment.boundingBox.y * geometry.size.height,
            width: segment.boundingBox.width * geometry.size.width,
            height: segment.boundingBox.height * geometry.size.height
        )
    }
    
    private var confidenceColor: Color {
        if segment.confidence >= 0.8 {
            return .green
        } else if segment.confidence >= 0.5 {
            return .yellow
        } else {
            return .orange
        }
    }
    
    var body: some View {
        ZStack(alignment: .topLeading) {
            // Bounding box outline
            Rectangle()
                .stroke(confidenceColor, lineWidth: 2)
                .frame(width: frame.width, height: frame.height)
                .position(x: frame.midX, y: frame.midY)
            
            // Translated text overlay
            Text(segment.translatedText)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    confidenceColor.opacity(0.8)
                        .cornerRadius(6)
                )
                .position(x: frame.midX, y: frame.minY - 16)
                .shadow(radius: 2)
        }
        .onTapGesture {
            onTap()
        }
    }
}

/// Detail sheet for a translation segment
struct SegmentDetailSheet: View {
    let segment: TranslationSegment
    @ObservedObject var viewModel: TranslationViewModel
    @Environment(\.dismiss) var dismiss
    @State private var showingSaveConfirmation = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Confidence indicator
                HStack {
                    Text("Confidence")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    ProgressView(value: segment.confidence)
                        .progressViewStyle(.linear)
                        .frame(width: 100)
                    
                    Text("\(Int(segment.confidence * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                // Original text
                VStack(alignment: .leading, spacing: 8) {
                    Text("Original")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(segment.originalText)
                        .font(.title3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                }
                
                Image(systemName: "arrow.down")
                    .foregroundColor(.secondary)
                
                // Translated text
                VStack(alignment: .leading, spacing: 8) {
                    Text("Translation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(segment.translatedText)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(12)
                }
                
                Spacer()
                
                // Actions
                HStack(spacing: 16) {
                    // Copy button
                    Button {
                        UIPasteboard.general.string = segment.translatedText
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    
                    // Save button
                    Button {
                        Task {
                            await viewModel.saveTranslation(segment: segment)
                            showingSaveConfirmation = true
                            try? await Task.sleep(nanoseconds: 1_500_000_000)
                            dismiss()
                        }
                    } label: {
                        Label("Save", systemImage: "star.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(.bottom)
            }
            .padding()
            .navigationTitle("Translation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
            .alert("Saved!", isPresented: $showingSaveConfirmation) {
                Button("OK", role: .cancel) { }
            }
        }
    }
}

/// Static image overlay renderer
struct StaticImageOverlay: View {
    let image: UIImage
    let segments: [TranslationSegment]
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                
                ForEach(segments) { segment in
                    OverlaySegmentView(
                        segment: segment,
                        geometry: geometry,
                        onTap: { }
                    )
                }
            }
        }
    }
}

/// Legacy overlay renderer for backward compatibility
struct OverlayRenderer: View {
    let segments: [OverlaySegment]
    
    struct OverlaySegment: Identifiable { 
        let id = UUID()
        let text: String
        let translated: String
    }
    
    var body: some View {
        ZStack {
            ForEach(segments) { seg in
                VStack(alignment: .leading, spacing: 4) {
                    Text(seg.text).font(.caption2)
                    Text(seg.translated).font(.caption).bold()
                }
                .padding(6)
                .background(.ultraThinMaterial)
                .cornerRadius(6)
            }
        }
    }
}
