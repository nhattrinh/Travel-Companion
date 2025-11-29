import SwiftUI

struct RecentTranslationsView: View {
    @ObservedObject var viewModel: TripOverviewViewModel
    @State private var selectedTranslation: TranslationHistory?
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                if viewModel.recentTranslations.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "clock")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("No translation history")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Your translations will appear here")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 60)
                } else {
                    ForEach(viewModel.recentTranslations) { translation in
                        TranslationHistoryRow(
                            translation: translation,
                            onTap: { selectedTranslation = translation },
                            onCopy: { viewModel.copyTranslation(translation) },
                            onDelete: {
                                Task {
                                    await viewModel.deleteTranslation(translation)
                                }
                            }
                        )
                        Divider()
                    }
                }
            }
        }
        .sheet(item: $selectedTranslation) { translation in
            TranslationDetailSheet(translation: translation, viewModel: viewModel)
        }
    }
}

struct TranslationHistoryRow: View {
    let translation: TranslationHistory
    let onTap: () -> Void
    let onCopy: () -> Void
    let onDelete: () -> Void
    
    @State private var showCopiedFeedback = false
    
    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 12) {
                // Thumbnail or icon
                if let _ = translation.imageUrl {
                    Image(systemName: "photo")
                        .font(.title2)
                        .foregroundColor(.blue)
                        .frame(width: 50, height: 50)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                } else {
                    Image(systemName: "text.bubble")
                        .font(.title2)
                        .foregroundColor(.blue)
                        .frame(width: 50, height: 50)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                }
                
                VStack(alignment: .leading, spacing: 6) {
                    // Translated text (prominent)
                    Text(translation.translatedText)
                        .font(.body)
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                        .lineLimit(2)
                    
                    // Original text
                    Text(translation.originalText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                    
                    // Metadata
                    HStack(spacing: 12) {
                        Label(translation.timeAgo, systemImage: "clock")
                            .font(.caption2)
                        
                        Label(translation.confidenceText, systemImage: "checkmark.circle")
                            .font(.caption2)
                        
                        Text("\(translation.sourceLanguage.uppercased()) → \(translation.targetLanguage.uppercased())")
                            .font(.caption2)
                    }
                    .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // Actions
                VStack(spacing: 8) {
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
                        Image(systemName: showCopiedFeedback ? "checkmark" : "doc.on.doc")
                            .font(.body)
                            .foregroundColor(.blue)
                    }
                    
                    Button(action: onDelete) {
                        Image(systemName: "trash")
                            .font(.body)
                            .foregroundColor(.red)
                    }
                }
            }
            .padding()
        }
        .buttonStyle(.plain)
    }
}

struct TranslationDetailSheet: View {
    let translation: TranslationHistory
    @ObservedObject var viewModel: TripOverviewViewModel
    @Environment(\.dismiss) var dismiss
    @State private var showCopiedFeedback = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Image preview
                    if let _ = translation.imageUrl {
                        Image(systemName: "photo")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity)
                            .frame(height: 200)
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                            .padding(.horizontal)
                    }
                    
                    // Translation content
                    VStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Translated")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Text(translation.translatedText)
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.blue)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(12)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Original")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Text(translation.originalText)
                                .font(.body)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }
                    .padding(.horizontal)
                    
                    // Metadata
                    VStack(spacing: 12) {
                        HStack {
                            Label("Confidence", systemImage: "checkmark.circle")
                            Spacer()
                            Text(translation.confidenceText)
                                .foregroundColor(.blue)
                        }
                        
                        Divider()
                        
                        HStack {
                            Label("Languages", systemImage: "globe")
                            Spacer()
                            Text("\(translation.sourceLanguage.uppercased()) → \(translation.targetLanguage.uppercased())")
                                .foregroundColor(.secondary)
                        }
                        
                        Divider()
                        
                        HStack {
                            Label("Time", systemImage: "clock")
                            Spacer()
                            Text(translation.timeAgo)
                                .foregroundColor(.secondary)
                        }
                        
                        if let tripId = translation.tripId {
                            Divider()
                            
                            HStack {
                                Label("Trip ID", systemImage: "airplane")
                                Spacer()
                                Text("#\(tripId)")
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    .padding(.horizontal)
                    
                    Spacer()
                    
                    // Actions
                    VStack(spacing: 12) {
                        Button {
                            viewModel.copyTranslation(translation)
                            withAnimation {
                                showCopiedFeedback = true
                            }
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                withAnimation {
                                    showCopiedFeedback = false
                                }
                            }
                        } label: {
                            HStack {
                                Image(systemName: showCopiedFeedback ? "checkmark" : "doc.on.doc")
                                Text(showCopiedFeedback ? "Copied!" : "Copy Translation")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(showCopiedFeedback)
                        
                        Button {
                            shareTranslation()
                        } label: {
                            HStack {
                                Image(systemName: "square.and.arrow.up")
                                Text("Share")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                        }
                        .buttonStyle(.bordered)
                        
                        Button(role: .destructive) {
                            Task {
                                await viewModel.deleteTranslation(translation)
                                dismiss()
                            }
                        } label: {
                            HStack {
                                Image(systemName: "trash")
                                Text("Delete")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding()
                }
            }
            .navigationTitle("Translation Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }
    
    private func shareTranslation() {
        let text = "\(translation.translatedText)\n(\(translation.originalText))"
        let activityVC = UIActivityViewController(activityItems: [text], applicationActivities: nil)
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootVC = windowScene.windows.first?.rootViewController {
            rootVC.present(activityVC, animated: true)
        }
    }
}
