import Foundation
import SwiftUI
import AVFoundation
import Combine
import UIKit

@MainActor
final class TranslationViewModel: ObservableObject {
    @Published var translationSegments: [TranslationSegment] = []
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var detectedLanguage: String?
    @Published var targetLanguage = "en"
    @Published var cameraPermissionGranted = false
    @Published var latencyMS: Double = 0
    
    // Legacy support
    @Published var segments: [OverlayRenderer.OverlaySegment] = []
    
    private let apiClient: APIClient
    private let authService: AuthService
    
    init(apiClient: APIClient = .shared, authService: AuthService = .shared) {
        self.apiClient = apiClient
        self.authService = authService
    }
    
    /// Request camera permission
    func requestCameraPermission() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        
        switch status {
        case .authorized:
            cameraPermissionGranted = true
        case .notDetermined:
            cameraPermissionGranted = await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            cameraPermissionGranted = false
            errorMessage = "Camera access denied. Please enable in Settings."
        @unknown default:
            cameraPermissionGranted = false
        }
    }
    
    /// Translate image frame from camera
    func translateFrame(_ image: UIImage) async {
        guard !isProcessing else { return }
        
        let start = Date()
        isProcessing = true
        errorMessage = nil
        
        do {
            // Encode image to base64
            guard let base64Image = ImageEncoding.encodeToBase64WithSizeLimit(image, maxSizeBytes: 2_097_152) else {
                throw TranslationError.imageEncodingFailed
            }
            
            // Get auth token
            let token = try await authService.getAccessToken()
            
            // Call live-frame endpoint
            let response = try await apiClient.postTranslationFrame(
                imageBase64: base64Image,
                targetLanguage: targetLanguage,
                token: token
            )
            
            // Handle envelope response
            guard response.status == "ok", let data = response.data else {
                throw TranslationError.apiError(response.error ?? "Unknown error")
            }
            
            // Update segments
            translationSegments = data.segments.map { $0.toSegment() }
            detectedLanguage = data.detectedLanguage
            latencyMS = Date().timeIntervalSince(start) * 1000
            
            // Update legacy segments for compatibility
            segments = translationSegments.map {
                OverlayRenderer.OverlaySegment(text: $0.originalText, translated: $0.translatedText)
            }
            
        } catch {
            errorMessage = error.localizedDescription
            translationSegments = []
            segments = []
        }
        
        isProcessing = false
    }
    
    /// Legacy method for mock translation
    func translateFrame(dummyText: [String]) async {
        let start = Date()
        segments = dummyText.map { OverlayRenderer.OverlaySegment(text: $0, translated: "[JA] \($0)") }
        latencyMS = Date().timeIntervalSince(start) * 1000
    }
    
    /// Save translation to history
    func saveTranslation(segment: TranslationSegment) async {
        do {
            let token = try await authService.getAccessToken()
            
            let request = SaveTranslationRequest(
                originalText: segment.originalText,
                translatedText: segment.translatedText,
                sourceLanguage: detectedLanguage ?? "unknown",
                targetLanguage: targetLanguage,
                confidence: segment.confidence
            )
            
            let response = try await apiClient.saveTranslation(request, token: token)
            
            guard response.status == "ok" else {
                throw TranslationError.apiError(response.error ?? "Failed to save")
            }
            
            // Show success feedback
            print("Translation saved with ID: \(response.data?.id ?? 0)")
            
        } catch {
            errorMessage = "Failed to save translation: \(error.localizedDescription)"
        }
    }
    
    /// Clear current segments
    func clearSegments() {
        translationSegments = []
        segments = []
        detectedLanguage = nil
        errorMessage = nil
    }
}

enum TranslationError: LocalizedError {
    case imageEncodingFailed
    case unauthorized
    case apiError(String)
    
    var errorDescription: String? {
        switch self {
        case .imageEncodingFailed:
            return "Failed to encode image"
        case .unauthorized:
            return "Authentication required"
        case .apiError(let message):
            return message
        }
    }
}
