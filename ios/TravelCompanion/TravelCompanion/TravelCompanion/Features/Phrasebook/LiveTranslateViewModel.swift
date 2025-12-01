import Foundation
import SwiftUI
import Speech
import AVFoundation
import Combine

@MainActor
final class LiveTranslateViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var translations: [TranslationCard] = []
    @Published var isListening = false
    @Published var isProcessing = false
    @Published var currentTranscription = ""
    @Published var errorMessage: String?
    @Published var showPermissionAlert = false
    @Published var targetLanguage = "en"
    
    // MARK: - Private Properties
    private let speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private let apiClient: APIClient
    
    private var lastProcessedText = ""
    private var silenceTimer: Timer?
    private let silenceThreshold: TimeInterval = 1.5 // seconds of silence before translating
    
    // MARK: - Initialization
    init(apiClient: APIClient? = nil) {
        self.apiClient = apiClient ?? APIClient.shared
        
        // Initialize speech recognizer with no specific locale to auto-detect
        self.speechRecognizer = SFSpeechRecognizer()
    }
    
    // MARK: - Public Methods
    func startListening() {
        // Check authorization status
        Task {
            let status = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status)
                }
            }
            switch status {
            case .authorized:
                requestMicrophonePermission()
            case .denied, .restricted:
                showPermissionAlert = true
            case .notDetermined:
                errorMessage = "Speech recognition not available"
            @unknown default:
                errorMessage = "Unknown authorization status"
            }
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        silenceTimer?.invalidate()
        silenceTimer = nil
        
        isListening = false
        
        // Process any remaining transcription
        if !currentTranscription.isEmpty && currentTranscription != lastProcessedText {
            let textToTranslate = currentTranscription
            Task {
                await translateAndAddCard(text: textToTranslate)
            }
        }
        
        currentTranscription = ""
    }
    
    func clearTranslations() {
        translations.removeAll()
        lastProcessedText = ""
    }
    
    // MARK: - Private Methods
    private func requestMicrophonePermission() {
        Task {
            let granted: Bool
            if #available(iOS 17.0, *) {
                granted = await AVAudioApplication.requestRecordPermission()
            } else {
                granted = await withCheckedContinuation { continuation in
                    AVAudioSession.sharedInstance().requestRecordPermission { result in
                        continuation.resume(returning: result)
                    }
                }
            }
            if granted {
                startRecognition()
            } else {
                showPermissionAlert = true
            }
        }
    }
    
    private func startRecognition() {
        // Cancel any existing task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Configure audio session
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            errorMessage = "Failed to configure audio session"
            return
        }
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            errorMessage = "Unable to create recognition request"
            return
        }
        
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.addsPunctuation = true
        
        // Get input node
        let inputNode = audioEngine.inputNode
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                
                if let result = result {
                    let transcription = result.bestTranscription.formattedString
                    self.currentTranscription = transcription
                    
                    // Reset silence timer
                    self.resetSilenceTimer()
                    
                    if result.isFinal {
                        self.handleFinalResult(transcription)
                    }
                }
                
                if let error = error {
                    // Check if it's just a timeout/no speech error (which is normal)
                    let nsError = error as NSError
                    if nsError.domain == "kAFAssistantErrorDomain" && nsError.code == 1110 {
                        // No speech detected, restart if still listening
                        if self.isListening {
                            self.restartRecognition()
                        }
                    } else {
                        self.errorMessage = "Recognition error: \(error.localizedDescription)"
                    }
                }
            }
        }
        
        // Configure audio input
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        // Start audio engine
        do {
            audioEngine.prepare()
            try audioEngine.start()
            isListening = true
            errorMessage = nil
        } catch {
            errorMessage = "Failed to start audio engine"
        }
    }
    
    private func resetSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleSilenceTimeout()
            }
        }
    }
    
    private func handleSilenceTimeout() {
        guard !currentTranscription.isEmpty && currentTranscription != lastProcessedText else { return }
        
        let textToTranslate = currentTranscription
        lastProcessedText = textToTranslate
        
        Task { [weak self] in
            guard let self else { return }
            await self.translateAndAddCard(text: textToTranslate)
            
            // Clear current transcription after translating
            self.currentTranscription = ""
            
            // Restart recognition for continuous listening
            if self.isListening {
                self.restartRecognition()
            }
        }
    }
    
    private func handleFinalResult(_ text: String) {
        guard !text.isEmpty && text != lastProcessedText else { return }
        
        lastProcessedText = text
        silenceTimer?.invalidate()
        
        Task { [weak self] in
            guard let self else { return }
            await self.translateAndAddCard(text: text)
            
            self.currentTranscription = ""
            
            // Restart for continuous listening
            if self.isListening {
                self.restartRecognition()
            }
        }
    }
    
    private func restartRecognition() {
        // Stop current recognition
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        // Small delay before restarting
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
            guard let self else { return }
            if self.isListening {
                self.startRecognition()
            }
        }
    }
    
    private func translateAndAddCard(text: String) async {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        isProcessing = true
        
        do {
            let response = try await apiClient.translateText(
                text: text,
                targetLanguage: targetLanguage
            )
            
            guard response.status == "ok", let data = response.data else {
                throw LiveTranslateError.apiError(response.error ?? "Translation failed")
            }
            
            let card = TranslationCard(
                originalText: data.originalText,
                translatedText: data.translatedText,
                sourceLanguage: data.sourceLanguage,
                targetLanguage: data.targetLanguage,
                timestamp: Date()
            )
            
            withAnimation(Animation.spring(response: 0.3, dampingFraction: 0.7)) {
                translations.append(card)
            }
            
            errorMessage = nil
            
        } catch {
            errorMessage = "Translation failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
}

// MARK: - Errors
enum LiveTranslateError: LocalizedError {
    case unauthorized
    case apiError(String)
    case recognitionFailed
    
    var errorDescription: String? {
        switch self {
        case .unauthorized:
            return "Authentication required"
        case .apiError(let message):
            return message
        case .recognitionFailed:
            return "Speech recognition failed"
        }
    }
}
