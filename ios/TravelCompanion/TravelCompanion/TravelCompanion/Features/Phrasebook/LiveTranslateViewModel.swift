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
    @Published var sourceLanguage: String = "en" // Language user will speak in
    
    // Target language is read from UserDefaults (set in Settings)
    var targetLanguage: String {
        UserDefaults.standard.string(forKey: "defaultLanguage") ?? "en"
    }
    
    // MARK: - Language Support
    // Map language codes to locales for speech recognition
    static let languageLocaleMap: [String: Locale] = [
        "en": Locale(identifier: "en-US"),
        "ko": Locale(identifier: "ko-KR"),
        "vi": Locale(identifier: "vi-VN")
    ]
    
    static let languageNames: [String: String] = [
        "en": "English",
        "ko": "Korean",
        "vi": "Vietnamese"
    ]
    
    // MARK: - Private Properties
    private var speechRecognizers: [String: SFSpeechRecognizer] = [:]
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private let apiClient: APIClient
    
    private var lastProcessedText = ""
    private var silenceTimer: Timer?
    private let silenceThreshold: TimeInterval = 1.5 // seconds of silence before translating
    
    // MARK: - Computed Properties
    var currentRecognizer: SFSpeechRecognizer? {
        speechRecognizers[sourceLanguage]
    }
    
    // MARK: - Initialization
    init(apiClient: APIClient? = nil) {
        self.apiClient = apiClient ?? APIClient.shared
        
        // Initialize speech recognizers for supported languages
        for (code, locale) in Self.languageLocaleMap {
            if let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable {
                // Enable default task hint for better accuracy
                if #available(iOS 13.0, *) {
                    recognizer.defaultTaskHint = .dictation
                }
                speechRecognizers[code] = recognizer
            }
        }
    }
    
    // MARK: - Contextual Strings for Better Accuracy
    /// Returns common travel phrases to help speech recognition accuracy
    private func getContextualStrings() -> [String] {
        // Common travel phrases in the source language
        switch sourceLanguage {
        case "en":
            return [
                "hello", "goodbye", "thank you", "please", "excuse me",
                "where is", "how much", "I need", "can you help",
                "the bathroom", "the restaurant", "the hotel", "the airport",
                "I don't understand", "do you speak English",
                "water", "food", "menu", "check please", "bill",
                "taxi", "bus", "train", "subway", "station"
            ]
        case "ko":
            return [
                "안녕하세요", "감사합니다", "죄송합니다", "실례합니다",
                "얼마예요", "어디예요", "도와주세요", "화장실",
                "식당", "호텔", "공항", "택시", "버스", "지하철",
                "물", "음식", "메뉴", "계산서", "영어"
            ]
        case "vi":
            return [
                "xin chào", "cảm ơn", "xin lỗi", "làm ơn",
                "bao nhiêu", "ở đâu", "giúp tôi", "nhà vệ sinh",
                "nhà hàng", "khách sạn", "sân bay", "taxi", "xe buýt",
                "nước", "thức ăn", "thực đơn", "tính tiền", "tiếng Anh"
            ]
        default:
            return []
        }
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
        do {
            audioEngine.inputNode.removeTap(onBus: 0)
        } catch {
            // Ignore errors when removing tap - it may not exist
        }
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
            // Use .playAndRecord to support external audio devices better
            // .allowBluetooth enables Bluetooth headset microphones
            // .defaultToSpeaker ensures audio comes from speaker when no headphones
            try audioSession.setCategory(
                .playAndRecord,
                mode: .measurement,
                options: [.duckOthers, .allowBluetooth, .defaultToSpeaker]
            )
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            errorMessage = "Failed to configure audio session: \(error.localizedDescription)"
            return
        }
        
        // Check if audio input is available (microphone connected)
        guard audioSession.isInputAvailable else {
            errorMessage = "No microphone available. Please connect a microphone or open your laptop lid."
            return
        }
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            errorMessage = "Unable to create recognition request"
            return
        }
        
        // Configure for better accuracy
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.addsPunctuation = true
        
        // Use on-device recognition for better accuracy and privacy when available
        if #available(iOS 13.0, *) {
            recognitionRequest.requiresOnDeviceRecognition = false // Set to true for offline, but less accurate
        }
        
        // Set task hint for dictation (better for general speech)
        if #available(iOS 16.0, *) {
            recognitionRequest.taskHint = .dictation
        }
        
        // Add contextual strings for common travel phrases to improve accuracy
        if #available(iOS 14.0, *) {
            recognitionRequest.contextualStrings = getContextualStrings()
        }
        
        // Get input node - wrap in do-catch to handle hardware unavailable
        let inputNode: AVAudioInputNode
        do {
            // Accessing inputNode can throw if no audio input hardware is available
            inputNode = audioEngine.inputNode
            
            // Check if the input node has valid format
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else {
                errorMessage = "No valid audio input format. Please check your microphone."
                return
            }
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                recognitionRequest.append(buffer)
            }
        } catch {
            errorMessage = "Failed to access microphone: \(error.localizedDescription)"
            return
        }
        
        // Start recognition task
        recognitionTask = currentRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
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
                    // Check if it's just a timeout/no speech/cancelled error (which is normal)
                    let nsError = error as NSError
                    
                    // kAFAssistantErrorDomain errors that are expected and should be ignored:
                    // 1110 - No speech detected (timeout)
                    // 216 - Recognition was cancelled (user stopped)
                    // 209 - Recognition request was invalidated
                    // 203 - Recognition was cancelled
                    // 1101 - Retry (temporary failure)
                    let ignorableErrorCodes = [216, 203, 209, 1101, 1110]
                    
                    if nsError.domain == "kAFAssistantErrorDomain" && ignorableErrorCodes.contains(nsError.code) {
                        // These are expected errors, don't show to user
                        // If it's a timeout (1110), restart if still listening
                        if nsError.code == 1110 && self.isListening {
                            self.restartRecognition()
                        }
                    } else if nsError.domain == "kAFAssistantErrorDomain" {
                        // Unknown kAFAssistantErrorDomain error - log but don't show unless significant
                        print("Speech recognition error: \(nsError.code) - \(error.localizedDescription)")
                    } else {
                        // Other errors should be shown
                        self.errorMessage = "Recognition error: \(error.localizedDescription)"
                    }
                }
            }
        }
        
        // Start audio engine
        do {
            audioEngine.prepare()
            try audioEngine.start()
            isListening = true
            errorMessage = nil
        } catch {
            // Clean up the tap we installed if engine fails to start
            inputNode.removeTap(onBus: 0)
            errorMessage = "Failed to start audio engine: \(error.localizedDescription)"
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
        do {
            audioEngine.inputNode.removeTap(onBus: 0)
        } catch {
            // Ignore errors when removing tap - it may not exist
        }
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
