import SwiftUI
import AVFoundation
import UIKit
import Combine

struct CameraView: View {
    @StateObject private var viewModel = TranslationViewModel()
    @StateObject private var cameraManager = CameraManager()
    @State private var showingSettings = false
    @State private var selectedSegment: TranslationSegment?
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background - always black
                Color.black
                    .ignoresSafeArea()
                
                // Camera preview layer - always present, let AVFoundation handle visibility
                CameraPreviewView(session: cameraManager.session)
                    .ignoresSafeArea()
                    .opacity(cameraManager.isRunning ? 1 : 0)
                
                // Show loading indicator while waiting for camera
                if viewModel.cameraPermissionGranted && !cameraManager.isRunning && cameraManager.setupError == nil {
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.white)
                        Text("Starting camera...")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                    }
                }
                
                // Show permission denied message
                if !viewModel.cameraPermissionGranted && viewModel.errorMessage != nil {
                    VStack(spacing: 20) {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.gray)
                        
                        Text("Camera Access Required")
                            .font(.title2)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                        
                        Text("Please enable camera access in Settings to use the translation feature.")
                            .font(.body)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                        
                        Button("Open Settings") {
                            if let url = URL(string: UIApplication.openSettingsURLString) {
                                UIApplication.shared.open(url)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                
                // Show camera error message
                if let setupError = cameraManager.setupError {
                    VStack(spacing: 20) {
                        Image(systemName: "exclamationmark.camera.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.red)
                        
                        Text("Camera Error")
                            .font(.title2)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                        
                        Text(setupError)
                            .font(.body)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                    }
                }
                
                // Translation overlay
                ForEach(viewModel.translationSegments) { segment in
                    OverlaySegmentView(
                        segment: segment,
                        geometry: geometry,
                        onTap: {
                            selectedSegment = segment
                        }
                    )
                }
                
                // Controls overlay - always show top bar, conditionally show bottom controls
                VStack {
                    // Top bar - always visible
                    HStack {
                        Text(viewModel.detectedLanguage?.uppercased() ?? "---")
                            .font(.caption)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(.ultraThinMaterial)
                            .cornerRadius(8)
                        
                        Spacer()
                        
                        Button {
                            showingSettings = true
                        } label: {
                            Image(systemName: "gearshape.fill")
                                .foregroundColor(.white)
                                .padding(12)
                                .background(.ultraThinMaterial)
                                .clipShape(Circle())
                        }
                    }
                    .padding()
                    
                    Spacer()
                    
                    // Bottom controls - only when camera is ready
                    if cameraManager.isRunning {
                        HStack(spacing: 24) {
                            // Clear segments
                            Button {
                                viewModel.clearSegments()
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.white)
                            }
                            .disabled(viewModel.translationSegments.isEmpty)
                            
                            // Capture & translate
                            Button {
                                captureAndTranslate()
                            } label: {
                                ZStack {
                                    Circle()
                                        .fill(.white)
                                        .frame(width: 70, height: 70)
                                    
                                    Circle()
                                        .stroke(.white, lineWidth: 3)
                                        .frame(width: 82, height: 82)
                                    
                                    if viewModel.isProcessing {
                                        ProgressView()
                                            .progressViewStyle(.circular)
                                            .tint(.blue)
                                    }
                                }
                            }
                            .disabled(viewModel.isProcessing)
                            
                            // Language selector
                            Menu {
                                Button("English") { viewModel.targetLanguage = "en" }
                                Button("Spanish") { viewModel.targetLanguage = "es" }
                                Button("French") { viewModel.targetLanguage = "fr" }
                                Button("Japanese") { viewModel.targetLanguage = "ja" }
                            } label: {
                                HStack {
                                    Text(viewModel.targetLanguage.uppercased())
                                    Image(systemName: "chevron.down")
                                }
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(.ultraThinMaterial)
                                .cornerRadius(8)
                            }
                        }
                        .padding(.bottom, 40)
                    }
                }
                
                // Error message (only for non-permission errors when camera is running)
                if let error = viewModel.errorMessage, cameraManager.isRunning {
                    VStack {
                        Spacer()
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding()
                            .background(.red.opacity(0.8))
                            .cornerRadius(8)
                            .padding()
                    }
                    .transition(.move(edge: .bottom))
                }
            }
        }
        .sheet(item: $selectedSegment) { segment in
            SegmentDetailSheet(segment: segment, viewModel: viewModel)
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
        .task {
            await viewModel.requestCameraPermission()
            if viewModel.cameraPermissionGranted {
                await cameraManager.startSession()
            }
        }
        .onDisappear {
            cameraManager.stopSession()
        }
    }
    
    private func captureAndTranslate() {
        Task {
            if let image = await cameraManager.captureFrame() {
                await viewModel.translateFrame(image)
            }
        }
    }
}

/// Camera preview using AVCaptureVideoPreviewLayer
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer.backgroundColor = UIColor.black.cgColor
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        // Ensure session is set (in case it changed)
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
    }
}

/// Custom UIView that properly manages AVCaptureVideoPreviewLayer frame
class CameraPreviewUIView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }
    
    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        backgroundColor = .black
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
}

/// Camera session manager - handles AVCaptureSession on a background queue
final class CameraManager: NSObject, ObservableObject {
    // Session must be accessed from sessionQueue
    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    
    @MainActor @Published var isRunning = false
    @MainActor @Published var setupError: String?
    
    private var videoOutput: AVCaptureVideoDataOutput?
    private var currentFrame: UIImage?
    private var isConfigured = false
    
    @MainActor
    func startSession() async {
        guard !isConfigured else {
            // Already configured, just check if running
            let running = await withCheckedContinuation { continuation in
                sessionQueue.async {
                    continuation.resume(returning: self.session.isRunning)
                }
            }
            if running {
                isRunning = true
            }
            return
        }
        
        // Configure and start session on sessionQueue
        let configResult = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            sessionQueue.async { [weak self] in
                guard let self = self else {
                    continuation.resume(returning: false)
                    return
                }
                
                self.session.beginConfiguration()
                self.session.sessionPreset = .photo
                
                guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                    print("Camera device not available")
                    self.session.commitConfiguration()
                    continuation.resume(returning: false)
                    return
                }
                
                do {
                    let input = try AVCaptureDeviceInput(device: camera)
                    if self.session.canAddInput(input) {
                        self.session.addInput(input)
                    } else {
                        print("Cannot add camera input")
                        self.session.commitConfiguration()
                        continuation.resume(returning: false)
                        return
                    }
                    
                    let output = AVCaptureVideoDataOutput()
                    output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame.queue"))
                    
                    if self.session.canAddOutput(output) {
                        self.session.addOutput(output)
                        self.videoOutput = output
                    } else {
                        print("Cannot add video output")
                        self.session.commitConfiguration()
                        continuation.resume(returning: false)
                        return
                    }
                    
                    self.session.commitConfiguration()
                    
                    // Start the session
                    self.session.startRunning()
                    continuation.resume(returning: self.session.isRunning)
                    
                } catch {
                    print("Camera setup error: \(error)")
                    self.session.commitConfiguration()
                    continuation.resume(returning: false)
                }
            }
        }
        
        isConfigured = configResult
        isRunning = configResult
        
        if !configResult {
            setupError = "Failed to start camera"
        }
    }
    
    @MainActor
    func stopSession() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
        isRunning = false
    }
    
    @MainActor
    func captureFrame() async -> UIImage? {
        return currentFrame
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage)
        
        Task { @MainActor in
            self.currentFrame = image
        }
    }
}
