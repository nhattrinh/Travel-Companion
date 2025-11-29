import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var viewModel = TranslationViewModel()
    @StateObject private var cameraManager = CameraManager()
    @State private var showingSettings = false
    @State private var selectedSegment: TranslationSegment?
    
    var body: some View {
        ZStack {
            // Camera preview layer
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()
            
            // Translation overlay
            GeometryReader { geometry in
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
            
            // Controls overlay
            VStack {
                // Top bar
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
                
                // Bottom controls
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
            
            // Error message
            if let error = viewModel.errorMessage {
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
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        context.coordinator.previewLayer = previewLayer
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // Update frame on layout changes
        DispatchQueue.main.async {
            context.coordinator.previewLayer?.frame = uiView.bounds
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}

/// Camera session manager
@MainActor
final class CameraManager: ObservableObject {
    let session = AVCaptureSession()
    private var videoOutput: AVCaptureVideoDataOutput?
    private var currentFrame: UIImage?
    
    func startSession() async {
        session.sessionPreset = .photo
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if session.canAddInput(input) {
                session.addInput(input)
            }
            
            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame.queue"))
            
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
            
            videoOutput = output
            
            session.startRunning()
        } catch {
            print("Camera setup error: \(error)")
        }
    }
    
    func stopSession() {
        session.stopRunning()
    }
    
    func captureFrame() async -> UIImage? {
        return currentFrame
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
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

/// Settings view placeholder
struct SettingsView: View {
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section("Translation") {
                    Text("Default target language: English")
                    Text("Auto-detect source language: On")
                }
                
                Section("Camera") {
                    Text("Resolution: High")
                    Text("Flash: Auto")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}
