#if canImport(UIKit)
import UIKit
public typealias PlatformImage = UIImage
#elseif canImport(AppKit)
import AppKit
public typealias PlatformImage = NSImage
#endif

/// Utility for encoding images to base64 for API transmission
enum ImageEncoding {
    
    /// Encode image to base64 string with compression
    /// - Parameters:
    ///   - image: Image to encode
    ///   - compressionQuality: JPEG compression quality (0.0-1.0), default 0.8
    /// - Returns: Base64 encoded string, or nil if encoding fails
    static func encodeToBase64(_ image: PlatformImage, compressionQuality: CGFloat = 0.8) -> String? {
        guard let jpegData = jpegData(from: image, quality: compressionQuality) else {
            return nil
        }
        return jpegData.base64EncodedString()
    }
    
    /// Encode image to base64 string with maximum size constraint
    /// - Parameters:
    ///   - image: Image to encode
    ///   - maxSizeBytes: Maximum size in bytes (default 2MB)
    /// - Returns: Base64 encoded string with compression adjusted to meet size constraint
    static func encodeToBase64WithSizeLimit(_ image: PlatformImage, maxSizeBytes: Int = 2_097_152) -> String? {
        var compressionQuality: CGFloat = 0.8
        var data: Data?
        
        // Iteratively reduce quality until size constraint met
        while compressionQuality > 0.1 {
            data = jpegData(from: image, quality: compressionQuality)
            if let d = data, d.count <= maxSizeBytes {
                break
            }
            compressionQuality -= 0.1
        }
        
        guard let finalData = data else {
            return nil
        }
        
        return finalData.base64EncodedString()
    }
    
    /// Decode base64 string to image
    /// - Parameter base64String: Base64 encoded image string
    /// - Returns: Image, or nil if decoding fails
    static func decodeFromBase64(_ base64String: String) -> PlatformImage? {
        guard let imageData = Data(base64Encoded: base64String) else {
            return nil
        }
        #if canImport(UIKit)
        return UIImage(data: imageData)
        #elseif canImport(AppKit)
        return NSImage(data: imageData)
        #endif
    }
    
    /// Get JPEG data from image
    static func jpegData(from image: PlatformImage, quality: CGFloat = 0.8) -> Data? {
        #if canImport(UIKit)
        return image.jpegData(compressionQuality: quality)
        #elseif canImport(AppKit)
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else { return nil }
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: quality])
        #endif
    }
}
