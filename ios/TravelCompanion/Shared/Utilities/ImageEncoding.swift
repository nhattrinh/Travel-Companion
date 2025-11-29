import UIKit

/// Utility for encoding images to base64 for API transmission
enum ImageEncoding {
    
    /// Encode UIImage to base64 string with compression
    /// - Parameters:
    ///   - image: UIImage to encode
    ///   - compressionQuality: JPEG compression quality (0.0-1.0), default 0.8
    /// - Returns: Base64 encoded string, or nil if encoding fails
    static func encodeToBase64(_ image: UIImage, compressionQuality: CGFloat = 0.8) -> String? {
        guard let jpegData = image.jpegData(compressionQuality: compressionQuality) else {
            return nil
        }
        return jpegData.base64EncodedString()
    }
    
    /// Encode UIImage to base64 string with maximum size constraint
    /// - Parameters:
    ///   - image: UIImage to encode
    ///   - maxSizeBytes: Maximum size in bytes (default 2MB)
    /// - Returns: Base64 encoded string with compression adjusted to meet size constraint
    static func encodeToBase64WithSizeLimit(_ image: UIImage, maxSizeBytes: Int = 2_097_152) -> String? {
        var compressionQuality: CGFloat = 0.8
        var jpegData: Data?
        
        // Iteratively reduce quality until size constraint met
        while compressionQuality > 0.1 {
            jpegData = image.jpegData(compressionQuality: compressionQuality)
            if let data = jpegData, data.count <= maxSizeBytes {
                break
            }
            compressionQuality -= 0.1
        }
        
        guard let finalData = jpegData else {
            return nil
        }
        
        return finalData.base64EncodedString()
    }
    
    /// Decode base64 string to UIImage
    /// - Parameter base64String: Base64 encoded image string
    /// - Returns: UIImage, or nil if decoding fails
    static func decodeFromBase64(_ base64String: String) -> UIImage? {
        guard let imageData = Data(base64Encoded: base64String) else {
            return nil
        }
        return UIImage(data: imageData)
    }
    
    /// Legacy method for compatibility
    static func jpegData(from image: UIImage, quality: CGFloat = 0.8) -> Data? {
        image.jpegData(compressionQuality: quality)
    }
}
