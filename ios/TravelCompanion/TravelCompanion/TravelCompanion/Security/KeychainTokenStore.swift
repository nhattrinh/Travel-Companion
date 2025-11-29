import Foundation
import Security

final class KeychainTokenStore {
    static let shared = KeychainTokenStore()
    private init() {}

    private let accessKey = "travel_access_token"
    private let refreshKey = "travel_refresh_token"

    func save(access: String, refresh: String) {
        save(key: accessKey, value: access)
        save(key: refreshKey, value: refresh)
    }

    func accessToken() -> String? { read(key: accessKey) }
    func refreshToken() -> String? { read(key: refreshKey) }

    func clear() {
        delete(key: accessKey)
        delete(key: refreshKey)
    }

    private func save(key: String, value: String) {
        let encoded = value.data(using: .utf8)!
        delete(key: key)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: encoded,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]
        SecItemAdd(query as CFDictionary, nil)
    }

    private func read(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func delete(key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        SecItemDelete(query as CFDictionary)
    }
}
