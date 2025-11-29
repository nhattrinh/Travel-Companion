import Foundation

struct UserDTO: Codable, Identifiable {
    let id: Int
    let email: String
    let preferences: [String: AnyCodable]?
}

// Lightweight wrapper to encode arbitrary JSON values in preferences
// This avoids prematurely modeling dynamic user preference keys.
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) { self.value = value }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) { value = intVal; return }
        if let dblVal = try? container.decode(Double.self) { value = dblVal; return }
        if let boolVal = try? container.decode(Bool.self) { value = boolVal; return }
        if let strVal = try? container.decode(String.self) { value = strVal; return }
        if let arrVal = try? container.decode([AnyCodable].self) { value = arrVal.map{ $0.value }; return }
        if let dictVal = try? container.decode([String: AnyCodable].self) { value = dictVal.mapValues{ $0.value }; return }
        throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON type")
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let intVal as Int: try container.encode(intVal)
        case let dblVal as Double: try container.encode(dblVal)
        case let boolVal as Bool: try container.encode(boolVal)
        case let strVal as String: try container.encode(strVal)
        case let arrVal as [Any]: try container.encode(arrVal.map { AnyCodable($0) })
        case let dictVal as [String: Any]: try container.encode(dictVal.mapValues { AnyCodable($0) })
        default:
            throw EncodingError.invalidValue(value, .init(codingPath: container.codingPath, debugDescription: "Unsupported JSON type"))
        }
    }
}
