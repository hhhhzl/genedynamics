import Foundation

// Codable mirror of the WorldSnapshot JSON contract (schema/world.fbs shape).
// JSON keeps this dependency-light; for the zero-copy hot path, swap the decode
// to the flatc-generated Swift (schema/generated/swift/world_generated.swift) +
// the FlatBuffers SPM package, and point world_ws at flatbuffers_codec.encode.

public struct WorldMessage: Codable {
    public let schema_version: Int?
    public let frame: String?
    public let tracker: String?
    public let units: String?
    public let is_keyframe: Bool
    public let entities: [EntityMsg]
    public let removed_ids: [String]?
}

public struct EntityMsg: Codable {
    public let id: String
    public let type: String
    public let frame: String?
    public let pose: PoseMsg
    public let geom: GeomMsg?
    public let color_rgba: UInt32?
    public let rev: UInt32?
    public let meta: String?
}

public struct PoseMsg: Codable {
    public let p: [Float]   // [x, y, z]   world frame
    public let q: [Float]   // [w, x, y, z] world frame
}

public struct GeomMsg: Codable {
    public let kind: String          // box|sphere|cylinder|capsule|usd|urdf|path
    public let half_extents: [Float]?
    public let radius: Float?
    public let height: Float?
    public let asset_uri: String?
}
