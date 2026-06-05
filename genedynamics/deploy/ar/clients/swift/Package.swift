// swift-tools-version:5.9
import PackageDescription

// Swift/RealityKit adapter for the corridor digital twin. RealityKit/simd are
// system frameworks (no external deps). To use the zero-copy FlatBuffers path,
// add the FlatBuffers SPM package and the generated Swift under
// ../../schema/generated/swift, then swap the JSON decode in WorldClient.
let package = Package(
    name: "CorridorTwin",
    platforms: [.visionOS(.v1), .iOS(.v17)],
    products: [
        .library(name: "CorridorTwin", targets: ["CorridorTwin"]),
    ],
    targets: [
        .target(name: "CorridorTwin"),
    ]
)
