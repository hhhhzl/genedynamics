import Foundation
import simd

// Per-entity live state + client-side INTERPOLATION buffer (world frame).
// Renders smoothly at display rate from a ~30 Hz network stream by sampling the
// pose at (now - interpolationDelay), between two buffered samples.
public final class EntityState {
    public let id: String
    public var type: String = "obstacle"
    public var geom: GeomMsg?
    public var colorRgba: UInt32 = 0xFFFFFFFF
    public var rev: UInt32 = 0
    public var geomDirty = true

    private struct Sample { let t: Double; let p: SIMD3<Float>; let q: simd_quatf }
    private var buf: [Sample] = []
    private let capacity = 8

    public init(id: String) { self.id = id }

    public func push(_ t: Double, _ p: SIMD3<Float>, _ q: simd_quatf) {
        if let last = buf.last, t <= last.t { return }   // keep monotonic
        buf.append(Sample(t: t, p: p, q: q))
        if buf.count > capacity { buf.removeFirst() }
    }

    /// Interpolated world pose at `t`; holds endpoints outside the buffer span
    /// (no extrapolation). nil until the first sample.
    public func sample(at t: Double) -> (SIMD3<Float>, simd_quatf)? {
        guard let first = buf.first else { return nil }
        if buf.count == 1 || t <= first.t { return (first.p, first.q) }
        let last = buf[buf.count - 1]
        if t >= last.t { return (last.p, last.q) }
        for i in 0 ..< (buf.count - 1) {
            let a = buf[i], b = buf[i + 1]
            if t >= a.t && t <= b.t {
                let u = Float((t - a.t) / max(1e-6, b.t - a.t))
                return (simd_mix(a.p, b.p, SIMD3<Float>(repeating: u)), simd_slerp(a.q, b.q, u))
            }
        }
        return (last.p, last.q)
    }
}
