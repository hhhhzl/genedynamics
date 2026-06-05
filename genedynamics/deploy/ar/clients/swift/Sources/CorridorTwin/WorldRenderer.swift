import Foundation
import simd
import RealityKit
#if canImport(UIKit)
import UIKit
#endif

// Renders the interpolated entity world under `worldAnchor` (placed at the Vicon
// world origin by registration). Call `update(now:)` every frame (e.g. from a
// SceneEvents.Update subscription): it drains queued WorldMessages (keyframe /
// delta), stamps each pose with `now`, then draws every entity at its
// INTERPOLATED pose at (now - interpolationDelay).
//
// World→RealityKit per conventions §2 (RH y-up, −z forward): rk = (-y, z, -x).
// Occluders get an OcclusionMaterial so the REAL robot occludes virtual obstacles.
public final class WorldRenderer {
    public let worldAnchor: AnchorEntity
    public var interpolationDelay: Double = 0.08

    private var states: [String: EntityState] = [:]
    private var models: [String: ModelEntity] = [:]
    private var pending: [WorldMessage] = []
    private let lock = NSLock()

    public init(worldAnchor: AnchorEntity) { self.worldAnchor = worldAnchor }

    /// Thread-safe handoff from WorldClient.onMessage (background).
    public func enqueue(_ m: WorldMessage) {
        lock.lock(); pending.append(m); lock.unlock()
    }

    public func update(now: Double) {
        drain(now: now)
        let t = now - interpolationDelay
        for (id, st) in states {
            guard let (pW, qW) = st.sample(at: t), let m = obtain(id: id, st: st) else { continue }
            m.position = Self.worldVecToRK(pW)
            m.orientation = Self.worldRotToRK(qW)
        }
    }

    // ---- apply keyframe / delta (main thread; stamp with `now`) -------------
    private func drain(now: Double) {
        lock.lock(); let msgs = pending; pending.removeAll(); lock.unlock()
        for msg in msgs {
            if msg.is_keyframe {
                var present = Set<String>()
                for e in msg.entities { upsert(e, now: now); present.insert(e.id) }
                for id in Array(states.keys) where !present.contains(id) { remove(id) }
            } else {
                for e in msg.entities { upsert(e, now: now) }
                for id in msg.removed_ids ?? [] { remove(id) }
            }
        }
    }

    private func upsert(_ e: EntityMsg, now: Double) {
        let st = states[e.id] ?? EntityState(id: e.id)
        if states[e.id] == nil { states[e.id] = st }
        if let g = e.geom, g.kind != st.geom?.kind { st.geomDirty = true }
        st.type = e.type
        st.colorRgba = e.color_rgba ?? 0xFFFFFFFF
        st.rev = e.rev ?? 0
        if let g = e.geom { st.geom = g }
        let p = SIMD3<Float>(e.pose.p[0], e.pose.p[1], e.pose.p[2])
        // contract q = (w,x,y,z); simd_quatf(ix,iy,iz,r)
        let q = simd_quatf(ix: e.pose.q[1], iy: e.pose.q[2], iz: e.pose.q[3], r: e.pose.q[0])
        st.push(now, p, q)
    }

    private func remove(_ id: String) {
        states[id] = nil
        if let m = models[id] { m.removeFromParent(); models[id] = nil }
    }

    // ---- mesh + material ----------------------------------------------------
    private func obtain(id: String, st: EntityState) -> ModelEntity? {
        if let m = models[id], !st.geomDirty { return m }
        models[id]?.removeFromParent()
        let mesh = Self.mesh(for: st.geom)
        let material: RealityKit.Material = (st.type == "occluder")
            ? OcclusionMaterial()
            : SimpleMaterial(color: Self.color(st.colorRgba), roughness: 0.7, isMetallic: false)
        let m = ModelEntity(mesh: mesh, materials: [material])
        m.name = id
        worldAnchor.addChild(m)
        models[id] = m
        st.geomDirty = false
        return m
    }

    private static func mesh(for g: GeomMsg?) -> MeshResource {
        guard let g = g else { return .generateBox(size: 0.1) }
        switch g.kind {
        case "sphere":
            return .generateSphere(radius: g.radius ?? 0.05)
        case "cylinder", "capsule":
            // generateCylinder requires recent RealityKit (visionOS 2 / iOS 18);
            // fall back to a box if targeting older runtimes.
            return .generateCylinder(height: g.height ?? 0.1, radius: g.radius ?? 0.05)
        default: // box / wall (usd/urdf would load a mesh — see README)
            let he = g.half_extents ?? [0.05, 0.05, 0.05]
            // world (x,y,z) half-sizes → RK full sizes (x=2*wy, y=2*wz, z=2*wx)
            return .generateBox(size: SIMD3<Float>(2 * he[1], 2 * he[2], 2 * he[0]))
        }
    }

    // ---- world (RH, z-up) → RealityKit (RH, y-up, −z forward) ---------------
    static func worldVecToRK(_ w: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(-w.y, w.z, -w.x)
    }

    static func worldRotToRK(_ q: simd_quatf) -> simd_quatf {
        let fwd = worldVecToRK(q.act(SIMD3<Float>(1, 0, 0)))   // world +x
        let up  = worldVecToRK(q.act(SIMD3<Float>(0, 0, 1)))   // world +z
        return lookRotation(forward: fwd, up: up)
    }

    /// Orientation whose −z aligns with `forward` and +y with `up` (RealityKit).
    static func lookRotation(forward f: SIMD3<Float>, up u: SIMD3<Float>) -> simd_quatf {
        let z = -simd_normalize(f)                 // RealityKit faces −z
        let x = simd_normalize(simd_cross(u, z))
        let y = simd_cross(z, x)
        return simd_quatf(simd_float3x3(x, y, z))
    }

    private static func color(_ rgba: UInt32) -> RealityKit.Material.Color {
        let r = CGFloat((rgba >> 24) & 0xFF) / 255.0
        let g = CGFloat((rgba >> 16) & 0xFF) / 255.0
        let b = CGFloat((rgba >> 8) & 0xFF) / 255.0
        let a = CGFloat(rgba & 0xFF) / 255.0
        #if canImport(UIKit)
        return UIColor(red: r, green: g, blue: b, alpha: a)
        #else
        return .init(red: r, green: g, blue: b, alpha: a)
        #endif
    }
}
