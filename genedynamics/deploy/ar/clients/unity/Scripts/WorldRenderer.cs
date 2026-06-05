using System.Collections.Generic;
using UnityEngine;

// Renders the interpolated entity world under FrameRegistration.WorldOriginAnchor.
// Each frame: drain WorldClient messages (apply keyframe/delta, stamping each
// pose with the Unity clock), then draw every entity at its INTERPOLATED pose at
// (now - InterpolationDelay) — so a ~30 Hz network stream renders smoothly at the
// display rate. World→Unity per conventions §2: position (-y,z,x); orientation by
// mapping the rotated basis vectors (handedness-safe, no quaternion guesswork).
namespace CorridorTwin
{
    public class WorldRenderer : MonoBehaviour
    {
        public WorldClient Client;
        public FrameRegistration Frame;

        [Tooltip("Render this far behind the latest state (s). ~1.5–2 network periods "
               + "hides jitter and keeps interpolation between two real samples.")]
        public double InterpolationDelay = 0.08;

        [Tooltip("Optional materials; occluder MUST be a depth-only / holdout material "
               + "(writes depth, draws no color) so the real robot occludes virtual obstacles.")]
        public Material DefaultMaterial;
        public Material OccluderMaterial;

        private readonly Dictionary<string, EntityState> _states = new Dictionary<string, EntityState>();
        private readonly Dictionary<string, GameObject> _objects = new Dictionary<string, GameObject>();

        private void Reset() { Frame = GetComponent<FrameRegistration>(); }

        private void Update()
        {
            ApplyInbox();
            double renderTime = Time.timeAsDouble - InterpolationDelay;
            foreach (var kv in _states)
                DrawEntity(kv.Key, kv.Value, renderTime);
        }

        // ---- apply keyframe / delta (main thread; stamp poses with Unity time) --
        private void ApplyInbox()
        {
            if (Client == null) return;
            while (Client.Inbox.TryDequeue(out var msg))
            {
                if (msg.IsKeyframe)
                {
                    var present = new HashSet<string>();
                    foreach (var e in msg.Entities) { Upsert(e); present.Add(e.Id); }
                    // keyframe is authoritative: drop anything not in it
                    var stale = new List<string>();
                    foreach (var id in _states.Keys) if (!present.Contains(id)) stale.Add(id);
                    foreach (var id in stale) Remove(id);
                }
                else
                {
                    foreach (var e in msg.Entities) Upsert(e);
                    foreach (var id in msg.RemovedIds) Remove(id);
                }
            }
        }

        private void Upsert(EntityMsg e)
        {
            if (!_states.TryGetValue(e.Id, out var st))
            {
                st = new EntityState { Id = e.Id };
                _states[e.Id] = st;
            }
            bool geomChanged = !st.GeomDirty && (st.Geom.kind != (e.HasGeom ? e.Geom.kind : st.Geom.kind));
            st.Type = e.Type; st.Frame = e.Frame; st.ColorRgba = e.ColorRgba; st.Meta = e.Meta; st.Rev = e.Rev;
            if (e.HasGeom) { st.Geom = e.Geom; }
            if (geomChanged) st.GeomDirty = true;
            st.PushPose(Time.timeAsDouble, e.PosWorld, e.QuatWorld);
        }

        private void Remove(string id)
        {
            _states.Remove(id);
            if (_objects.TryGetValue(id, out var go)) { Destroy(go); _objects.Remove(id); }
        }

        // ---- draw one entity at its interpolated pose ------------------------
        private void DrawEntity(string id, EntityState st, double renderTime)
        {
            if (!st.SampleAt(renderTime, out var pWorld, out var qWorld)) return;
            var go = ObtainObject(id, st);
            if (go == null) return;
            go.transform.localPosition = FrameRegistration.WorldToAnchorLocal(pWorld);
            go.transform.localRotation = WorldRotToUnity(qWorld);
        }

        private GameObject ObtainObject(string id, EntityState st)
        {
            if (_objects.TryGetValue(id, out var go) && go != null && !st.GeomDirty) return go;
            if (go != null) Destroy(go);

            var (prim, scale) = PrimitiveFor(st.Geom);
            go = GameObject.CreatePrimitive(prim);
            go.name = id;
            var col = go.GetComponent<Collider>();
            if (col != null) Destroy(col); // holograms don't collide
            if (Frame != null && Frame.WorldOriginAnchor != null)
                go.transform.SetParent(Frame.WorldOriginAnchor, false);
            go.transform.localScale = scale;

            var rend = go.GetComponent<Renderer>();
            if (st.Type == "occluder" && OccluderMaterial != null) rend.sharedMaterial = OccluderMaterial;
            else
            {
                if (DefaultMaterial != null) rend.material = DefaultMaterial;
                rend.material.color = RgbaToColor(st.ColorRgba);
            }
            _objects[id] = go;
            st.GeomDirty = false;
            return go;
        }

        // geom (world sizes) → Unity primitive + localScale (axis map: world x,y,z → unity z,x,y)
        private static (PrimitiveType, Vector3) PrimitiveFor(GeomInfo g)
        {
            switch (g.kind)
            {
                case "sphere":
                    return (PrimitiveType.Sphere, Vector3.one * (2f * g.radius));
                case "cylinder":
                    return (PrimitiveType.Cylinder, new Vector3(2f * g.radius, 0.5f * g.height, 2f * g.radius));
                case "capsule":
                    return (PrimitiveType.Capsule, new Vector3(2f * g.radius, 0.5f * g.height, 2f * g.radius));
                default: // box / wall (usd/urdf would load a mesh — see README)
                    return (PrimitiveType.Cube,
                            new Vector3(2f * g.halfExtents.y, 2f * g.halfExtents.z, 2f * g.halfExtents.x));
            }
        }

        // ---- world (RH, z-up) rotation → Unity (LH, y-up) -------------------
        // Map the rotated basis vectors through the position axis map and rebuild
        // with LookRotation (handedness-safe). `q` holds the WORLD quaternion.
        private static Quaternion WorldRotToUnity(Quaternion q)
        {
            Vector3 fwd = FrameRegistration.WorldToAnchorLocal(RotateWorldVec(q, new Vector3(1, 0, 0)));
            Vector3 up = FrameRegistration.WorldToAnchorLocal(RotateWorldVec(q, new Vector3(0, 0, 1)));
            if (fwd.sqrMagnitude < 1e-9f || up.sqrMagnitude < 1e-9f) return Quaternion.identity;
            return Quaternion.LookRotation(fwd, up);
        }

        // Rotate vector v by quaternion q (components frame-agnostic).
        private static Vector3 RotateWorldVec(Quaternion q, Vector3 v)
        {
            var u = new Vector3(q.x, q.y, q.z);
            return v + 2f * q.w * Vector3.Cross(u, v) + 2f * Vector3.Cross(u, Vector3.Cross(u, v));
        }

        private static Color RgbaToColor(uint rgba)
        {
            return new Color(((rgba >> 24) & 0xFF) / 255f, ((rgba >> 16) & 0xFF) / 255f,
                             ((rgba >> 8) & 0xFF) / 255f, (rgba & 0xFF) / 255f);
        }
    }
}
