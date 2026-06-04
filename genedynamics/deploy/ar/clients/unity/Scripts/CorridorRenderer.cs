using System.Collections.Generic;
using UnityEngine;

namespace CorridorTwin
{
    // Builds/updates holographic obstacles + corridor walls from each contract,
    // parented under FrameRegistration.WorldOriginAnchor so they stay locked to
    // the Vicon world frame. Reuses GameObjects keyed by name to avoid churn.
    [RequireComponent(typeof(FrameRegistration))]
    public class CorridorRenderer : MonoBehaviour
    {
        public SceneClient Client;
        public FrameRegistration Frame;

        [Tooltip("Optional material for obstacles (semi-transparent looks best in AR).")]
        public Material ObstacleMaterial;
        [Tooltip("Optional material for the corridor walls.")]
        public Material WallMaterial;

        private readonly Dictionary<string, GameObject> _objects = new Dictionary<string, GameObject>();
        private readonly HashSet<string> _seen = new HashSet<string>();

        private void Reset() { Frame = GetComponent<FrameRegistration>(); }

        private void OnEnable()
        {
            if (Frame == null) Frame = GetComponent<FrameRegistration>();
            if (Client != null) Client.OnContract += Render;
        }

        private void OnDisable()
        {
            if (Client != null) Client.OnContract -= Render;
        }

        public void Render(SceneContractMsg c)
        {
            if (c == null || Frame == null) return;
            Frame.SetTWorldScene(c.T_world_scene);
            _seen.Clear();

            // Corridor walls (two long thin boxes at y = wall_y_min / wall_y_max).
            if (c.corridor != null)
            {
                float L = c.corridor.corridor_length;
                float wlo = c.corridor.wall_y_min, whi = c.corridor.wall_y_max;
                PlaceBox("__wall_lo", L * 0.5f, wlo, 1.0f, L, 0.02f, 2.0f, true);
                PlaceBox("__wall_hi", L * 0.5f, whi, 1.0f, L, 0.02f, 2.0f, true);
            }

            // Obstacles.
            if (c.obstacles != null)
            {
                foreach (var o in c.obstacles)
                {
                    if (o == null || string.IsNullOrEmpty(o.name)) continue;
                    float zc = 0.5f * (o.z_min + o.z_max);
                    float zh = Mathf.Max(0.02f, o.z_max - o.z_min);
                    switch (o.shape)
                    {
                        case "sphere":
                            PlaceSphere(o.name, o.cx, o.cy, zc, o.radius);
                            break;
                        case "qc": // quarter-circle taper — approximate with a cylinder
                            PlaceCylinder(o.name, o.cx, o.cy, zc, o.radius, zh);
                            break;
                        default: // "box"
                            PlaceBox(o.name,
                                0.5f * (o.x_min + o.x_max), 0.5f * (o.y_min + o.y_max), zc,
                                Mathf.Max(0.02f, o.x_max - o.x_min),
                                Mathf.Max(0.02f, o.y_max - o.y_min), zh, false);
                            break;
                    }
                }
            }

            // Drop obstacles that vanished from the scene.
            var stale = new List<string>();
            foreach (var kv in _objects) if (!_seen.Contains(kv.Key)) stale.Add(kv.Key);
            foreach (var k in stale) { Destroy(_objects[k]); _objects.Remove(k); }
        }

        // ---- primitive helpers (scene-frame inputs) --------------------------
        private GameObject Obtain(string name, PrimitiveType type, bool wall)
        {
            _seen.Add(name);
            if (_objects.TryGetValue(name, out var go) && go != null) return go;
            go = GameObject.CreatePrimitive(type);
            go.name = name;
            var col = go.GetComponent<Collider>();
            if (col != null) Destroy(col); // holograms shouldn't collide
            if (Frame.WorldOriginAnchor != null) go.transform.SetParent(Frame.WorldOriginAnchor, false);
            var mat = wall ? WallMaterial : ObstacleMaterial;
            if (mat != null) go.GetComponent<Renderer>().sharedMaterial = mat;
            _objects[name] = go;
            return go;
        }

        // Box: scene size (sx, sy, sz) -> Unity localScale (sy, sz, sx); the scene
        // frame's yaw becomes the object's local rotation.
        private void PlaceBox(string name, float cx, float cy, float cz,
                              float sx, float sy, float sz, bool wall)
        {
            var go = Obtain(name, PrimitiveType.Cube, wall);
            go.transform.localPosition = Frame.SceneToAnchorLocal(cx, cy, cz);
            go.transform.localRotation = Frame.SceneYawToAnchorLocal(0f);
            go.transform.localScale = new Vector3(sy, sz, sx);
        }

        private void PlaceSphere(string name, float cx, float cy, float cz, float r)
        {
            var go = Obtain(name, PrimitiveType.Sphere, false);
            go.transform.localPosition = Frame.SceneToAnchorLocal(cx, cy, cz);
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one * (2f * r);
        }

        // Unity cylinder: 1 unit diameter, 2 units tall along local +y (== world up here).
        private void PlaceCylinder(string name, float cx, float cy, float cz, float r, float height)
        {
            var go = Obtain(name, PrimitiveType.Cylinder, false);
            go.transform.localPosition = Frame.SceneToAnchorLocal(cx, cy, cz);
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = new Vector3(2f * r, height * 0.5f, 2f * r);
        }
    }
}
