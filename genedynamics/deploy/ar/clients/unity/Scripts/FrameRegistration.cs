using UnityEngine;

namespace CorridorTwin
{
    // Owns the two transforms needed to place scene-frame obstacles into the AR
    // view, registered to the Vicon world frame:
    //
    //   scene (x,y)  --T_world_scene-->  Vicon world (X,Y,Z)  --axis map-->  Unity
    //
    // The Unity placement is expressed RELATIVE TO a "world-origin anchor" that
    // sits at the Vicon origin in the AR session (set by image/marker tracking).
    // Parent your rendered obstacles under WorldOriginAnchor and use
    // SceneToAnchorLocal() for their localPosition / localRotation.
    public class FrameRegistration : MonoBehaviour
    {
        [Tooltip("A Transform placed at the Vicon WORLD origin in the AR session "
               + "(e.g. driven by ARTrackedImageManager on a surveyed fiducial, "
               + "or by a Vicon-streamed headset pose). Obstacles are parented here.")]
        public Transform WorldOriginAnchor;

        // T_world_scene (scene -> world), refreshed from each contract.
        private float _tx, _ty, _tyaw;

        public void SetTWorldScene(Se2 t)
        {
            if (t == null) return;
            _tx = t.x; _ty = t.y; _tyaw = t.yaw;
        }

        // scene (x,y) + height z  ->  Vicon world (X, Y, Z)
        public Vector3 SceneToWorld(float x, float y, float z)
        {
            float c = Mathf.Cos(_tyaw), s = Mathf.Sin(_tyaw);
            return new Vector3(_tx + c * x - s * y, _ty + s * x + c * y, z);
        }

        // Vicon world (X,Y,Z; x=fwd, y=left, z=up, right-handed)
        //   -> Unity local under the world-origin anchor (x=right, y=up, z=fwd, LH)
        //   unity = (-Y, Z, X)
        public static Vector3 WorldToAnchorLocal(Vector3 world)
        {
            return new Vector3(-world.y, world.z, world.x);
        }

        // scene (x,y,z) -> Unity local position under WorldOriginAnchor.
        public Vector3 SceneToAnchorLocal(float x, float y, float z)
        {
            return WorldToAnchorLocal(SceneToWorld(x, y, z));
        }

        // scene yaw (about world +z) -> Unity local rotation (about Unity +y).
        // World +z maps to Unity +y, and the RH->LH flip negates the angle.
        public Quaternion SceneYawToAnchorLocal(float sceneYawRad)
        {
            float worldYaw = sceneYawRad + _tyaw;
            return Quaternion.Euler(0f, -worldYaw * Mathf.Rad2Deg, 0f);
        }

        // Convenience: turn an anchor-local point into Unity world space for
        // gizmos / non-parented objects.
        public Vector3 AnchorLocalToUnityWorld(Vector3 local)
        {
            return WorldOriginAnchor != null ? WorldOriginAnchor.TransformPoint(local) : local;
        }
    }
}
