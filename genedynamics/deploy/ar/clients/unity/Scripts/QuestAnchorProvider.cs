using UnityEngine;

// Drives FrameRegistration.WorldOriginAnchor on Meta Quest (3 / 3S / Pro,
// passthrough MR via OpenXR + Meta XR SDK) — the Quest-side replacement for the
// phone path's ARTrackedImageManager-on-a-fiducial. Same swappable contract:
// it only writes a single Transform (WorldOriginAnchor); the renderer never
// knows which provider placed it (see clients/README "swap the registration
// source without touching the renderer").
//
//   ── Registration math ─────────────────────────────────────────────────────
// The robot and all obstacles live in the Vicon WORLD frame. The renderer draws
// each entity under WorldOriginAnchor at FrameRegistration.WorldToAnchorLocal(v)
// (the (-y,z,x) basis change). So the anchor must be the Unity-space pose of the
// (axis-mapped) Vicon origin. We solve it from ONE continuous correspondence —
// the headset itself, which we observe in BOTH frames:
//   * Vicon world  : a rigid Vicon marker cluster on the Quest → its world pose
//                    arrives over the net (fed in via SetHeadsetPoseWorld; the
//                    marker→head mount offset is applied UPSTREAM, producer-side,
//                    exactly like the robot base in producers/tracker_pose.py).
//   * Unity space  : Quest's own inside-out tracking → HeadPoseSource.transform.
// If the headset entity were rendered, it would sit at  anchor ∘ L  where
// L = (WorldToAnchorLocal(p), WorldToAnchorLocalRot(q)) is its anchor-local pose.
// Forcing that to equal the live Unity head pose H gives the closed form
//   anchor.rotation = H.rotation · L.rotation⁻¹
//   anchor.position = H.position − anchor.rotation · L.position
//
//   ── Why this beats a one-shot fiducial ─────────────────────────────────────
// The anchor pose is QUASI-STATIC (it only moves to correct Quest SLAM drift).
// Each fresh Vicon sample yields an estimate; we EMA-smooth it. That smoothing
// does double duty: it kills per-sample jitter AND continuously re-pins the world
// to Vicon ground truth, so the holograms don't drift away from the real robot
// over a long session (a fixed fiducial would). High-rate head motion is rendered
// entirely by Quest's low-latency on-device tracking — the Vicon stream's network
// latency only feeds the slow drift correction, so latency is harmless here.
//
//   ── No Meta/OpenXR dependency ──────────────────────────────────────────────
// This compiles against plain UnityEngine (head pose via a Camera). It therefore
// works under Meta XR SDK / Unity OpenXR / AR Foundation alike — the XR provider
// is a project-level plugin swap, not a code dependency. That keeps this a thin,
// runtime-agnostic adapter (ARCHITECTURE §4).
namespace CorridorTwin
{
    public class QuestAnchorProvider : MonoBehaviour
    {
        [Header("Wiring")]
        [Tooltip("The renderer's FrameRegistration; its WorldOriginAnchor is what we drive.")]
        public FrameRegistration Frame;

        [Tooltip("The XR head camera (Quest center-eye). Defaults to Camera.main. "
               + "Its transform must be in the SAME Unity space the anchor lives in.")]
        public Camera HeadPoseSource;

        [Header("Smoothing & drift correction")]
        [Tooltip("EMA blend toward each new solved anchor pose, per Vicon sample "
               + "(0 = frozen, 1 = snap). Low values = steadier holograms + slow "
               + "drift tracking. The first sample always snaps.")]
        [Range(0f, 1f)] public float Smoothing = 0.1f;

        [Tooltip("Once converged, stop updating the anchor (behaves like a one-shot "
               + "fiducial). Leave OFF on Quest to keep correcting SLAM drift; turn "
               + "ON if the headset Vicon stream is unreliable. Recenter() re-arms it.")]
        public bool LockWhenConverged = false;

        [Header("Convergence thresholds (pre-smoothing solve stability)")]
        public float ConvergePosMeters = 0.01f;
        public float ConvergeAngleDeg = 0.5f;
        [Tooltip("Consecutive stable solves required to declare convergence.")]
        public int ConvergeStableSamples = 10;

        public bool IsConverged { get; private set; }

        // ---- latest Vicon-world headset pose (set from any thread) -------------
        private readonly object _gate = new object();
        private Vector3 _viconPos;
        private Quaternion _viconQuat = Quaternion.identity;
        private bool _havePending;

        // ---- smoothed anchor state (main thread) -------------------------------
        private bool _haveAnchor;
        private Vector3 _anchorPos;
        private Quaternion _anchorRot = Quaternion.identity;
        private int _stableCount;

        // Inject the headset pose in the VICON WORLD frame. `quatWorld` is the
        // Vicon-world quaternion as a Unity Quaternion (x,y,z,w order, components
        // only — no basis change; this provider applies it). Transport-agnostic:
        // call it from a dedicated headset-pose WS subscriber, or have the producer
        // publish a `headset/<id>` entity (producers/tracker_pose.py) and forward
        // its pose here. Thread-safe; the Transform write happens in Update().
        public void SetHeadsetPoseWorld(Vector3 posWorld, Quaternion quatWorld)
        {
            lock (_gate) { _viconPos = posWorld; _viconQuat = quatWorld; _havePending = true; }
        }

        // Drop convergence + re-arm continuous updates (e.g. after the operator
        // re-seats the headset, or to recover from a bad lock).
        public void Recenter()
        {
            _haveAnchor = false;
            IsConverged = false;
            _stableCount = 0;
        }

        private void Awake()
        {
            if (HeadPoseSource == null) HeadPoseSource = Camera.main;
            if (Frame == null) Frame = GetComponent<FrameRegistration>();
        }

        private void Update()
        {
            if (LockWhenConverged && IsConverged) return;

            Vector3 vPos; Quaternion vQuat;
            lock (_gate)
            {
                if (!_havePending) return;
                vPos = _viconPos; vQuat = _viconQuat; _havePending = false;
            }

            if (Frame == null || Frame.WorldOriginAnchor == null || HeadPoseSource == null) return;

            // Headset's anchor-local pose L (what the renderer would place it at).
            Vector3 lPos = FrameRegistration.WorldToAnchorLocal(vPos);
            Quaternion lRot = FrameRegistration.WorldToAnchorLocalRot(vQuat);

            // Live Unity head pose H, then solve anchor ∘ L = H.
            Transform head = HeadPoseSource.transform;
            Quaternion solvedRot = head.rotation * Quaternion.Inverse(lRot);
            Vector3 solvedPos = head.position - solvedRot * lPos;

            UpdateConvergence(solvedPos, solvedRot);

            if (!_haveAnchor)
            {
                _anchorPos = solvedPos; _anchorRot = solvedRot; _haveAnchor = true; // first sample snaps
            }
            else
            {
                _anchorPos = Vector3.Lerp(_anchorPos, solvedPos, Smoothing);
                _anchorRot = Quaternion.Slerp(_anchorRot, solvedRot, Smoothing);
            }

            Frame.WorldOriginAnchor.SetPositionAndRotation(_anchorPos, _anchorRot);
        }

        // Count consecutive solves that barely move vs the smoothed anchor.
        private void UpdateConvergence(Vector3 solvedPos, Quaternion solvedRot)
        {
            if (!_haveAnchor) { _stableCount = 0; return; }
            float dp = Vector3.Distance(solvedPos, _anchorPos);
            float da = Quaternion.Angle(solvedRot, _anchorRot);
            if (dp <= ConvergePosMeters && da <= ConvergeAngleDeg)
            {
                if (++_stableCount >= ConvergeStableSamples) IsConverged = true;
            }
            else { _stableCount = 0; IsConverged = false; }
        }
    }
}
