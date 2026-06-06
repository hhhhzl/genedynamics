using UnityEngine;

// Meta Quest variant of OpenXRAnchorProvider. The registration logic is
// VENDOR-AGNOSTIC and lives entirely in the base — Pico 4/Neo, Magic Leap 2,
// Android XR, HTC Vive XR, Varjo, etc. can attach OpenXRAnchorProvider DIRECTLY,
// because the only per-vendor differences (OpenXR feature group, passthrough,
// build target) are Unity project settings, not code.
//
// This subclass exists so Quest scenes/prefabs and docs (ARCHITECTURE §4a) keep
// resolving, and as the template for a vendor-branded variant: if a vendor's head
// pose isn't on Camera.main (e.g. a custom rig), set HeadPoseSource in the
// Inspector or override Awake() here. With Quest it needs nothing extra.
namespace CorridorTwin
{
    public class QuestAnchorProvider : OpenXRAnchorProvider { }
}
