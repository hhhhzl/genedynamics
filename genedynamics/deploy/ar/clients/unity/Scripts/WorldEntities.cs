using System;
using System.Collections.Generic;
using UnityEngine;

// Entity-model client state for the generic WorldSnapshot contract (P1/P2),
// superseding the M1 corridor-contract SceneContract. One EntityState per
// entity id, each holding a small timestamped pose buffer for client-side
// INTERPOLATION (render at 90 Hz from ~30 Hz network state; see WorldRenderer).
//
// Poses are stored in the WORLD frame as received; WorldRenderer converts to
// Unity at draw time (conventions §2: unity = (-y, z, x)).
namespace CorridorTwin
{
    public struct GeomInfo
    {
        public string kind;          // box|sphere|cylinder|capsule|usd|urdf|path
        public Vector3 halfExtents;  // box (world x,y,z half-sizes)
        public float radius;         // sphere/cylinder
        public float height;         // cylinder
        public string assetUri;      // usd/urdf
    }

    public class EntityState
    {
        public string Id;
        public string Type;          // obstacle|wall|robot|arm_link|goal|path|occluder
        public string Frame = "world";
        public uint ColorRgba = 0xFFFFFFFF;
        public string Meta = "";
        public GeomInfo Geom;
        public uint Rev;
        public bool GeomDirty = true; // renderer rebuilds mesh/material when set

        // ---- interpolation buffer (world-frame samples, monotonic stamps) ----
        private struct Sample { public double t; public Vector3 p; public Quaternion q; }
        private const int Capacity = 8;
        private readonly List<Sample> _buf = new List<Sample>(Capacity);

        public void PushPose(double stampSec, Vector3 worldPos, Quaternion worldQuat)
        {
            // Drop out-of-order / duplicate stamps (keep the buffer monotonic).
            if (_buf.Count > 0 && stampSec <= _buf[_buf.Count - 1].t) return;
            _buf.Add(new Sample { t = stampSec, p = worldPos, q = worldQuat });
            if (_buf.Count > Capacity) _buf.RemoveAt(0);
        }

        // Interpolated world pose at renderTime (= now - interpolationDelay).
        // Holds the endpoints outside the buffer span (no extrapolation — avoids
        // overshoot on packet gaps). Returns false until the first sample.
        public bool SampleAt(double renderTime, out Vector3 pos, out Quaternion rot)
        {
            int n = _buf.Count;
            if (n == 0) { pos = Vector3.zero; rot = Quaternion.identity; return false; }
            if (n == 1 || renderTime <= _buf[0].t) { pos = _buf[0].p; rot = _buf[0].q; return true; }
            var last = _buf[n - 1];
            if (renderTime >= last.t) { pos = last.p; rot = last.q; return true; }
            for (int i = 0; i < n - 1; i++)
            {
                var a = _buf[i];
                var b = _buf[i + 1];
                if (renderTime >= a.t && renderTime <= b.t)
                {
                    float u = (float)((renderTime - a.t) / Math.Max(1e-6, b.t - a.t));
                    pos = Vector3.Lerp(a.p, b.p, u);
                    rot = Quaternion.Slerp(a.q, b.q, u);
                    return true;
                }
            }
            pos = last.p; rot = last.q; return true;
        }

        public double LatestStamp => _buf.Count > 0 ? _buf[_buf.Count - 1].t : 0.0;
    }
}
