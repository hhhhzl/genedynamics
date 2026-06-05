using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;   // com.unity.nuget.newtonsoft-json
using UnityEngine;

// Subscribes to the generic WorldState stream (transport/world_ws.py): a keyframe
// on connect, then deltas. Parses the JSON contract (schema/world.fbs shape) off
// the main thread and queues WorldMessages; WorldRenderer drains + applies them
// on the main thread (where the Unity clock stamps each pose for interpolation).
//
// JSON keeps this dependency-light. For the zero-copy hot path, swap the parse to
// the flatc-generated C# (schema/generated/csharp/twin) + Google.FlatBuffers and
// point world_ws at flatbuffers_codec.encode.
namespace CorridorTwin
{
    public struct EntityMsg
    {
        public string Id, Type, Frame, Meta;
        public uint Rev, ColorRgba;
        public bool HasGeom;
        public GeomInfo Geom;
        public Vector3 PosWorld;     // contract pose.p (world)
        public Quaternion QuatWorld; // contract pose.q (world; components only)
    }

    public class WorldMessage
    {
        public bool IsKeyframe;
        public readonly List<EntityMsg> Entities = new List<EntityMsg>();
        public readonly List<string> RemovedIds = new List<string>();
    }

    public class WorldClient : MonoBehaviour
    {
        [Tooltip("world_ws endpoint, e.g. ws://192.168.1.50:8766/")]
        public string Url = "ws://localhost:8766/";

        public readonly ConcurrentQueue<WorldMessage> Inbox = new ConcurrentQueue<WorldMessage>();

        private CancellationTokenSource _cts;
        private ClientWebSocket _ws;

        private async void OnEnable() { _cts = new CancellationTokenSource(); await Loop(_cts.Token); }
        private void OnDisable() { _cts?.Cancel(); try { _ws?.Abort(); } catch { } _ws?.Dispose(); _ws = null; }

        private async Task Loop(CancellationToken ct)
        {
            var buf = new byte[1 << 18];
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    _ws = new ClientWebSocket();
                    await _ws.ConnectAsync(new Uri(Url), ct);
                    Debug.Log($"[WorldClient] connected {Url}");
                    var sb = new StringBuilder();
                    while (_ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
                    {
                        sb.Clear();
                        WebSocketReceiveResult r;
                        do
                        {
                            r = await _ws.ReceiveAsync(new ArraySegment<byte>(buf), ct);
                            if (r.MessageType == WebSocketMessageType.Close) break;
                            sb.Append(Encoding.UTF8.GetString(buf, 0, r.Count));
                        } while (!r.EndOfMessage);
                        if (sb.Length == 0) continue;
                        try { Inbox.Enqueue(Parse(sb.ToString())); }
                        catch (Exception e) { Debug.LogWarning($"[WorldClient] parse: {e.Message}"); }
                    }
                }
                catch (OperationCanceledException) { break; }
                catch (Exception e) { Debug.LogWarning($"[WorldClient] {e.Message}; retry 1s"); }
                try { _ws?.Dispose(); } catch { }
                _ws = null;
                if (!ct.IsCancellationRequested) await Task.Delay(1000, ct);
            }
        }

        private static WorldMessage Parse(string json)
        {
            var o = JObject.Parse(json);
            var msg = new WorldMessage { IsKeyframe = (bool?)o["is_keyframe"] ?? true };
            if (o["entities"] is JArray ents)
            {
                foreach (var je in ents)
                {
                    var pp = (JArray)je["pose"]["p"];
                    var pq = (JArray)je["pose"]["q"];
                    var e = new EntityMsg
                    {
                        Id = (string)je["id"],
                        Type = (string)je["type"],
                        Frame = (string)(je["frame"] ?? "world"),
                        Rev = (uint)(je["rev"] ?? 0),
                        ColorRgba = (uint)(je["color_rgba"] ?? 0xFFFFFFFF),
                        Meta = (string)(je["meta"] ?? ""),
                        PosWorld = new Vector3((float)pp[0], (float)pp[1], (float)pp[2]),
                        // contract q = (w,x,y,z); Unity ctor = (x,y,z,w)
                        QuatWorld = new Quaternion((float)pq[1], (float)pq[2], (float)pq[3], (float)pq[0]),
                    };
                    var g = je["geom"];
                    if (g != null && g.Type != JTokenType.Null)
                    {
                        var gi = new GeomInfo
                        {
                            kind = (string)g["kind"],
                            radius = (float)(g["radius"] ?? 0f),
                            height = (float)(g["height"] ?? 0f),
                            assetUri = (string)(g["asset_uri"] ?? ""),
                        };
                        if (g["half_extents"] is JArray he)
                            gi.halfExtents = new Vector3((float)he[0], (float)he[1], (float)he[2]);
                        e.Geom = gi;
                        e.HasGeom = true;
                    }
                    msg.Entities.Add(e);
                }
            }
            if (o["removed_ids"] is JArray rem)
                foreach (var r in rem) msg.RemovedIds.Add((string)r);
            return msg;
        }
    }
}
