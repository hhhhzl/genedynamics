using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace CorridorTwin
{
    // Subscribes to the scene_server WebSocket and raises OnContract on the main
    // thread whenever a new scene revision arrives. ClientWebSocket works on
    // iOS / Android / visionOS native players (not WebGL).
    public class SceneClient : MonoBehaviour
    {
        [Tooltip("scene_server endpoint, e.g. ws://192.168.1.50:8765/ws")]
        public string Url = "ws://localhost:8765/ws";

        [Tooltip("Only raise OnContract when the scene revision changes.")]
        public bool OnlyOnRevisionChange = true;

        public event Action<SceneContractMsg> OnContract;

        private ClientWebSocket _ws;
        private CancellationTokenSource _cts;
        private readonly object _gate = new object();
        private SceneContractMsg _pending;
        private int _lastRevision = -1;

        private async void OnEnable()
        {
            _cts = new CancellationTokenSource();
            await ConnectLoop(_cts.Token);
        }

        private void OnDisable()
        {
            _cts?.Cancel();
            try { _ws?.Abort(); } catch { /* ignore */ }
            _ws?.Dispose();
            _ws = null;
        }

        private void Update()
        {
            // Hand the latest parsed contract to listeners on the main thread.
            SceneContractMsg msg = null;
            lock (_gate) { if (_pending != null) { msg = _pending; _pending = null; } }
            if (msg == null) return;
            if (OnlyOnRevisionChange && msg.revision == _lastRevision) return;
            _lastRevision = msg.revision;
            OnContract?.Invoke(msg);
        }

        private async Task ConnectLoop(CancellationToken ct)
        {
            var buf = new byte[1 << 16];
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    _ws = new ClientWebSocket();
                    await _ws.ConnectAsync(new Uri(Url), ct);
                    Debug.Log($"[SceneClient] connected {Url}");
                    var sb = new StringBuilder();
                    while (_ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
                    {
                        sb.Clear();
                        WebSocketReceiveResult r;
                        do
                        {
                            r = await _ws.ReceiveAsync(new ArraySegment<byte>(buf), ct);
                            if (r.MessageType == WebSocketMessageType.Close)
                            {
                                await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "", ct);
                                break;
                            }
                            sb.Append(Encoding.UTF8.GetString(buf, 0, r.Count));
                        } while (!r.EndOfMessage);

                        if (sb.Length == 0) continue;
                        try
                        {
                            var msg = SceneContractMsg.Parse(sb.ToString());
                            if (msg != null) lock (_gate) { _pending = msg; }
                        }
                        catch (Exception e) { Debug.LogWarning($"[SceneClient] parse: {e.Message}"); }
                    }
                }
                catch (OperationCanceledException) { break; }
                catch (Exception e) { Debug.LogWarning($"[SceneClient] {e.Message}; retrying in 1s"); }

                try { _ws?.Dispose(); } catch { /* ignore */ }
                _ws = null;
                if (!ct.IsCancellationRequested) await Task.Delay(1000, ct);
            }
        }

        // Optional: send a control message, e.g. SendControl("{\"cmd\":\"set_preset\",\"preset\":\"zone_a\"}")
        public async void SendControl(string json)
        {
            if (_ws == null || _ws.State != WebSocketState.Open) return;
            var bytes = Encoding.UTF8.GetBytes(json);
            await _ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, _cts.Token);
        }
    }
}
