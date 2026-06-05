# WebXR client (Android Chrome — zero install)

`index.html` is a self-contained three.js + WebXR viewer of the WorldSnapshot
stream (`transport/world_ws.py`): WebSocket → entities → meshes, with client-side
interpolation and an occluder holdout (so the real robot occludes virtual
obstacles). No app store, no Mac — just a URL in the browser.

## Support
- **Android Chrome:** full `immersive-ar` (works).
- **iOS Safari:** no immersive AR WebXR (Apple) — use the Unity/RealityKit clients.
- **visionOS Safari:** partial WebXR (visionOS 2+).

## Run
WebXR requires a **secure context (HTTPS)** unless served from `localhost`.
On a server reachable from the tablet over LAN you must use HTTPS:

```bash
# from genedynamics/deploy/ar/clients/web/, with a self-signed cert (one-time):
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=twin"
python -c "import http.server,ssl,functools; \
h=functools.partial(http.server.SimpleHTTPRequestHandler); \
s=http.server.HTTPServer(('0.0.0.0',8443),h); \
import ssl; ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain('cert.pem','key.pem'); \
s.socket=ctx.wrap_socket(s.socket,server_side=True); s.serve_forever()"
# tablet Chrome → https://<TWIN_IP>:8443/  (accept the self-signed warning)
```
(Or host the file anywhere with HTTPS, e.g. a tunnel.)

The page connects to `ws://<page-host>:8766/` by default (edit `WS_URL` in
`index.html` if world_ws is elsewhere). Note: an HTTPS page connecting to a
plain `ws://` is blocked by mixed-content — run world_ws behind `wss://`
(a TLS terminator) for production, or serve both over the same scheme.

## Registration
The starter uses **tap-to-place** (WebXR hit-test): tap the floor to drop the
world origin. That is NOT Vicon-accurate. For accurate registration, place the
origin from a fiducial (a WebXR image-tracking lib / marker library) using the
fiducial's surveyed Vicon pose (instruction.md A6), or fall back to the
Unity/RealityKit clients whose Vicon registration is straightforward.

## Notes
- Axis map world→three: `(-y, z, -x)` (RH y-up −z forward; conventions §2).
- JSON wire form (simplest). For the zero-copy path, the generated TS
  FlatBuffers under `../../schema/generated/ts` can replace the JSON parse.
