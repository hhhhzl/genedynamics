import Foundation

// Subscribes to transport/world_ws.py over a URLSession WebSocket: a keyframe on
// connect, then deltas. Decodes the JSON contract and hands each WorldMessage to
// `onMessage` (called off the main thread — the renderer hops to main and stamps
// poses with its own clock for interpolation). Auto-reconnects.
public final class WorldClient {
    private let url: URL
    private var task: URLSessionWebSocketTask?
    private let session: URLSession
    public var onMessage: ((WorldMessage) -> Void)?

    public init(url: URL, session: URLSession = .shared) {
        self.url = url
        self.session = session
    }

    public func connect() {
        let t = session.webSocketTask(with: url)
        task = t
        t.resume()
        receive()
    }

    public func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success(let message):
                let data: Data?
                switch message {
                case .string(let s): data = s.data(using: .utf8)
                case .data(let d): data = d
                @unknown default: data = nil
                }
                if let d = data, let wm = try? JSONDecoder().decode(WorldMessage.self, from: d) {
                    self.onMessage?(wm)
                }
                self.receive()  // keep listening
            case .failure(let error):
                print("[WorldClient] \(error.localizedDescription); reconnecting in 1s")
                DispatchQueue.global().asyncAfter(deadline: .now() + 1) { [weak self] in
                    self?.connect()
                }
            }
        }
    }
}
