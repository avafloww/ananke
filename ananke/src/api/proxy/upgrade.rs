//! WebSocket (and other HTTP/1.1 upgrade) handshake proxying: dials a
//! dedicated upstream connection, forwards the handshake, and splices the
//! two upgraded sockets together for the lifetime of the session.

use std::{net::SocketAddr, time::Duration};

use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, Empty, StreamBody};
use hyper::{
    Request, Response, StatusCode,
    body::{Frame, Incoming},
    header,
};
use hyper_util::rt::TokioIo;
use tracing::{debug, warn};

use crate::{
    api::{
        errors::ApiErrorCode,
        proxy::{ProxyBody, WebSocketLifecycle, error_response},
    },
    tracking::inflight::InflightGuard,
};

/// How often to bump the per-service activity stamp while a WebSocket
/// session is open. Without this, the supervisor's idle-eviction loop
/// reads a stale stamp and SIGTERMs the service mid-session — there is
/// no traffic-based renewal because `copy_bidirectional` never touches
/// the activity table itself. 5 s is fast enough to keep any reasonable
/// `idle_timeout_ms` value (the smallest configurable is in the tens of
/// seconds) from racing the ticker, and the ping is just a single
/// `Mutex<Instant>` write so the overhead is negligible.
const WS_ACTIVITY_PING_INTERVAL: Duration = Duration::from_secs(5);

/// Proxy a single HTTP/1.1 upgrade request (typically WebSocket).
///
/// Opens a dedicated upstream connection (the pooled legacy client cannot
/// retain a half-upgraded socket), forwards the handshake, and — on a 101 —
/// splices both sides' upgraded I/O bidirectionally. The 101 response is
/// returned to the caller with the upstream's `Connection` and `Upgrade`
/// headers verbatim; this is the constraint aiohttp's WebSocket client
/// enforces (it rejects any `Connection` value other than literally
/// `upgrade`).
pub(crate) async fn handle_upgrade(
    mut req: Request<Incoming>,
    upstream_port: u16,
    peer: SocketAddr,
    ws_lifecycle: Option<WebSocketLifecycle>,
) -> Result<Response<ProxyBody>, Box<dyn std::error::Error + Send + Sync>> {
    // Detach the client-side upgrade handle before consuming the request.
    // It will resolve once hyper writes our 101 back and releases the TCP
    // socket for splicing.
    let client_upgrade = hyper::upgrade::on(&mut req);

    let (parts, _body) = req.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let uri = format!("http://127.0.0.1:{upstream_port}{path_and_query}").parse::<hyper::Uri>()?;

    let upstream_stream = match tokio::net::TcpStream::connect(("127.0.0.1", upstream_port)).await {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, peer = %peer, "upstream upgrade dial failed");
            return Ok(error_response(ApiErrorCode::UpstreamUnavailable {
                reason: e.to_string(),
            }));
        }
    };
    let upstream_io = TokioIo::new(upstream_stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(upstream_io).await?;
    // The connection driver future must be polled for `send_request` to
    // make progress AND for the upstream's upgrade to fire — without
    // `with_upgrades()` the conn returns the 101 but never hands over
    // the socket. We let the spawned task run unobserved; it terminates
    // naturally once the upgrade fires (or the connection errors out).
    tokio::spawn(async move {
        if let Err(e) = conn.with_upgrades().await {
            debug!(error = %e, "upstream upgrade connection driver ended");
        }
    });

    let mut upstream_builder = Request::builder().method(parts.method.clone()).uri(uri);
    upstream_builder = upstream_builder.header(header::HOST, format!("127.0.0.1:{upstream_port}"));
    for (k, v) in parts.headers.iter() {
        if k == header::HOST {
            continue;
        }
        upstream_builder = upstream_builder.header(k, v);
    }
    let empty_body: ProxyBody = Empty::<Bytes>::new()
        .map_err(|never| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed();
    let upstream_req = upstream_builder.body(empty_body)?;

    let mut upstream_resp = match sender.send_request(upstream_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, peer = %peer, "upstream upgrade request failed");
            return Ok(error_response(ApiErrorCode::UpstreamUnavailable {
                reason: e.to_string(),
            }));
        }
    };

    if upstream_resp.status() != StatusCode::SWITCHING_PROTOCOLS {
        // Upstream refused the upgrade. Pass the response back as a normal
        // streamed body and abandon the upgrade. Strip the connection
        // controls so hyper's encoder can pick the right framing for the
        // downstream socket.
        let (mut resp_parts, body) = upstream_resp.into_parts();
        resp_parts.headers.remove(header::CONNECTION);
        resp_parts.headers.remove("transfer-encoding");
        let stream = body.into_data_stream().map_ok(Frame::data);
        let boxed: ProxyBody = BodyExt::map_err(
            StreamBody::new(stream),
            |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
        )
        .boxed();
        return Ok(Response::from_parts(resp_parts, boxed));
    }

    let upstream_upgrade = hyper::upgrade::on(&mut upstream_resp);
    let upstream_version = upstream_resp.version();
    let upstream_headers = upstream_resp.headers().clone();

    // Mint the session-scoped lifecycle bookkeeping *before* spawning the
    // splice task so the guard is owned by the spawned future from the
    // very first poll — the drain pipeline races the counter and we
    // don't want a window where the count has dipped to zero. The
    // closure-scoped guard the caller installed is still alive at this
    // point, so the count is at least 2 until `handle_upgrade` returns
    // and the caller's guard drops back to 1.
    let session_guard = ws_lifecycle
        .as_ref()
        .map(|l| InflightGuard::new(l.inflight.clone()));
    let activity_ping = ws_lifecycle.map(|l| l.activity_ping);

    // Once hyper writes our 101 the client_upgrade future resolves; the
    // upstream_upgrade future resolves the moment the connection driver
    // sees the 101 on the wire. Joining both gives us paired I/O halves
    // we can splice byte-for-byte.
    tokio::spawn(async move {
        // Pin the guard into the task's stack so an early-return path
        // (e.g. handshake failure on `try_join!`) still drops it when
        // the task ends rather than at any earlier scope boundary.
        let _session_guard = session_guard;
        let (client_upg, upstream_upg) = match tokio::try_join!(client_upgrade, upstream_upgrade) {
            Ok(pair) => pair,
            Err(e) => {
                warn!(error = %e, peer = %peer, "websocket upgrade negotiation failed");
                return;
            }
        };
        let client_io = TokioIo::new(client_upg);
        let upstream_io = TokioIo::new(upstream_upg);
        let (mut client_r, mut client_w) = tokio::io::split(client_io);
        let (mut upstream_r, mut upstream_w) = tokio::io::split(upstream_io);

        // First-close-wins splice. `copy_bidirectional` waits for *both*
        // halves to EOF, which leaks the task if either peer half-
        // closes — the standard failure mode when a client process
        // exits without sending a WebSocket Close frame. select! drops
        // the loser the moment one direction returns, and dropping it
        // closes the underlying socket halves on the next poll.
        let splice = async {
            tokio::select! {
                r = tokio::io::copy(&mut client_r, &mut upstream_w) => r.map(|_| ()),
                r = tokio::io::copy(&mut upstream_r, &mut client_w) => r.map(|_| ()),
            }
        };
        if let Some(ping) = activity_ping {
            tokio::pin!(splice);
            let mut ticker = tokio::time::interval(WS_ACTIVITY_PING_INTERVAL);
            // `Interval` fires immediately on first tick; skip that so
            // the first real ping happens one interval in, by which
            // point the closure-scoped `_guard` from the caller has
            // dropped and the session-scoped guard is the sole holder.
            ticker.tick().await;
            loop {
                tokio::select! {
                    result = &mut splice => {
                        if let Err(e) = result {
                            debug!(error = %e, peer = %peer, "websocket splice ended");
                        }
                        break;
                    }
                    _ = ticker.tick() => {
                        ping();
                    }
                }
            }
        } else if let Err(e) = splice.await {
            debug!(error = %e, peer = %peer, "websocket splice ended");
        }
    });

    // Forward the upstream's response headers verbatim — most importantly,
    // `Connection: Upgrade` and `Upgrade: websocket`. aiohttp's WebSocket
    // client validates `Connection` with exact equality after lowercasing,
    // so any rewrite (adding `keep-alive`, replacing with the encoder's
    // default, reordering) breaks the handshake.
    let mut builder = Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .version(upstream_version);
    for (k, v) in upstream_headers.iter() {
        builder = builder.header(k, v);
    }
    let empty_body: ProxyBody = Empty::<Bytes>::new()
        .map_err(|never| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed();
    Ok(builder.body(empty_body)?)
}
