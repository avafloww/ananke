//! Talking to the server under measurement, and to the port it has to bind.
//!
//! A hand-rolled blocking client rather than a dependency: the harness makes
//! three kinds of call to one loopback address, and adding an async stack to a
//! sequential tool would cost a runtime for nothing.
//!
//! The port check is the subtle one. Stopping a server is not the same as the
//! port being free — the previous listener's socket can outlive it — and
//! ik_llama's server does not set `SO_REUSEADDR`, so it loses the bind and exits
//! instead of retrying. Downstream that reads as a load failure, which is how a
//! whole run of ik cells fails for a reason that has nothing to do with the
//! model. Worse, a leftover server that *keeps* the bind means every later cell
//! measures the same process at every point, which once invalidated an entire
//! sweep.

use std::{
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    os::fd::AsRawFd,
    time::Duration,
};

use serde::{Serialize, de::DeserializeOwned};

pub trait Http: Send + Sync {
    /// Whether a fresh server could bind the port right now.
    fn port_free(&self, port: u16) -> bool;
    /// Whether the server answers `/health` with a 200. It answers 503 while the
    /// model is still loading, so this is the readiness signal.
    fn healthy(&self, port: u16) -> bool;
    /// Send a request, returning the decoded body when there is one. The body
    /// carries the server's own token accounting, which is what ties a memory
    /// reading to the work that produced it.
    ///
    /// The JSON transport: callers that know their request and response shapes
    /// go through [`post_json`] instead. This stays `Value`-shaped because
    /// `Deps` holds `Http` as `Arc<dyn Http>`, and a generic method would make
    /// the trait non-object-safe.
    fn post(&self, port: u16, path: &str, body: &str, timeout: Duration) -> Option<String>;
}

/// Serialize `body`, `POST` it, and decode the reply as `Res` — the typed
/// wrapper over [`Http::post`]'s `Value` transport.
///
/// A free function rather than a default trait method: every caller holds
/// `Arc<dyn Http>`, and a generic method is not callable through a trait
/// object even when declared `where Self: Sized` (that bound only makes the
/// trait itself object-safe; it still excludes the method from `dyn Http`).
pub fn post_json<Req: Serialize, Res: DeserializeOwned>(
    http: &dyn Http,
    port: u16,
    path: &str,
    body: &Req,
    timeout: Duration,
) -> Option<Res> {
    let body = serde_json::to_string(body).ok()?;
    let reply = http.post(port, path, &body, timeout)?;
    serde_json::from_str(&reply).ok()
}

pub struct LoopbackHttp;

impl Http for LoopbackHttp {
    fn port_free(&self, port: u16) -> bool {
        // Deliberately a raw socket rather than `TcpListener::bind`, which sets
        // `SO_REUSEADDR` on Unix: that option is exactly what lets *this*
        // process bind a port still in `TIME_WAIT`, and the question here is
        // whether the *next* server can, without it.
        let Ok(socket) = nix::sys::socket::socket(
            nix::sys::socket::AddressFamily::Inet,
            nix::sys::socket::SockType::Stream,
            nix::sys::socket::SockFlag::empty(),
            None,
        ) else {
            return false;
        };
        let address = nix::sys::socket::SockaddrIn::new(127, 0, 0, 1, port);
        nix::sys::socket::bind(socket.as_raw_fd(), &address).is_ok()
    }

    fn healthy(&self, port: u16) -> bool {
        request(port, "GET", "/health", None, Duration::from_secs(5))
            .is_some_and(|response| response.status == 200)
    }

    fn post(&self, port: u16, path: &str, body: &str, timeout: Duration) -> Option<String> {
        // A failed probe still leaves the process measurable, so a refusal or a
        // timeout is `None` rather than an error that ends the cell.
        request(port, "POST", path, Some(body), timeout).map(|response| response.body)
    }
}

struct Response {
    status: u16,
    body: String,
}

fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout: Duration,
) -> Option<Response> {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let stream = TcpStream::connect_timeout(&address, Duration::from_secs(5)).ok()?;
    stream.set_read_timeout(Some(timeout)).ok()?;
    stream.set_write_timeout(Some(timeout)).ok()?;
    let mut stream = stream;

    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n");
    // `close` so the whole response ends at EOF: it removes any need to reason
    // about keep-alive framing for a client that makes one call per connection.
    head.push_str("Connection: close\r\n");
    if let Some(body) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).ok()?;
    if let Some(body) = body {
        stream.write_all(body.as_bytes()).ok()?;
    }
    stream.flush().ok()?;

    let mut raw = Vec::new();
    let read = stream.read_to_end(&mut raw);
    let _ = stream.shutdown(Shutdown::Both);
    // A read that timed out part-way still holds whatever arrived, and a
    // response whose head is complete is worth having.
    if read.is_err() && raw.is_empty() {
        return None;
    }
    parse_response(&String::from_utf8_lossy(&raw))
}

/// Split a response into its status and its body, decoding a chunked one.
///
/// Chunked is handled because llama.cpp's HTTP layer picks the framing, and a
/// body silently truncated at the first chunk header would read as a server that
/// returned no token accounting.
fn parse_response(raw: &str) -> Option<Response> {
    let (head, body) = raw.split_once("\r\n\r\n")?;
    let status = head
        .lines()
        .next()?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    let chunked = head.lines().any(|line| {
        line.to_ascii_lowercase().starts_with("transfer-encoding:") && line.contains("chunked")
    });
    let body = if chunked {
        dechunk(body)
    } else {
        body.to_owned()
    };
    Some(Response { status, body })
}

fn dechunk(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    while let Some((header, tail)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(header.split(';').next().unwrap_or("").trim(), 16);
        let Ok(size) = size else { break };
        if size == 0 || tail.len() < size {
            out.push_str(&tail[..size.min(tail.len())]);
            break;
        }
        out.push_str(&tail[..size]);
        rest = tail[size..].strip_prefix("\r\n").unwrap_or("");
    }
    out
}

/// A server that is whatever a test says: busy or free, loading for a scripted
/// number of polls, and replying with canned bodies.
///
/// Requests are recorded as the bodies they were sent as, which is what the
/// seam carries. A test that wants to assert on one deserialises it into the
/// type it expects the caller to have sent — a stronger check than indexing a
/// key, since it also proves the body parses as that type.
#[cfg(any(test, feature = "test-fakes"))]
pub struct FakeHttp {
    inner: parking_lot::Mutex<FakeHttpState>,
}

#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default)]
struct FakeHttpState {
    port_free: bool,
    /// How many `/health` polls come back unhealthy before the server answers.
    unhealthy_polls: u32,
    polls: u32,
    replies: std::collections::VecDeque<String>,
    requests: Vec<(String, String)>,
}

#[cfg(any(test, feature = "test-fakes"))]
impl Default for FakeHttp {
    /// Deliberately [`FakeHttp::new`] rather than a derive: the derived form would
    /// report the port as busy, which is the interesting case and so a poor default.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl FakeHttp {
    /// Free and immediately healthy, which is the uninteresting case every test
    /// starts from.
    pub fn new() -> Self {
        Self {
            inner: parking_lot::Mutex::new(FakeHttpState {
                port_free: true,
                ..FakeHttpState::default()
            }),
        }
    }

    pub fn port_busy(self) -> Self {
        self.inner.lock().port_free = false;
        self
    }

    pub fn loading_for(self, polls: u32) -> Self {
        self.inner.lock().unhealthy_polls = polls;
        self
    }

    /// Never healthy: the load-timeout path.
    pub fn never_healthy(self) -> Self {
        self.inner.lock().unhealthy_polls = u32::MAX;
        self
    }

    /// Script a reply. Takes anything serialisable so a test can hand over the
    /// response type the caller will parse, rather than spelling its JSON.
    pub fn with_reply(self, reply: impl serde::Serialize) -> Self {
        let body = serde_json::to_string(&reply).expect("the scripted reply serialises");
        self.inner.lock().replies.push_back(body);
        self
    }

    /// Every request made, as the body it was sent as.
    pub fn requests(&self) -> Vec<(String, String)> {
        self.inner.lock().requests.clone()
    }

    /// The nth request, parsed as the type the caller should have sent.
    pub fn request_as<T: serde::de::DeserializeOwned>(&self, index: usize) -> Option<T> {
        let state = self.inner.lock();
        let (_, body) = state.requests.get(index)?;
        serde_json::from_str(body).ok()
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl Http for FakeHttp {
    fn port_free(&self, _port: u16) -> bool {
        self.inner.lock().port_free
    }

    fn healthy(&self, _port: u16) -> bool {
        let mut state = self.inner.lock();
        state.polls += 1;
        state.polls > state.unhealthy_polls
    }

    fn post(&self, _port: u16, path: &str, body: &str, _timeout: Duration) -> Option<String> {
        let mut state = self.inner.lock();
        state.requests.push((path.to_owned(), body.to_owned()));
        // The last scripted reply repeats, so a soak of forty turns does not
        // need forty entries.
        match state.replies.len() {
            0 => None,
            1 => state.replies.front().cloned(),
            _ => state.replies.pop_front(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunked_body_is_reassembled() {
        let raw = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n{\"a\"\r\n4\r\n:1}\n\r\n0\r\n\r\n";
        let response = parse_response(raw).expect("the head is complete");
        assert_eq!(response.status, 200);
        assert_eq!(response.body, "{\"a\":1}\n");
    }

    #[test]
    fn a_content_length_body_is_taken_verbatim() {
        let raw = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 21\r\n\r\n{\"status\":\"loading\"}\n";
        let response = parse_response(raw).expect("the head is complete");
        assert_eq!(response.status, 503);
        assert_eq!(response.body, "{\"status\":\"loading\"}\n");
    }
}
