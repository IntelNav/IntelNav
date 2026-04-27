//! Shaper smoke tests. We boot a tokio echo server on a free port,
//! a [`Shaper`] in front of it on another free port, and measure
//! that the round-trip delay matches what we configured. The numbers
//! aren't exact (token bucket + scheduler jitter add a few ms) so
//! assertions use generous bounds.

use std::time::Duration;

use intelnav_netsim::{LinkParams, Shaper, ShaperConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Bind to :0, return the picked address. Drops the listener — caller
/// re-binds. Race-prone but fine for a test that immediately reuses.
async fn pick_port() -> std::net::SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    drop(l);
    a
}

async fn spawn_echo() -> std::net::SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = match l.accept().await { Ok(v) => v, Err(_) => break };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                loop {
                    let n = match s.read(&mut buf).await { Ok(0) | Err(_) => break, Ok(n) => n };
                    if s.write_all(&buf[..n]).await.is_err() { break; }
                }
            });
        }
    });
    addr
}

#[tokio::test]
async fn delay_is_applied_per_direction() {
    let echo = spawn_echo().await;
    let listen = pick_port().await;
    let cfg = ShaperConfig {
        upstream: echo,
        forward:  LinkParams { delay_ms: 30.0, ..Default::default() },
        reverse:  LinkParams { delay_ms: 30.0, ..Default::default() },
        label:    "test".into(),
    };
    let shaper = Shaper::new(cfg);
    let s2 = shaper.clone();
    tokio::spawn(async move { let _ = s2.serve(listen).await; });

    // Give the listener a beat to come up.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut c = TcpStream::connect(listen).await.unwrap();
    let t0 = std::time::Instant::now();
    c.write_all(b"hello").await.unwrap();
    let mut buf = [0u8; 5];
    c.read_exact(&mut buf).await.unwrap();
    let rtt = t0.elapsed().as_millis() as u64;

    assert_eq!(&buf, b"hello");
    // Configured 30ms each way + scheduler/token-bucket noise.
    assert!(rtt >= 55, "rtt {rtt}ms is below the configured floor (~60ms)");
    assert!(rtt < 200, "rtt {rtt}ms is way above configured");

    // Snapshot picks up the bytes counters.
    let snap = shaper.snapshot().await;
    assert!(snap.forward_stats.bytes >= 5, "fwd bytes={}", snap.forward_stats.bytes);
    assert!(snap.reverse_stats.bytes >= 5, "rev bytes={}", snap.reverse_stats.bytes);
}

#[tokio::test]
async fn live_param_change_takes_effect() {
    let echo = spawn_echo().await;
    let listen = pick_port().await;
    let shaper = Shaper::new(ShaperConfig {
        upstream: echo,
        forward:  LinkParams::default(),
        reverse:  LinkParams::default(),
        label:    String::new(),
    });
    let s2 = shaper.clone();
    tokio::spawn(async move { let _ = s2.serve(listen).await; });
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Cold connection: no delay configured → fast.
    let mut c = TcpStream::connect(listen).await.unwrap();
    let t0 = std::time::Instant::now();
    c.write_all(b"a").await.unwrap();
    let mut buf = [0u8; 1];
    c.read_exact(&mut buf).await.unwrap();
    let cold = t0.elapsed().as_millis() as u64;
    assert!(cold < 30, "cold rtt {cold}ms should be near-zero");

    // Live-tune the forward leg to 50ms one-way; round-trip should
    // jump even though the connection is the same one.
    shaper.set_link(true,  LinkParams { delay_ms: 50.0, ..Default::default() }).await;
    shaper.set_link(false, LinkParams { delay_ms: 50.0, ..Default::default() }).await;

    let t1 = std::time::Instant::now();
    c.write_all(b"b").await.unwrap();
    c.read_exact(&mut buf).await.unwrap();
    let warm = t1.elapsed().as_millis() as u64;
    assert!(warm >= 90, "warm rtt {warm}ms should be near 100ms");
}

#[tokio::test]
async fn parse_params_accepts_csv() {
    let p = intelnav_netsim::parse_params("delay=40,jitter=4,bw=100,loss=0.01").unwrap();
    assert_eq!(p.delay_ms,  40.0);
    assert_eq!(p.jitter_ms, 4.0);
    assert_eq!(p.bw_mbps,   100.0);
    assert_eq!(p.loss_pct,  0.01);
    assert_eq!(p.reorder_pct, 0.0);
    let empty = intelnav_netsim::parse_params("").unwrap();
    assert_eq!(empty.delay_ms, 0.0);
}
