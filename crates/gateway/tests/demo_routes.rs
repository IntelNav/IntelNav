//! Smoke tests for the demo SPA routes.
//!
//! Builds the gateway router in-process and hits the demo endpoints
//! through tower's `oneshot` so no socket / port allocation needed.
//! Verifies:
//!   * `GET /` returns the HTML SPA with the elements the JS depends on
//!   * `GET /v1/swarm/topology` returns the expected JSON shape
//!   * `GET /banner` still serves the plain-text banner

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use intelnav_core::Config;
use intelnav_gateway::server::router;
use intelnav_gateway::GatewayState;
use intelnav_net::{DhtDirectory, StaticDirectory};
use intelnav_runtime::{StepEvent, StepPhase, Telemetry};

fn test_state() -> (GatewayState, Telemetry) {
    let telemetry = Telemetry::default();
    let state = GatewayState {
        config:     Arc::new(Config::default()),
        http:       reqwest::Client::new(),
        static_dir: Arc::new(StaticDirectory::new()),
        dht_dir:    Arc::new(DhtDirectory::new()),
        mdns_dir:   None,
        registry_dir: None,
        started_at: std::time::Instant::now(),
        telemetry:  telemetry.clone(),
    };
    (state, telemetry)
}

async fn get(path: &str) -> (StatusCode, String, Option<String>) {
    let (state, _t) = test_state();
    let app = router(state);
    let resp = app
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let ctype = resp.headers().get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).to_string();
    (status, body, ctype)
}

#[tokio::test]
async fn index_serves_demo_spa() {
    let (status, body, ctype) = get("/").await;
    assert_eq!(status, StatusCode::OK);
    assert!(ctype.unwrap_or_default().contains("text/html"),
            "expected HTML content-type");
    // Sanity-check that the SPA actually shipped, not a stub.
    assert!(body.contains("<title>IntelNav · swarm demo</title>"),
            "SPA title missing — embedded HTML stale?");
    assert!(body.contains("/v1/swarm/topology"),
            "SPA must reference the topology endpoint");
    assert!(body.contains("/v1/chat/completions"),
            "SPA must reference the chat endpoint");
}

#[tokio::test]
async fn topology_endpoint_returns_expected_shape() {
    let (status, body, _ctype) = get("/v1/swarm/topology").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body)
        .expect("topology must be valid JSON");
    assert_eq!(v["gateway"]["kind"], "gateway");
    assert!(v["gateway"]["id"].is_string());
    assert!(v["peers"].is_array());
    assert!(v["models"].is_array());
    assert!(v["uptime_sec"].is_number());
    assert!(v["upstream"].is_string());
}

#[tokio::test]
async fn banner_still_works() {
    let (status, body, _ctype) = get("/banner").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("IntelNav gateway"));
}

#[tokio::test]
async fn swarm_events_streams_telemetry_as_sse() {
    use std::time::Duration;
    let (state, telemetry) = test_state();
    let app = router(state);

    // Drive the SSE stream and the producer concurrently.
    let stream_task = tokio::spawn(async move {
        let resp = app
            .oneshot(
                Request::builder().uri("/v1/swarm/events").body(Body::empty()).unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ctype = resp.headers().get("content-type")
            .and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        assert!(ctype.starts_with("text/event-stream"),
                "expected text/event-stream, got {ctype}");

        // Read just enough bytes to see the first `step` frame.
        let mut body = resp.into_body();
        let mut acc = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            let frame_opt = tokio::time::timeout(
                Duration::from_millis(500),
                http_body_util::BodyExt::frame(&mut body),
            )
            .await;
            match frame_opt {
                Ok(Some(Ok(frame))) => {
                    if let Some(bytes) = frame.data_ref() {
                        acc.extend_from_slice(bytes);
                        let s = String::from_utf8_lossy(&acc).to_string();
                        if s.contains("event: step") || s.contains("event:step") {
                            return s;
                        }
                    }
                }
                _ => continue,
            }
        }
        String::from_utf8_lossy(&acc).into_owned()
    });

    // Give the subscriber a moment to register before emitting.
    tokio::time::sleep(Duration::from_millis(50)).await;
    telemetry.emit(StepEvent {
        seq: 0, at_ms: 0,
        peer_index: 1,
        peer_id: "abc…xyz".into(),
        phase: StepPhase::Decode,
        rtt_ms: 4.2, bytes_up: 12_000, bytes_down: 11_000,
        synthetic: false,
    });

    let body = stream_task.await.unwrap();
    assert!(body.contains("\"peer_id\":\"abc…xyz\"")
            || body.contains("\"peer_id\":\"abc"),
            "step event payload missing in stream: {body}");
    assert!(body.contains("\"phase\":\"decode\""),
            "phase missing in payload: {body}");
}
