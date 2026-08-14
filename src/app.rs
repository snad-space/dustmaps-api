use axum::{Router, extract::State, http::StatusCode, response::Json, routing::get};
use serde::Serialize;
use tower_http::trace::TraceLayer;

#[derive(Clone, Default)]
struct AppState;

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

pub fn router() -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .layer(TraceLayer::new_for_http())
        .with_state(AppState)
}

async fn health(State(_state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    (StatusCode::OK, Json(HealthResponse { status: "ok" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_returns_ok_json() {
        let response = router()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "application/json");
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            r#"{"status":"ok"}"#
        );
    }
}
