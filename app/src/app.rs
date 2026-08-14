use std::sync::Arc;

use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
};
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;

use crate::csfd::CsfdMap;

#[derive(Clone, Default)]
struct AppState {
    csfd: Option<Arc<CsfdMap>>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[allow(dead_code)]
pub fn router() -> Router {
    router_with_state(AppState::default())
}

pub fn router_with_csfd(csfd: Arc<CsfdMap>) -> Router {
    router_with_state(AppState { csfd: Some(csfd) })
}

fn router_with_state(state: AppState) -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/csfd", get(csfd))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health(State(_state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    (StatusCode::OK, Json(HealthResponse { status: "ok" }))
}

#[derive(Debug, Deserialize)]
struct CsfdQuery {
    ra: f64,
    dec: f64,
}

#[derive(Debug, Serialize)]
struct EbvResponse {
    ebv: f64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

async fn csfd(
    State(state): State<AppState>,
    Query(query): Query<CsfdQuery>,
) -> Result<Json<EbvResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !query.ra.is_finite() || !query.dec.is_finite() || query.dec.abs() > 90.0 {
        return Err(bad_request(
            "ra and dec must be finite, with -90 <= dec <= 90",
        ));
    }
    let Some(map) = state.csfd else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "CSFD map is not configured".into(),
            }),
        ));
    };
    let result = tokio::task::spawn_blocking(move || map.query(query.ra, query.dec))
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: error.to_string(),
                }),
            )
        })?
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: error.to_string(),
                }),
            )
        })?;
    Ok(Json(EbvResponse { ebv: result }))
}

fn bad_request(error: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.into(),
        }),
    )
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

    #[tokio::test]
    async fn csfd_rejects_invalid_declination() {
        let response = router()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/v1/csfd?ra=0&dec=91")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn csfd_reports_unconfigured_map() {
        let response = router()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/v1/csfd?ra=0&dec=0")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
