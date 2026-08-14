use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

mod app;

const LISTEN_ADDR: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 80);

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("dustmaps_api=info,tower_http=info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer())
        .init();
}

#[tokio::main]
async fn main() {
    init_tracing();

    let listener = tokio::net::TcpListener::bind(LISTEN_ADDR)
        .await
        .unwrap_or_else(|error| {
            eprintln!("failed to bind {LISTEN_ADDR}: {error}");
            std::process::exit(1);
        });
    tracing::info!(address = %LISTEN_ADDR, "dustmaps API listening");

    axum::serve(listener, app::router())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server failed");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => tracing::info!("received Ctrl+C"),
        _ = terminate => tracing::info!("received SIGTERM"),
    }
}
