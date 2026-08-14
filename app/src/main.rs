use std::{
    env,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
};

use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

mod app;
mod bayestar;
mod csfd;
pub mod geometry;

const DEFAULT_LISTEN_ADDR: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 80);

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

    let data_dir = env::var_os("DUSTMAPS_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/data"));
    let csfd_path = data_dir.join("csfd_ebv.npy");
    let csfd = csfd::CsfdMap::open(&csfd_path).unwrap_or_else(|error| {
        eprintln!("failed to load CSFD map: {error}");
        std::process::exit(1);
    });
    let bayestar = bayestar::BayestarMap::open(
        data_dir.join("bayestar_lookup.npy"),
        data_dir.join("bayestar_bestfit.npy"),
    )
    .unwrap_or_else(|error| {
        eprintln!("failed to load Bayestar map: {error}");
        std::process::exit(1);
    });

    let listen_addr = env::var("DUSTMAPS_LISTEN_ADDR")
        .map(|value| {
            value.parse().unwrap_or_else(|error| {
                eprintln!("invalid DUSTMAPS_LISTEN_ADDR {value:?}: {error}");
                std::process::exit(1);
            })
        })
        .unwrap_or(DEFAULT_LISTEN_ADDR);

    let listener = tokio::net::TcpListener::bind(listen_addr)
        .await
        .unwrap_or_else(|error| {
            eprintln!("failed to bind {listen_addr}: {error}");
            std::process::exit(1);
        });
    tracing::info!(address = %listen_addr, "dustmaps API listening");

    axum::serve(listener, app::router_with_maps(csfd, bayestar))
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
