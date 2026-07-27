mod db;
mod llm;
mod run;
mod sessions;
mod tools;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Json;
use axum::Router;
use axum::routing::{get, patch, post};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::db::Db;
use crate::tools::ToolCatalog;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Db>,
    pub llm: Arc<llm::Client>,
    pub catalog: Arc<ToolCatalog>,
    pub run_lock: Arc<Mutex<()>>,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let host = env_or("SIBYL_HOST", "0.0.0.0");
    let port: u16 = env_or("SIBYL_PORT", "8090")
        .parse()
        .context("SIBYL_PORT must be a port number")?;
    let db_path = PathBuf::from(env_or("SIBYL_DB_PATH", "/data/sibyl.db"));
    let api_key = std::env::var("XAI_API_KEY").context("XAI_API_KEY must be set")?;
    let model = env_or("SIBYL_MODEL", "grok-4-1-fast-reasoning");
    let api_base = env_or("SIBYL_API_BASE", "https://api.x.ai/v1");
    let geolang_url = env_or("GEOLANG_URL", "http://geolang-api:8080");
    let tool_timeout = Duration::from_secs(
        env_or("SIBYL_TOOL_TIMEOUT_SECS", "600")
            .parse()
            .context("SIBYL_TOOL_TIMEOUT_SECS must be a number of seconds")?,
    );

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let state = AppState {
        db: Arc::new(Db::open(&db_path)?),
        llm: Arc::new(llm::Client::new(
            http.clone(),
            api_base.trim_end_matches('/').to_string(),
            api_key,
            model.clone(),
        )),
        catalog: Arc::new(ToolCatalog::new(
            http,
            geolang_url.trim_end_matches('/').to_string(),
            tool_timeout,
        )),
        run_lock: Arc::new(Mutex::new(())),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/sessions", get(sessions::list).post(sessions::create))
        .route(
            "/sessions/{id}",
            patch(sessions::rename).delete(sessions::delete),
        )
        .route("/sessions/{id}/activate", post(sessions::activate))
        .route("/sessions/{id}/messages", post(sessions::add_message))
        .route("/runs", post(run::post_run))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .with_context(|| format!("binding {host}:{port}"))?;
    info!(
        "sibyl listening on {host}:{port}, model {model}, db {}",
        db_path.display()
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}
