mod db;
mod llm;
mod run;
mod sessions;
mod tools;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
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

const DEFAULT_API_BASE: &str = "https://api.x.ai/v1";

/// compose files pass unset vars through as `${VAR:-}`, so an empty or blank
/// value has to mean unset or it silently blanks a default
fn present(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_var(key: &str) -> Option<String> {
    present(std::env::var(key).ok())
}

fn env_or(key: &str, default: &str) -> String {
    env_var(key).unwrap_or_else(|| default.to_string())
}

/// x.ai always needs a key, so keep forgetting it a startup failure there. a base
/// the operator pointed elsewhere may be a keyless local server.
fn resolve_api_key(api_base: &str, api_key: Option<String>) -> Result<Option<String>> {
    if api_key.is_none() && api_base == DEFAULT_API_BASE {
        bail!("XAI_API_KEY must be set to reach {DEFAULT_API_BASE}");
    }
    Ok(api_key)
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
    let model = env_or("SIBYL_MODEL", "grok-4-1-fast-reasoning");
    let api_base = env_or("SIBYL_API_BASE", DEFAULT_API_BASE)
        .trim_end_matches('/')
        .to_string();
    let api_key = resolve_api_key(&api_base, env_var("XAI_API_KEY"))?;
    if api_key.is_none() {
        info!("no XAI_API_KEY set, calling {api_base} without authentication");
    }
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
            api_base,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_env_values_fall_back_to_the_default() {
        for blank in ["", "  ", "\n"] {
            assert_eq!(
                present(Some(blank.to_string())).unwrap_or_else(|| DEFAULT_API_BASE.to_string()),
                DEFAULT_API_BASE
            );
        }
        assert_eq!(present(None), None);
        assert_eq!(
            present(Some(" http://127.0.0.1:8080/v1 \n".into())).as_deref(),
            Some("http://127.0.0.1:8080/v1")
        );
    }

    #[test]
    fn the_cloud_base_still_demands_a_key() {
        assert!(resolve_api_key(DEFAULT_API_BASE, None).is_err());
        assert_eq!(
            resolve_api_key(DEFAULT_API_BASE, Some("xai-secret".into())).unwrap(),
            Some("xai-secret".into())
        );
    }

    #[test]
    fn a_custom_base_may_run_keyless() {
        let local = "http://127.0.0.1:18099/v1";
        assert_eq!(resolve_api_key(local, None).unwrap(), None);
        assert_eq!(
            resolve_api_key(local, Some("local".into())).unwrap(),
            Some("local".into())
        );
    }

    /// a trailing slash must not sneak past the cloud check and drop the key requirement
    #[test]
    fn a_trailing_slash_is_still_the_cloud_base() {
        let normalized = "https://api.x.ai/v1/".trim_end_matches('/');
        assert!(resolve_api_key(normalized, None).is_err());
    }
}
