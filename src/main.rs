mod db;
mod llm;
mod models;
mod run;
mod salvage;
mod sessions;
mod tools;

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Json;
use axum::Router;
use axum::routing::{get, patch, post, put};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::db::Db;
use crate::models::{ACTIVE_KEY, DEFAULT_MODEL, Models};
use crate::run::{DEFAULT_MAX_MODEL_CALLS, DEFAULT_RUN_BUDGET_SECS, RunLimits};
use crate::tools::ToolCatalog;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Db>,
    pub models: Arc<Models>,
    pub catalog: Arc<ToolCatalog>,
    pub limits: RunLimits,
    pub run_lock: Arc<Mutex<()>>,
}

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

fn parse_or<T>(key: &str, raw: Option<String>, default: T) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match raw {
        Some(raw) => raw
            .parse()
            .map_err(|err| anyhow::anyhow!("{key} must be a number: {err}")),
        None => Ok(default),
    }
}

fn env_parsed<T>(key: &str, default: T) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    parse_or(key, env_var(key), default)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let host = env_or("SIBYL_HOST", "0.0.0.0");
    let port: u16 = env_parsed("SIBYL_PORT", 8090)?;
    let db_path = PathBuf::from(env_or("SIBYL_DB_PATH", "/data/sibyl.db"));
    let model = env_or("SIBYL_MODEL", DEFAULT_MODEL);
    let api_base = env_var("SIBYL_API_BASE").map(|base| base.trim_end_matches('/').to_string());
    let specs = models::specs(env_var("XAI_API_KEY"), api_base, model)?;
    if specs.cloud.is_none() {
        info!("no XAI_API_KEY set, the cloud profile is unavailable");
    }
    if let Some(local) = specs.local.as_ref().filter(|local| local.key.is_none()) {
        info!("local profile calls {} without authentication", local.base);
    }
    let geolang_url = env_or("GEOLANG_URL", "http://geolang-api:8080");
    let tool_timeout = Duration::from_secs(env_parsed("SIBYL_TOOL_TIMEOUT_SECS", 600u64)?);
    let limits = RunLimits {
        max_model_calls: env_parsed("SIBYL_MAX_MODEL_CALLS", DEFAULT_MAX_MODEL_CALLS)?,
        budget: Duration::from_secs(env_parsed(
            "SIBYL_RUN_BUDGET_SECS",
            DEFAULT_RUN_BUDGET_SECS,
        )?),
    };

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let db = Db::open(&db_path)?;
    let models = Models::new(&http, specs, db.get_config(ACTIVE_KEY)?);
    let active = models.active_label();
    let state = AppState {
        db: Arc::new(db),
        models: Arc::new(models),
        catalog: Arc::new(ToolCatalog::new(
            http,
            geolang_url.trim_end_matches('/').to_string(),
            tool_timeout,
        )),
        limits,
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
        .route("/models", get(models::list))
        .route("/model", put(models::switch))
        .route("/runs", post(run::post_run))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .with_context(|| format!("binding {host}:{port}"))?;
    info!(
        "sibyl listening on {host}:{port}, active profile {active}, db {}",
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
            assert_eq!(present(Some(blank.to_string())), None, "blank {blank:?}");
        }
        assert_eq!(present(None), None);
        assert_eq!(
            present(Some(" http://127.0.0.1:8080/v1 \n".into())).as_deref(),
            Some("http://127.0.0.1:8080/v1")
        );
    }

    /// the two run knobs share the blank-is-unset handling of every other var
    #[test]
    fn the_run_knobs_fall_back_when_unset_or_blank() {
        assert_eq!(
            parse_or("SIBYL_MAX_MODEL_CALLS", None, DEFAULT_MAX_MODEL_CALLS).unwrap(),
            DEFAULT_MAX_MODEL_CALLS
        );
        assert_eq!(
            parse_or("SIBYL_MAX_MODEL_CALLS", present(Some("  ".into())), 30usize).unwrap(),
            30
        );
        assert_eq!(
            parse_or(
                "SIBYL_MAX_MODEL_CALLS",
                Some("4".into()),
                DEFAULT_MAX_MODEL_CALLS
            )
            .unwrap(),
            4
        );
        assert_eq!(
            parse_or(
                "SIBYL_RUN_BUDGET_SECS",
                Some("120".into()),
                DEFAULT_RUN_BUDGET_SECS
            )
            .unwrap(),
            120
        );
    }

    #[test]
    fn a_non_numeric_knob_names_itself_in_the_error() {
        let err = parse_or("SIBYL_RUN_BUDGET_SECS", Some("soon".into()), 900u64).unwrap_err();
        assert!(err.to_string().contains("SIBYL_RUN_BUDGET_SECS"));
    }
}
