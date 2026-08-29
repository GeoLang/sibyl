mod auth;
mod db;
mod llm;
mod memory;
mod models;
mod run;
mod salvage;
mod sessions;
mod tools;

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Json;
use axum::Router;
use axum::routing::{delete, get, patch, post, put};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::auth::Auth;
use crate::db::Db;
use crate::models::{ACTIVE_KEY, Models};
use crate::run::{DEFAULT_MAX_MODEL_CALLS, DEFAULT_RUN_BUDGET_SECS, RunLimits};
use crate::tools::ToolCatalog;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Db>,
    pub models: Arc<Models>,
    pub catalog: Arc<ToolCatalog>,
    pub auth: Arc<Auth>,
    pub limits: RunLimits,
    /// one mutex per session id, so two runs on different threads overlap and
    /// two runs on the same thread still cannot interleave
    pub session_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
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

fn parse_env<T>(key: &str, raw: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    raw.parse()
        .map_err(|err| anyhow::anyhow!("{key} must be a number: {err}"))
}

fn parse_or<T>(key: &str, raw: Option<String>, default: T) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match raw {
        Some(raw) => parse_env(key, &raw),
        None => Ok(default),
    }
}

/// for knobs whose absence means "do not send this at all", not "use a default"
fn parse_optional<T>(key: &str, raw: Option<String>) -> Result<Option<T>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    raw.map(|raw| parse_env(key, &raw)).transpose()
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

    let auth = Auth::new(
        env_var(auth::SECRET_ENV),
        auth::truthy(env_var(auth::UNAUTHENTICATED_ENV)),
    )?;
    if auth.unauthenticated() {
        info!(
            "{} is not set: sessions have no owner, so anyone who can reach this port can read, \
             rename, delete and write into any of them",
            auth::SECRET_ENV
        );
    }

    let host = env_or("SIBYL_HOST", "0.0.0.0");
    let port: u16 = env_parsed("SIBYL_PORT", 8090)?;
    let db_path = PathBuf::from(env_or("SIBYL_DB_PATH", "/data/sibyl.db"));
    let db = Db::open(&db_path)?;
    let model_config = models::Config {
        cloud_key: env_var(models::CLOUD_KEY_ENV),
        cloud_base: env_or(models::CLOUD_BASE_ENV, models::DEFAULT_CLOUD_API_BASE),
        cloud_models: env_or(models::CLOUD_MODELS_ENV, models::DEFAULT_CLOUD_MODELS),
        local_base: env_var(models::LOCAL_BASE_ENV),
        local_models: env_var(models::LOCAL_MODELS_ENV),
    };
    let providers = models::load_providers(model_config, &db)?;
    if providers
        .iter()
        .any(|provider| provider.server == models::Server::Cloud && provider.key.is_none())
    {
        info!(
            "cloud profiles are unavailable until a key is saved in Settings"
        );
    }
    if let Some(local) = providers
        .iter()
        .find(|provider| provider.server == models::Server::Local)
    {
        info!("local profiles call {} without authentication", local.base);
    }
    let geolang_url = env_or("GEOLANG_URL", "http://geolang-api:8080");
    let tool_timeout = Duration::from_secs(env_parsed("SIBYL_TOOL_TIMEOUT_SECS", 600u64)?);
    // unset sends no max_tokens at all, so a cloud request looks exactly as before
    let max_tokens: Option<u32> = parse_optional("SIBYL_MAX_TOKENS", env_var("SIBYL_MAX_TOKENS"))?;
    let thinking = env_var("SIBYL_THINKING")
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    if thinking {
        info!("thinking enabled for the local profiles");
    }
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
    let models = Models::new(
        &http,
        providers,
        db.get_config(ACTIVE_KEY)?,
        max_tokens,
        thinking,
    );
    let active = models.active_label();
    let state = AppState {
        db: Arc::new(db),
        models: Arc::new(models),
        catalog: Arc::new(ToolCatalog::new(
            http,
            geolang_url.trim_end_matches('/').to_string(),
            tool_timeout,
        )),
        auth: Arc::new(auth),
        limits,
        session_locks: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = router(state);

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

fn router(state: AppState) -> Router {
    Router::new()
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
        .route("/model/cloud", put(models::configure))
        .route("/model/providers", put(models::upsert))
        .route("/model/providers/{id}", delete(models::remove))
        .route("/runs", post(run::post_run))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

#[cfg(test)]
pub mod testing {
    use super::*;

    /// a state whose model profile and tool catalog are never reached, for the
    /// route tests
    pub fn state(db: Arc<Db>, auth: Auth) -> AppState {
        let http = reqwest::Client::new();
        let providers = models::providers_from_config(models::Config {
            cloud_key: Some("test-key".into()),
            ..models::Config::default()
        })
        .expect("cloud profile from a key");
        AppState {
            db,
            models: Arc::new(Models::new(&http, providers, None, None, false)),
            catalog: Arc::new(ToolCatalog::new(
                http,
                "http://127.0.0.1:1".into(),
                Duration::from_secs(1),
            )),
            auth: Arc::new(auth),
            limits: RunLimits::default(),
            session_locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
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

    /// unset must stay unset here: sending no cap is what keeps cloud requests
    /// byte-identical to before
    #[test]
    fn the_token_cap_is_absent_unless_it_is_set() {
        let cap = |raw: Option<&str>| {
            parse_optional::<u32>("SIBYL_MAX_TOKENS", present(raw.map(str::to_string))).unwrap()
        };
        assert_eq!(cap(None), None);
        assert_eq!(cap(Some("")), None);
        assert_eq!(cap(Some("  ")), None);
        assert_eq!(cap(Some("512")), Some(512));
        assert_eq!(cap(Some(" 2048 ")), Some(2048));

        let err = parse_optional::<u32>("SIBYL_MAX_TOKENS", Some("lots".into())).unwrap_err();
        assert!(err.to_string().contains("SIBYL_MAX_TOKENS"));
    }

    #[test]
    fn a_non_numeric_knob_names_itself_in_the_error() {
        let err = parse_or("SIBYL_RUN_BUDGET_SECS", Some("soon".into()), 900u64).unwrap_err();
        assert!(err.to_string().contains("SIBYL_RUN_BUDGET_SECS"));
    }
}
