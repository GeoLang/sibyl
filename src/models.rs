//! Named providers, each with its own base URL, optional key and model list.
//! Env seeds one `cloud` and one `local` provider. Settings can add more of
//! either kind; the full list is stored in sqlite and overrides env on restart.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};
use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::AppState;
use crate::auth;
use crate::db::Db;
use crate::llm::Client;
use crate::sessions::ApiError;

pub const CLOUD_KEY_ENV: &str = "SIBYL_CLOUD_API_KEY";
pub const CLOUD_BASE_ENV: &str = "SIBYL_CLOUD_API_BASE";
pub const CLOUD_MODELS_ENV: &str = "SIBYL_CLOUD_MODELS";
pub const LOCAL_BASE_ENV: &str = "SIBYL_LOCAL_API_BASE";
pub const LOCAL_MODELS_ENV: &str = "SIBYL_LOCAL_MODELS";
pub const LOCAL2_BASE_ENV: &str = "SIBYL_LOCAL2_API_BASE";
pub const LOCAL2_MODELS_ENV: &str = "SIBYL_LOCAL2_MODELS";

pub const DEFAULT_CLOUD_API_BASE: &str = "https://api.x.ai/v1";
pub const DEFAULT_CLOUD_MODELS: &str = "grok-4-1-fast-reasoning";

const MODEL_SEPARATOR: char = ',';
const ID_SEPARATOR: char = ':';

const CLOUD_NAME: &str = "cloud";
const LOCAL_NAME: &str = "local";
const LOCAL2_NAME: &str = "local2";

pub const ACTIVE_KEY: &str = "active_model";
pub const CLOUD_KEY_KEY: &str = "cloud_api_key";
pub const CLOUD_BASE_KEY: &str = "cloud_api_base";
pub const CLOUD_MODELS_KEY: &str = "cloud_models";
pub const PROVIDERS_KEY: &str = "providers";

const LOCAL_PROBE_TIMEOUT: Duration = Duration::from_millis(1500);

pub const LOCAL_DOWN_MESSAGE: &str =
    "The local model isn't running. Start it, or pick a cloud model in Settings.";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Server {
    Cloud,
    Local,
}

impl Server {
    fn name(self) -> &'static str {
        match self {
            Self::Cloud => CLOUD_NAME,
            Self::Local => LOCAL_NAME,
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            CLOUD_NAME => Some(Self::Cloud),
            LOCAL_NAME => Some(Self::Local),
            _ => None,
        }
    }

    fn needs_key(self) -> bool {
        matches!(self, Self::Cloud)
    }

    fn takes_thinking(self) -> bool {
        matches!(self, Self::Local)
    }
}

pub struct Config {
    pub cloud_key: Option<String>,
    pub cloud_base: String,
    pub cloud_models: String,
    pub local_base: Option<String>,
    pub local_models: Option<String>,
    pub local2_base: Option<String>,
    pub local2_models: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cloud_key: None,
            cloud_base: DEFAULT_CLOUD_API_BASE.to_string(),
            cloud_models: DEFAULT_CLOUD_MODELS.to_string(),
            local_base: None,
            local_models: None,
            local2_base: None,
            local2_models: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provider {
    pub id: String,
    pub label: String,
    pub server: Server,
    pub base: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub models: Vec<String>,
}

impl Provider {
    fn available(&self) -> bool {
        self.key.is_some() || !self.server.needs_key()
    }

    fn to_specs(&self) -> Vec<Spec> {
        self.models
            .iter()
            .map(|model| Spec {
                provider: self.id.clone(),
                label: self.label.clone(),
                server: self.server,
                base: self.base.clone(),
                model: model.clone(),
                key: self.key.clone(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    pub provider: String,
    pub label: String,
    pub server: Server,
    pub base: String,
    pub model: String,
    pub key: Option<String>,
}

impl Spec {
    fn available(&self) -> bool {
        self.key.is_some() || !self.server.needs_key()
    }
}

fn parse_models(raw: &str) -> Vec<String> {
    raw.split(MODEL_SEPARATOR)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .collect()
}

fn local_provider(
    name: &str,
    base_env: &str,
    models_env: &str,
    base: Option<String>,
    models: Option<String>,
) -> Result<Option<Provider>> {
    let base = base.map(|base| base.trim_end_matches('/').to_string());
    let models = parse_models(models.as_deref().unwrap_or_default());
    match (&base, models.is_empty()) {
        (Some(_), true) => {
            bail!("{base_env} is set, so {models_env} must name the models it serves")
        }
        (None, false) => {
            bail!("{models_env} is set without {base_env}, so no server serves them")
        }
        _ => {}
    }
    Ok(base.map(|base| Provider {
        id: name.into(),
        label: name.into(),
        server: Server::Local,
        base,
        key: None,
        models,
    }))
}

pub fn providers_from_config(config: Config) -> Result<Vec<Provider>> {
    let mut providers = Vec::new();
    providers.extend(local_provider(
        LOCAL_NAME,
        LOCAL_BASE_ENV,
        LOCAL_MODELS_ENV,
        config.local_base,
        config.local_models,
    )?);
    providers.extend(local_provider(
        LOCAL2_NAME,
        LOCAL2_BASE_ENV,
        LOCAL2_MODELS_ENV,
        config.local2_base,
        config.local2_models,
    )?);
    let cloud_models = parse_models(&config.cloud_models);
    if !cloud_models.is_empty() {
        providers.push(Provider {
            id: CLOUD_NAME.into(),
            label: CLOUD_NAME.into(),
            server: Server::Cloud,
            base: config.cloud_base.trim_end_matches('/').to_string(),
            key: config.cloud_key,
            models: cloud_models,
        });
    }
    if providers.is_empty() {
        bail!("no model is configured, set {CLOUD_KEY_ENV} or {LOCAL_BASE_ENV}");
    }
    Ok(providers)
}

fn present_stored(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn overlay_legacy_cloud(config: &mut Config, db: &Db) -> Result<()> {
    if let Some(base) = present_stored(db.get_config(CLOUD_BASE_KEY)?) {
        config.cloud_base = base;
    }
    if let Some(models) = present_stored(db.get_config(CLOUD_MODELS_KEY)?) {
        config.cloud_models = models;
    }
    if let Some(key) = present_stored(db.get_config(CLOUD_KEY_KEY)?) {
        config.cloud_key = Some(key);
    }
    Ok(())
}

/// sqlite providers list wins; otherwise env plus the old single-cloud rows
pub fn load_providers(config: Config, db: &Db) -> Result<Vec<Provider>> {
    if let Some(raw) = present_stored(db.get_config(PROVIDERS_KEY)?)
        && let Ok(stored) = serde_json::from_str::<Vec<Provider>>(&raw)
        && !stored.is_empty()
    {
        tracing::info!(
            "{} stored providers from Settings override the SIBYL_* env config",
            stored.len()
        );
        return Ok(stored);
    }
    let mut config = config;
    overlay_legacy_cloud(&mut config, db)?;
    providers_from_config(config)
}

fn valid_base(base: &str) -> bool {
    (base.starts_with("http://") || base.starts_with("https://"))
        && !base.contains(char::is_whitespace)
}

fn slug(raw: &str, server: Server) -> String {
    let mut out = String::new();
    for ch in raw.chars().flat_map(char::to_lowercase) {
        let separator = ch == ' ' || ch == '-' || ch == '_';
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if separator && !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        server.name().to_string()
    } else {
        out
    }
}

fn unique_id(preferred: &str, existing: &[Provider]) -> String {
    if !existing.iter().any(|provider| provider.id == preferred) {
        return preferred.to_string();
    }
    for n in 2..1000 {
        let candidate = format!("{preferred}-{n}");
        if !existing.iter().any(|provider| provider.id == candidate) {
            return candidate;
        }
    }
    format!("{preferred}-new")
}

pub struct Profile {
    id: String,
    label: String,
    model: String,
    server: Server,
    provider: String,
    client: Option<Arc<Client>>,
}

impl Profile {
    fn new(spec: Spec, http: &reqwest::Client, max_tokens: Option<u32>, thinking: bool) -> Self {
        let available = spec.available();
        let Spec {
            provider,
            label,
            server,
            base,
            model,
            key,
        } = spec;
        let id = format!("{provider}{ID_SEPARATOR}{model}");
        let display = format!("{model} ({label})");
        let client = available.then(|| {
            Arc::new(Client::new(
                http.clone(),
                base,
                key,
                model.clone(),
                max_tokens,
                thinking && server.takes_thinking(),
            ))
        });
        Self {
            id,
            label: display,
            model,
            server,
            provider,
            client,
        }
    }

    fn available(&self) -> bool {
        self.client.is_some()
    }
}

#[derive(Serialize)]
struct ProfileView {
    id: String,
    label: String,
    model: String,
    server: String,
    provider: String,
    available: bool,
    reachable: bool,
}

#[derive(Debug, PartialEq)]
pub enum SwitchError {
    Unknown,
    Unavailable,
}

#[derive(Debug, PartialEq)]
pub enum CloudError {
    Invalid(&'static str),
    Unreachable,
    UnknownProvider,
}

struct Inner {
    providers: Vec<Provider>,
    profiles: Vec<Profile>,
    active: String,
}

pub struct Models {
    http: reqwest::Client,
    max_tokens: Option<u32>,
    thinking: bool,
    inner: Mutex<Inner>,
}

impl Models {
    pub fn new(
        http: &reqwest::Client,
        providers: Vec<Provider>,
        stored: Option<String>,
        max_tokens: Option<u32>,
        thinking: bool,
    ) -> Self {
        let profiles = profiles_from(http, &providers, max_tokens, thinking);
        let fallback = fallback_id(&profiles);
        let active = stored
            .filter(|id| {
                profiles
                    .iter()
                    .any(|profile| profile.id == *id && profile.available())
            })
            .unwrap_or(fallback);
        Self {
            http: http.clone(),
            max_tokens,
            thinking,
            inner: Mutex::new(Inner {
                providers,
                profiles,
                active,
            }),
        }
    }

    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("models mutex")
    }

    fn rebuild_locked(
        inner: &mut Inner,
        http: &reqwest::Client,
        max_tokens: Option<u32>,
        thinking: bool,
    ) {
        inner.profiles = profiles_from(http, &inner.providers, max_tokens, thinking);
        if !inner
            .profiles
            .iter()
            .any(|profile| profile.id == inner.active && profile.available())
        {
            inner.active = fallback_id(&inner.profiles);
        }
    }

    pub fn active(&self) -> String {
        self.inner().active.clone()
    }

    pub fn active_client(&self) -> Option<Arc<Client>> {
        let inner = self.inner();
        inner
            .profiles
            .iter()
            .find(|profile| profile.id == inner.active)
            .and_then(|profile| profile.client.clone())
    }

    pub fn active_label(&self) -> String {
        let inner = self.inner();
        inner
            .profiles
            .iter()
            .find(|profile| profile.id == inner.active)
            .map_or_else(String::new, |profile| profile.label.clone())
    }

    pub fn resolve(&self, id: &str) -> Result<String, SwitchError> {
        let inner = self.inner();
        let profile = inner
            .profiles
            .iter()
            .find(|profile| profile.id == id)
            .ok_or(SwitchError::Unknown)?;
        if profile.available() {
            Ok(profile.id.clone())
        } else {
            Err(SwitchError::Unavailable)
        }
    }

    fn profile_meta(&self, id: &str) -> Option<(Server, String, String)> {
        let inner = self.inner();
        inner
            .profiles
            .iter()
            .find(|profile| profile.id == id)
            .map(|profile| (profile.server, profile.provider.clone(), profile.id.clone()))
    }

    pub fn activate(&self, id: &str) {
        self.inner().active = id.to_string();
    }

    /// a local host that is down fails the run and keeps the chosen profile
    pub async fn client_for_run(&self) -> Result<Arc<Client>, String> {
        let Some(client) = self.active_client() else {
            return Err("No model is configured. Add one in Settings.".into());
        };
        let Some((Server::Local, provider, _)) = self.profile_meta(&self.active()) else {
            return Ok(client);
        };
        let reach = self.probe_locals().await;
        if reach.get(&provider) == Some(&false) {
            return Err(LOCAL_DOWN_MESSAGE.into());
        }
        Ok(client)
    }

    pub fn view(&self) -> Value {
        self.view_with_reachability(&HashMap::new())
    }

    pub fn view_with_reachability(&self, local_reach: &HashMap<String, bool>) -> Value {
        let inner = self.inner();
        let profiles: Vec<ProfileView> = inner
            .profiles
            .iter()
            .map(|profile| {
                let reachable = match profile.server {
                    Server::Local => *local_reach.get(&profile.provider).unwrap_or(&true),
                    Server::Cloud => profile.available(),
                };
                ProfileView {
                    id: profile.id.clone(),
                    label: profile.label.clone(),
                    model: profile.model.clone(),
                    server: profile.server.name().to_string(),
                    provider: profile.provider.clone(),
                    available: profile.available(),
                    reachable,
                }
            })
            .collect();
        let providers: Vec<Value> = inner
            .providers
            .iter()
            .map(|provider| {
                let reachable = match provider.server {
                    Server::Local => *local_reach.get(&provider.id).unwrap_or(&true),
                    Server::Cloud => provider.available(),
                };
                json!({
                    "id": provider.id,
                    "label": provider.label,
                    "server": provider.server.name(),
                    "base": provider.base,
                    "models": provider.models,
                    "has_key": provider.key.is_some(),
                    "reachable": reachable,
                })
            })
            .collect();
        let cloud = inner
            .providers
            .iter()
            .find(|provider| provider.server == Server::Cloud);
        json!({
            "active": inner.active,
            "profiles": profiles,
            "providers": providers,
            "cloud": cloud.map(|provider| json!({
                "id": provider.id,
                "base": provider.base,
                "models": provider.models.join(","),
                "has_key": provider.key.is_some(),
            })),
        })
    }

    pub fn stored_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.inner().providers)?)
    }

    pub async fn probe_locals(&self) -> HashMap<String, bool> {
        let locals: Vec<(String, String)> = self
            .inner()
            .providers
            .iter()
            .filter(|provider| provider.server == Server::Local)
            .map(|provider| (provider.id.clone(), provider.base.clone()))
            .collect();
        let mut probes = tokio::task::JoinSet::new();
        for (id, base) in locals {
            let http = self.http.clone();
            probes.spawn(async move { (id, probe_base(&http, &base).await) });
        }
        let mut out = HashMap::new();
        while let Some(probe) = probes.join_next().await {
            let (id, reachable) = probe.expect("local probe task");
            out.insert(id, reachable);
        }
        out
    }

    pub fn upsert_provider(&self, update: ProviderUpdate) -> Result<Upserted, CloudError> {
        let mut inner = self.inner();
        let existing = update
            .id
            .as_deref()
            .and_then(|id| inner.providers.iter().find(|provider| provider.id == id))
            .cloned();

        let server = match update
            .server
            .or(existing.as_ref().map(|provider| provider.server))
        {
            Some(server) => server,
            None => return Err(CloudError::Invalid("server must be cloud or local")),
        };
        let base = match update.base {
            Some(base) => {
                let base = base.trim().trim_end_matches('/').to_string();
                if !valid_base(&base) {
                    return Err(CloudError::Invalid("base must be an http or https URL"));
                }
                base
            }
            None => existing
                .as_ref()
                .map(|provider| provider.base.clone())
                .ok_or(CloudError::Invalid("base must be an http or https URL"))?,
        };
        let models = match update.models {
            Some(models) => {
                let parsed = parse_models(&models);
                if parsed.is_empty() {
                    return Err(CloudError::Invalid("models must name at least one model"));
                }
                parsed
            }
            None => existing
                .as_ref()
                .map(|provider| provider.models.clone())
                .filter(|models| !models.is_empty())
                .ok_or(CloudError::Invalid("models must name at least one model"))?,
        };
        let provided_key = match update.key {
            Some(key) => {
                let key = key.trim().to_string();
                if key.is_empty() {
                    return Err(CloudError::Invalid("key must not be empty"));
                }
                Some(key)
            }
            None => None,
        };
        let key = if server.needs_key() {
            provided_key
                .clone()
                .or_else(|| existing.as_ref().and_then(|provider| provider.key.clone()))
        } else {
            None
        };
        let label = update
            .label
            .map(|label| label.trim().to_string())
            .filter(|label| !label.is_empty())
            .or_else(|| existing.as_ref().map(|provider| provider.label.clone()))
            .unwrap_or_else(|| server.name().to_string());
        let id = match update.id.filter(|id| !id.trim().is_empty()) {
            Some(id) => id.trim().to_string(),
            None => unique_id(&slug(&label, server), &inner.providers),
        };

        let provider = Provider {
            id: id.clone(),
            label,
            server,
            base,
            key,
            models,
        };
        if let Some(index) = inner.providers.iter().position(|item| item.id == id) {
            inner.providers[index] = provider;
        } else {
            inner.providers.push(provider);
        }

        let first_new = format!(
            "{id}{ID_SEPARATOR}{}",
            inner
                .providers
                .iter()
                .find(|item| item.id == id)
                .and_then(|item| item.models.first())
                .cloned()
                .unwrap_or_default()
        );
        Self::rebuild_locked(&mut inner, &self.http, self.max_tokens, self.thinking);
        if inner
            .profiles
            .iter()
            .any(|profile| profile.id == first_new && profile.available())
        {
            inner.active = first_new;
        }
        Ok(Upserted {
            id,
            active: inner.active.clone(),
            key: provided_key,
        })
    }

    pub fn remove_provider(&self, id: &str) -> Result<(), CloudError> {
        let mut inner = self.inner();
        let before = inner.providers.len();
        inner.providers.retain(|provider| provider.id != id);
        if inner.providers.len() == before {
            return Err(CloudError::UnknownProvider);
        }
        Self::rebuild_locked(&mut inner, &self.http, self.max_tokens, self.thinking);
        Ok(())
    }

    pub fn configure_cloud(&self, update: CloudUpdate) -> Result<CloudApplied, CloudError> {
        let provided = update.key.clone();
        self.upsert_provider(ProviderUpdate {
            id: Some(CLOUD_NAME.into()),
            label: Some(CLOUD_NAME.into()),
            server: Some(Server::Cloud),
            base: update.base,
            key: update.key,
            models: update.models,
        })?;
        let inner = self.inner();
        let cloud = inner
            .providers
            .iter()
            .find(|provider| provider.id == CLOUD_NAME)
            .ok_or(CloudError::Unreachable)?;
        Ok(CloudApplied {
            base: cloud.base.clone(),
            models: cloud.models.join(","),
            key: provided
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty()),
            active: inner.active.clone(),
        })
    }
}

fn profiles_from(
    http: &reqwest::Client,
    providers: &[Provider],
    max_tokens: Option<u32>,
    thinking: bool,
) -> Vec<Profile> {
    providers
        .iter()
        .flat_map(Provider::to_specs)
        .map(|spec| Profile::new(spec, http, max_tokens, thinking))
        .collect()
}

fn fallback_id(profiles: &[Profile]) -> String {
    profiles
        .iter()
        .find(|profile| profile.available())
        .map(|profile| profile.id.clone())
        .unwrap_or_default()
}

async fn probe_base(http: &reqwest::Client, base: &str) -> bool {
    http.get(format!("{base}/models"))
        .timeout(LOCAL_PROBE_TIMEOUT)
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

pub async fn list(State(state): State<AppState>) -> Json<Value> {
    let reach = state.models.probe_locals().await;
    Json(state.models.view_with_reachability(&reach))
}

#[derive(Deserialize)]
pub struct SwitchPayload {
    pub id: String,
}

pub async fn switch(State(state): State<AppState>, Json(payload): Json<SwitchPayload>) -> Response {
    let id = match state.models.resolve(&payload.id) {
        Ok(id) => id,
        Err(SwitchError::Unknown) => return StatusCode::NOT_FOUND.into_response(),
        Err(SwitchError::Unavailable) => return StatusCode::CONFLICT.into_response(),
    };
    if let Some((Server::Local, provider, _)) = state.models.profile_meta(&id) {
        let reach = state.models.probe_locals().await;
        if reach.get(&provider) == Some(&false) {
            return StatusCode::CONFLICT.into_response();
        }
    }
    if let Err(err) = state.db.set_config(ACTIVE_KEY, &id) {
        return ApiError::from(err).into_response();
    }
    state.models.activate(&id);
    StatusCode::NO_CONTENT.into_response()
}

#[derive(Deserialize)]
pub struct CloudPayload {
    pub base: Option<String>,
    pub key: Option<String>,
    pub models: Option<String>,
}

pub struct CloudUpdate {
    pub base: Option<String>,
    pub key: Option<String>,
    pub models: Option<String>,
}

#[derive(Debug, PartialEq)]
pub struct CloudApplied {
    pub base: String,
    pub models: String,
    pub key: Option<String>,
    pub active: String,
}

#[derive(Deserialize)]
pub struct ProviderPayload {
    pub id: Option<String>,
    pub label: Option<String>,
    pub server: Option<String>,
    pub base: Option<String>,
    pub key: Option<String>,
    pub models: Option<String>,
}

pub struct ProviderUpdate {
    pub id: Option<String>,
    pub label: Option<String>,
    pub server: Option<Server>,
    pub base: Option<String>,
    pub key: Option<String>,
    pub models: Option<String>,
}

pub struct Upserted {
    pub id: String,
    pub active: String,
    pub key: Option<String>,
}

fn require_auth(state: &AppState, headers: &HeaderMap) -> Result<(), auth::AuthError> {
    state.auth.subject(auth::bearer(headers)).map(|_| ())
}

fn persist_state(state: &AppState) -> Result<(), ApiError> {
    let json = state.models.stored_json()?;
    state.db.set_config(PROVIDERS_KEY, &json)?;
    state.db.set_config(ACTIVE_KEY, &state.models.active())?;
    Ok(())
}

fn cloud_err(err: CloudError) -> Response {
    match err {
        CloudError::Invalid(reason) => {
            (StatusCode::BAD_REQUEST, Json(json!({ "error": reason }))).into_response()
        }
        CloudError::Unreachable => StatusCode::CONFLICT.into_response(),
        CloudError::UnknownProvider => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn configure(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<CloudPayload>,
) -> Response {
    if let Err(err) = require_auth(&state, &headers) {
        return err.into_response();
    }
    if let Err(err) = state.models.configure_cloud(CloudUpdate {
        base: payload.base,
        key: payload.key,
        models: payload.models,
    }) {
        return cloud_err(err);
    }
    if let Err(err) = persist_state(&state) {
        return err.into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

pub async fn upsert(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<ProviderPayload>,
) -> Response {
    if let Err(err) = require_auth(&state, &headers) {
        return err.into_response();
    }
    let server = match payload.server.as_deref() {
        None => None,
        Some(raw) => match Server::parse(raw) {
            Some(server) => Some(server),
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "server must be cloud or local" })),
                )
                    .into_response();
            }
        },
    };
    if let Err(err) = state.models.upsert_provider(ProviderUpdate {
        id: payload.id,
        label: payload.label,
        server,
        base: payload.base,
        key: payload.key,
        models: payload.models,
    }) {
        return cloud_err(err);
    }
    if let Err(err) = persist_state(&state) {
        return err.into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

pub async fn remove(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(err) = require_auth(&state, &headers) {
        return err.into_response();
    }
    if let Err(err) = state.models.remove_provider(&id) {
        return cloud_err(err);
    }
    if let Err(err) = persist_state(&state) {
        return err.into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

#[cfg(test)]
mod tests {
    fn specs(config: Config) -> Result<Vec<Spec>> {
        Ok(providers_from_config(config)?
            .iter()
            .flat_map(Provider::to_specs)
            .collect())
    }

    use super::*;

    const KEY: &str = "xai-super-secret-key";
    const LOCAL_BASE: &str = "http://127.0.0.1:18099/v1";
    const LOCAL2_BASE: &str = "http://127.0.0.1:18100/v1";
    const LOCAL_MODEL: &str = "Qwen3.5-9B-Q4_K_M";
    const CLOUD_ID: &str = "cloud:grok-4-1-fast-reasoning";
    const LOCAL_ID: &str = "local:Qwen3.5-9B-Q4_K_M";

    fn both() -> Vec<Provider> {
        providers_from_config(Config {
            cloud_key: Some(KEY.into()),
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap()
    }

    fn local_only() -> Vec<Provider> {
        providers_from_config(Config {
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(LOCAL_MODEL.into()),
            cloud_models: String::new(),
            ..Config::default()
        })
        .unwrap()
    }

    fn cloud_only() -> Vec<Provider> {
        providers_from_config(Config {
            cloud_key: Some(KEY.into()),
            ..Config::default()
        })
        .unwrap()
    }

    fn models(providers: Vec<Provider>, stored: Option<&str>) -> Models {
        Models::new(
            &reqwest::Client::new(),
            providers,
            stored.map(str::to_string),
            None,
            false,
        )
    }

    #[test]
    fn a_model_list_drops_spaces_and_empty_entries() {
        assert_eq!(
            parse_models(" grok-4 , qwen3 ,, ,glm-5 "),
            vec![
                "grok-4".to_string(),
                "qwen3".to_string(),
                "glm-5".to_string()
            ]
        );
        assert!(parse_models(" , ").is_empty());
    }

    #[test]
    fn one_cloud_provider_carries_each_listed_model() {
        let providers = providers_from_config(Config {
            cloud_key: Some(KEY.into()),
            cloud_models: "grok-4, grok-3".into(),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].models, vec!["grok-4", "grok-3"]);
        assert_eq!(providers[0].id, CLOUD_NAME);
    }

    #[test]
    fn local_models_on_one_server_come_first_and_carry_no_key() {
        let specs = specs(Config {
            cloud_key: Some(KEY.into()),
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(format!("{LOCAL_MODEL}, gpt-oss-20b")),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].server, Server::Local);
        assert_eq!(specs[1].server, Server::Local);
        assert_eq!(specs[2].server, Server::Cloud);
        for spec in specs.iter().take(2) {
            assert_eq!(spec.base, LOCAL_BASE);
            assert_eq!(spec.key, None, "the cloud key reached the local server");
            assert_eq!(spec.provider, LOCAL_NAME);
        }
        assert_eq!(specs[2].key.as_deref(), Some(KEY));
    }

    #[test]
    fn both_base_urls_lose_a_trailing_slash() {
        let providers = providers_from_config(Config {
            cloud_key: Some(KEY.into()),
            cloud_base: "https://api.example.com/v1/".into(),
            local_base: Some(format!("{LOCAL_BASE}/")),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(providers[0].base, LOCAL_BASE);
        assert_eq!(providers[1].base, "https://api.example.com/v1");
    }

    #[test]
    fn a_local_base_without_models_is_refused() {
        let err = providers_from_config(Config {
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(" , ".into()),
            ..Config::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains(LOCAL_BASE_ENV), "{err}");
    }

    #[test]
    fn local_models_without_a_base_are_refused() {
        let err = providers_from_config(Config {
            cloud_key: Some(KEY.into()),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains(LOCAL_BASE_ENV), "{err}");
    }

    #[test]
    fn both_local_pairs_give_two_local_providers() {
        let providers = providers_from_config(Config {
            cloud_key: Some(KEY.into()),
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(LOCAL_MODEL.into()),
            local2_base: Some(format!("{LOCAL2_BASE}/")),
            local2_models: Some("gpt-oss-20b".into()),
            ..Config::default()
        })
        .unwrap();
        let ids: Vec<&str> = providers.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec![LOCAL_NAME, LOCAL2_NAME, CLOUD_NAME]);
        assert_eq!(providers[1].base, LOCAL2_BASE);
        assert_eq!(providers[1].label, LOCAL2_NAME);
        assert_eq!(providers[1].server, Server::Local);
        assert_eq!(providers[1].key, None);
        let specs = specs(Config {
            cloud_key: Some(KEY.into()),
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(LOCAL_MODEL.into()),
            local2_base: Some(LOCAL2_BASE.into()),
            local2_models: Some("gpt-oss-20b".into()),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(
            format!("{}{ID_SEPARATOR}{}", specs[1].provider, specs[1].model),
            "local2:gpt-oss-20b"
        );
    }

    #[test]
    fn a_local2_base_without_models_is_refused() {
        let err = providers_from_config(Config {
            local2_base: Some(LOCAL2_BASE.into()),
            local2_models: Some(" , ".into()),
            ..Config::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains(LOCAL2_BASE_ENV), "{err}");
        assert!(err.contains(LOCAL2_MODELS_ENV), "{err}");
    }

    #[test]
    fn local2_models_without_a_base_are_refused() {
        let err = providers_from_config(Config {
            local2_models: Some("gpt-oss-20b".into()),
            ..Config::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains(LOCAL2_BASE_ENV), "{err}");
        assert!(err.contains(LOCAL2_MODELS_ENV), "{err}");
    }

    #[test]
    fn local2_alone_serves_its_models() {
        let providers = providers_from_config(Config {
            local2_base: Some(LOCAL2_BASE.into()),
            local2_models: Some("gpt-oss-20b".into()),
            cloud_models: String::new(),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].id, LOCAL2_NAME);
        let models = models(providers, None);
        assert_eq!(models.active(), "local2:gpt-oss-20b");
        assert_eq!(models.view()["profiles"][0]["server"], LOCAL_NAME);
    }

    #[test]
    fn no_key_starts_with_cloud_unavailable() {
        let models = models(providers_from_config(Config::default()).unwrap(), None);
        let view = models.view();
        assert_eq!(view["cloud"]["has_key"], false);
        assert_eq!(view["profiles"][0]["available"], false);
        assert!(models.active_client().is_none());
    }

    #[test]
    fn a_pasted_key_makes_the_default_cloud_profile_live() {
        let models = models(providers_from_config(Config::default()).unwrap(), None);
        models
            .configure_cloud(CloudUpdate {
                base: None,
                key: Some("sk-live".into()),
                models: None,
            })
            .unwrap();
        assert_eq!(models.active(), CLOUD_ID);
        assert_eq!(models.view()["cloud"]["has_key"], true);
    }

    #[test]
    fn adding_a_second_cloud_keeps_the_first() {
        let models = models(cloud_only(), None);
        models
            .upsert_provider(ProviderUpdate {
                id: Some("anthropic".into()),
                label: Some("Anthropic".into()),
                server: Some(Server::Cloud),
                base: Some("https://api.anthropic.com/v1".into()),
                key: Some("sk-ant".into()),
                models: Some("claude-sonnet-4-5".into()),
            })
            .unwrap();
        let view = models.view();
        let ids: Vec<&str> = view["providers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|provider| provider["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["cloud", "anthropic"]);
        let profile_ids: Vec<&str> = view["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|profile| profile["id"].as_str().unwrap())
            .collect();
        assert_eq!(profile_ids, vec![CLOUD_ID, "anthropic:claude-sonnet-4-5"]);
        assert_eq!(models.active(), "anthropic:claude-sonnet-4-5");
        let body = serde_json::to_string(&view).unwrap();
        assert!(!body.contains("sk-ant"), "{body}");
        assert!(!body.contains(KEY), "{body}");
    }

    #[test]
    fn adding_a_second_local_server_keeps_the_first() {
        let models = models(local_only(), None);
        models
            .upsert_provider(ProviderUpdate {
                id: None,
                label: Some("Workshop".into()),
                server: Some(Server::Local),
                base: Some("http://127.0.0.1:18100/v1".into()),
                key: None,
                models: Some("gpt-oss-20b, qwen3".into()),
            })
            .unwrap();
        let view = models.view();
        assert_eq!(view["providers"].as_array().unwrap().len(), 2);
        assert_eq!(view["profiles"].as_array().unwrap().len(), 3);
        assert_eq!(view["profiles"][1]["id"], "workshop:gpt-oss-20b");
        assert_eq!(view["profiles"][2]["provider"], "workshop");
    }

    #[test]
    fn removing_a_provider_drops_its_profiles() {
        let models = models(both(), None);
        models.remove_provider(CLOUD_NAME).unwrap();
        let view = models.view();
        assert_eq!(view["providers"].as_array().unwrap().len(), 1);
        assert_eq!(view["profiles"][0]["id"], LOCAL_ID);
        assert_eq!(models.active(), LOCAL_ID);
    }

    #[test]
    fn an_id_names_its_provider() {
        let view = models(both(), None).view();
        assert_eq!(view["profiles"][0]["id"], LOCAL_ID);
        assert_eq!(view["profiles"][0]["label"], "Qwen3.5-9B-Q4_K_M (local)");
        assert_eq!(view["profiles"][0]["provider"], LOCAL_NAME);
        assert_eq!(view["profiles"][1]["id"], CLOUD_ID);
    }

    #[test]
    fn local_wins_the_default_when_both_are_available() {
        assert_eq!(models(both(), None).active(), LOCAL_ID);
    }

    #[test]
    fn a_stored_choice_is_honoured() {
        assert_eq!(models(both(), Some(CLOUD_ID)).active(), CLOUD_ID);
    }

    #[test]
    fn an_unavailable_stored_choice_falls_back() {
        assert_eq!(models(local_only(), Some(CLOUD_ID)).active(), LOCAL_ID);
    }

    #[tokio::test]
    async fn a_run_on_a_down_local_host_fails_and_keeps_the_choice() {
        let providers = providers_from_config(Config {
            cloud_key: Some(KEY.into()),
            local_base: Some("http://127.0.0.1:1/v1".into()),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap();
        let models = models(providers, Some(LOCAL_ID));
        assert_eq!(models.active(), LOCAL_ID);
        let err = models.client_for_run().await.err();
        assert_eq!(err.as_deref(), Some(LOCAL_DOWN_MESSAGE));
        assert_eq!(models.active(), LOCAL_ID);
    }

    #[test]
    fn switching_changes_the_client_a_run_would_use() {
        let models = models(both(), None);
        assert_eq!(models.active_client().unwrap().model(), LOCAL_MODEL);
        models.activate(&models.resolve(CLOUD_ID).unwrap());
        assert_eq!(
            models.active_client().unwrap().model(),
            DEFAULT_CLOUD_MODELS
        );
    }

    #[test]
    fn the_view_never_carries_the_key() {
        let view = models(both(), None).view();
        let body = serde_json::to_string(&view).unwrap();
        assert!(!body.contains(KEY), "{body}");
        assert_eq!(view["providers"][1]["has_key"], true);
        assert!(view["providers"][1].get("key").is_none());
        assert_eq!(view["providers"][1]["base"], DEFAULT_CLOUD_API_BASE);
    }

    #[test]
    fn load_providers_prefers_the_stored_list() {
        let temp = crate::db::testing::TempDb::new();
        let stored = vec![Provider {
            id: "anthropic".into(),
            label: "Anthropic".into(),
            server: Server::Cloud,
            base: "https://api.anthropic.com/v1".into(),
            key: Some("sk-ant".into()),
            models: vec!["claude-sonnet-4-5".into()],
        }];
        temp.db
            .set_config(PROVIDERS_KEY, &serde_json::to_string(&stored).unwrap())
            .unwrap();
        let loaded = load_providers(
            Config {
                cloud_key: Some(KEY.into()),
                ..Config::default()
            },
            &temp.db,
        )
        .unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "anthropic");
        assert_eq!(loaded[0].key.as_deref(), Some("sk-ant"));
    }

    #[test]
    fn load_providers_falls_back_to_legacy_cloud_rows() {
        let temp = crate::db::testing::TempDb::new();
        temp.db
            .set_config(CLOUD_BASE_KEY, "https://api.anthropic.com/v1")
            .unwrap();
        temp.db
            .set_config(CLOUD_MODELS_KEY, "claude-sonnet-4-5")
            .unwrap();
        temp.db.set_config(CLOUD_KEY_KEY, "sk-ant-test").unwrap();
        let loaded = load_providers(
            Config {
                cloud_key: Some(KEY.into()),
                ..Config::default()
            },
            &temp.db,
        )
        .unwrap();
        assert_eq!(loaded[0].base, "https://api.anthropic.com/v1");
        assert_eq!(loaded[0].models, vec!["claude-sonnet-4-5"]);
        assert_eq!(loaded[0].key.as_deref(), Some("sk-ant-test"));
    }

    #[test]
    fn configuring_cloud_keeps_local_and_other_clouds() {
        let models = models(both(), None);
        models
            .upsert_provider(ProviderUpdate {
                id: Some("anthropic".into()),
                label: Some("Anthropic".into()),
                server: Some(Server::Cloud),
                base: Some("https://api.anthropic.com/v1".into()),
                key: Some("sk-ant".into()),
                models: Some("claude-sonnet-4-5".into()),
            })
            .unwrap();
        models
            .configure_cloud(CloudUpdate {
                base: Some("https://api.x.ai/v1".into()),
                key: Some("xai-new".into()),
                models: Some("grok-4".into()),
            })
            .unwrap();
        let ids: Vec<String> = models.view()["providers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|provider| provider["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, vec!["local", "cloud", "anthropic"]);
        assert_eq!(models.active(), "cloud:grok-4");
    }

    #[test]
    fn a_blank_or_invalid_cloud_update_is_refused() {
        let models = models(both(), None);
        assert_eq!(
            models.configure_cloud(CloudUpdate {
                base: Some("ftp://example.com".into()),
                key: None,
                models: None,
            }),
            Err(CloudError::Invalid("base must be an http or https URL"))
        );
        assert_eq!(models.active(), LOCAL_ID);
    }

    mod endpoints {
        use super::*;
        use crate::db::testing::TempDb;
        use crate::run::RunLimits;
        use crate::tools::ToolCatalog;
        use std::time::Duration;

        fn state(providers: Vec<Provider>, db: crate::db::Db) -> AppState {
            AppState {
                db: Arc::new(db),
                models: Arc::new(models(providers, None)),
                catalog: Arc::new(ToolCatalog::new(
                    reqwest::Client::new(),
                    "http://127.0.0.1:1".into(),
                    Duration::from_secs(1),
                )),
                auth: Arc::new(crate::auth::Auth::new(None, true).unwrap()),
                limits: RunLimits::default(),
                session_locks: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
            }
        }

        async fn put(state: &AppState, id: &str) -> StatusCode {
            switch(
                State(state.clone()),
                Json(SwitchPayload { id: id.to_string() }),
            )
            .await
            .status()
        }

        #[tokio::test]
        async fn switching_returns_204_and_persists_the_choice() {
            let temp = TempDb::new();
            let state = state(both(), temp.reopen());
            assert_eq!(put(&state, CLOUD_ID).await, StatusCode::NO_CONTENT);
            assert_eq!(state.models.active(), CLOUD_ID);
        }

        #[tokio::test]
        async fn adding_a_provider_persists_the_list_without_the_key_in_the_listing() {
            let temp = TempDb::new();
            let state = state(cloud_only(), temp.reopen());
            let response = upsert(
                State(state.clone()),
                HeaderMap::new(),
                Json(ProviderPayload {
                    id: Some("anthropic".into()),
                    label: Some("Anthropic".into()),
                    server: Some("cloud".into()),
                    base: Some("https://api.anthropic.com/v1".into()),
                    key: Some("sk-ant-saved".into()),
                    models: Some("claude-sonnet-4-5".into()),
                }),
            )
            .await;
            assert_eq!(response.status(), StatusCode::NO_CONTENT);
            let stored: Vec<Provider> =
                serde_json::from_str(&temp.db.get_config(PROVIDERS_KEY).unwrap().unwrap()).unwrap();
            assert_eq!(stored.len(), 2);
            assert_eq!(stored[1].key.as_deref(), Some("sk-ant-saved"));
            let Json(view) = list(State(state)).await;
            let body = serde_json::to_string(&view).unwrap();
            assert!(!body.contains("sk-ant-saved"), "{body}");
            assert_eq!(view["providers"].as_array().unwrap().len(), 2);
        }

        #[tokio::test]
        async fn deleting_a_provider_is_204() {
            let temp = TempDb::new();
            let state = state(both(), temp.reopen());
            let response = remove(
                State(state.clone()),
                HeaderMap::new(),
                Path(CLOUD_NAME.into()),
            )
            .await;
            assert_eq!(response.status(), StatusCode::NO_CONTENT);
            assert_eq!(state.models.active(), LOCAL_ID);
        }

        #[tokio::test]
        async fn switching_to_an_unreachable_local_server_is_409() {
            let temp = TempDb::new();
            let providers = providers_from_config(Config {
                cloud_key: Some(KEY.into()),
                local_base: Some("http://127.0.0.1:1/v1".into()),
                local_models: Some(LOCAL_MODEL.into()),
                ..Config::default()
            })
            .unwrap();
            let state = state(providers, temp.reopen());
            state.models.activate(CLOUD_ID);
            assert_eq!(put(&state, LOCAL_ID).await, StatusCode::CONFLICT);
            assert_eq!(state.models.active(), CLOUD_ID);
        }

        #[tokio::test]
        async fn upserting_without_a_bearer_is_401_when_gated() {
            use crate::auth::testing::{SECRET, token_for};
            use axum::body::Body;
            use axum::http::Request;
            use tower::ServiceExt;

            let temp = TempDb::new();
            let db = std::sync::Arc::new(temp.reopen());
            let mut state = crate::testing::state(
                db,
                crate::auth::Auth::new(Some(SECRET.into()), false).unwrap(),
            );
            state.models = std::sync::Arc::new(models(cloud_only(), None));
            let app = crate::router(state);
            let body = r#"{"id":"anthropic","server":"cloud","base":"https://api.anthropic.com/v1","key":"sk","models":"claude-sonnet-4-5"}"#;

            let unauth = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("PUT")
                        .uri("/model/providers")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

            let authed = app
                .oneshot(
                    Request::builder()
                        .method("PUT")
                        .uri("/model/providers")
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {}", token_for("alice")))
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(authed.status(), StatusCode::NO_CONTENT);
        }
    }
}
