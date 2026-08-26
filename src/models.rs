//! One profile per model on the cloud server and the local one, built from env at
//! startup, with the active one persisted in sqlite so a restart keeps the
//! operator's choice.

use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::AppState;
use crate::llm::Client;
use crate::sessions::ApiError;

pub const CLOUD_KEY_ENV: &str = "SIBYL_CLOUD_API_KEY";
pub const CLOUD_BASE_ENV: &str = "SIBYL_CLOUD_API_BASE";
pub const CLOUD_MODELS_ENV: &str = "SIBYL_CLOUD_MODELS";
pub const LOCAL_BASE_ENV: &str = "SIBYL_LOCAL_API_BASE";
pub const LOCAL_MODELS_ENV: &str = "SIBYL_LOCAL_MODELS";

pub const DEFAULT_CLOUD_API_BASE: &str = "https://api.x.ai/v1";
pub const DEFAULT_CLOUD_MODELS: &str = "grok-4-1-fast-reasoning";

const MODEL_SEPARATOR: char = ',';
const ID_SEPARATOR: char = ':';

const CLOUD_NAME: &str = "cloud";
const LOCAL_NAME: &str = "local";

/// config key holding the active profile id
pub const ACTIVE_KEY: &str = "active_model";

/// which of the two servers a profile calls
#[derive(Debug, Clone, Copy, PartialEq)]
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

    /// the cloud endpoint rejects an unauthenticated call, llama-server takes one
    fn needs_key(self) -> bool {
        matches!(self, Self::Cloud)
    }

    /// `chat_template_kwargs` is a llama-server extension the cloud api would
    /// reject or ignore
    fn takes_thinking(self) -> bool {
        matches!(self, Self::Local)
    }
}

/// the env the profiles are built from, blank values already read as unset
pub struct Config {
    pub cloud_key: Option<String>,
    pub cloud_base: String,
    pub cloud_models: String,
    pub local_base: Option<String>,
    pub local_models: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cloud_key: None,
            cloud_base: DEFAULT_CLOUD_API_BASE.to_string(),
            cloud_models: DEFAULT_CLOUD_MODELS.to_string(),
            local_base: None,
            local_models: None,
        }
    }
}

/// what one profile connects to
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
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

/// comma separated model ids, blanks dropped so a stray comma is harmless
fn parse_models(raw: &str) -> Vec<String> {
    raw.split(MODEL_SEPARATOR)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .collect()
}

/// one spec per model, local ones first. fails when nothing is reachable, which
/// keeps a forgotten key loud.
pub fn specs(config: Config) -> Result<Vec<Spec>> {
    let local_base = config
        .local_base
        .map(|base| base.trim_end_matches('/').to_string());
    let local_models = parse_models(config.local_models.as_deref().unwrap_or_default());
    match (&local_base, local_models.is_empty()) {
        (Some(_), true) => {
            bail!("{LOCAL_BASE_ENV} is set, so {LOCAL_MODELS_ENV} must name the models it serves")
        }
        (None, false) => {
            bail!("{LOCAL_MODELS_ENV} is set without {LOCAL_BASE_ENV}, so no server serves them")
        }
        _ => {}
    }

    let mut specs = Vec::new();
    if let Some(base) = local_base {
        specs.extend(local_models.into_iter().map(|model| Spec {
            server: Server::Local,
            base: base.clone(),
            model,
            // never forward the cloud key to a local server. an authenticated
            // alternate provider would need its own key config, not this one
            key: None,
        }));
    }
    let cloud_base = config.cloud_base.trim_end_matches('/').to_string();
    specs.extend(
        parse_models(&config.cloud_models)
            .into_iter()
            .map(|model| Spec {
                server: Server::Cloud,
                base: cloud_base.clone(),
                model,
                key: config.cloud_key.clone(),
            }),
    );

    if !specs.iter().any(Spec::available) {
        bail!("no model is reachable, set {CLOUD_KEY_ENV} or {LOCAL_BASE_ENV}");
    }
    Ok(specs)
}

/// deliberately not `Serialize`: the client holds the api key, and the only way
/// out to the wire is `ProfileView`, which has no field for it
pub struct Profile {
    id: String,
    label: String,
    model: String,
    server: Server,
    client: Option<Arc<Client>>,
}

impl Profile {
    fn new(spec: Spec, http: &reqwest::Client, max_tokens: Option<u32>, thinking: bool) -> Self {
        let available = spec.available();
        let Spec {
            server,
            base,
            model,
            key,
        } = spec;
        let id = format!("{}{ID_SEPARATOR}{model}", server.name());
        let label = format!("{model} ({})", server.name());
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
            label,
            model,
            server,
            client,
        }
    }

    fn available(&self) -> bool {
        self.client.is_some()
    }
}

#[derive(Serialize)]
struct ProfileView<'a> {
    id: &'a str,
    label: &'a str,
    model: &'a str,
    server: &'a str,
    available: bool,
}

#[derive(Debug, PartialEq)]
pub enum SwitchError {
    Unknown,
    Unavailable,
}

pub struct Models {
    profiles: Vec<Profile>,
    active: Mutex<String>,
}

impl Models {
    /// `stored` is the persisted choice, ignored when it names a profile that is
    /// no longer available or no longer configured, so a pulled key cannot leave
    /// sibyl pointing at nothing
    pub fn new(
        http: &reqwest::Client,
        specs: Vec<Spec>,
        stored: Option<String>,
        max_tokens: Option<u32>,
        thinking: bool,
    ) -> Self {
        let profiles: Vec<Profile> = specs
            .into_iter()
            .map(|spec| Profile::new(spec, http, max_tokens, thinking))
            .collect();
        let fallback = profiles
            .iter()
            .find(|profile| profile.available())
            .map(|profile| profile.id.clone())
            .unwrap_or_default();
        let active = stored
            .filter(|id| {
                profiles
                    .iter()
                    .any(|profile| profile.id == *id && profile.available())
            })
            .unwrap_or(fallback);
        Self {
            profiles,
            active: Mutex::new(active),
        }
    }

    pub fn active(&self) -> String {
        self.active.lock().expect("active model mutex").clone()
    }

    fn profile(&self, id: &str) -> Option<&Profile> {
        self.profiles.iter().find(|profile| profile.id == id)
    }

    /// the client a run should use. every reachable active id has one, so an
    /// unavailable profile can never become active.
    pub fn active_client(&self) -> Arc<Client> {
        self.profile(&self.active())
            .and_then(|profile| profile.client.clone())
            .expect("the active profile is always available")
    }

    pub fn active_label(&self) -> String {
        self.profile(&self.active())
            .map_or_else(String::new, |profile| profile.label.clone())
    }

    /// checks an id the operator asked for without switching to it yet
    pub fn resolve(&self, id: &str) -> Result<&str, SwitchError> {
        let profile = self.profile(id).ok_or(SwitchError::Unknown)?;
        if profile.available() {
            Ok(&profile.id)
        } else {
            Err(SwitchError::Unavailable)
        }
    }

    /// takes effect on the next run, a run already going keeps its own client
    pub fn activate(&self, id: &str) {
        *self.active.lock().expect("active model mutex") = id.to_string();
    }

    pub fn view(&self) -> Value {
        let profiles: Vec<ProfileView<'_>> = self
            .profiles
            .iter()
            .map(|profile| ProfileView {
                id: &profile.id,
                label: &profile.label,
                model: &profile.model,
                server: profile.server.name(),
                available: profile.available(),
            })
            .collect();
        json!({ "active": self.active(), "profiles": profiles })
    }
}

pub async fn list(State(state): State<AppState>) -> Json<Value> {
    Json(state.models.view())
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
    if let Err(err) = state.db.set_config(ACTIVE_KEY, id) {
        return ApiError::from(err).into_response();
    }
    state.models.activate(id);
    StatusCode::NO_CONTENT.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "xai-super-secret-key";
    const LOCAL_BASE: &str = "http://127.0.0.1:18099/v1";
    const LOCAL_MODEL: &str = "Qwen3.5-9B-Q4_K_M";
    const CLOUD_ID: &str = "cloud:grok-4-1-fast-reasoning";
    const LOCAL_ID: &str = "local:Qwen3.5-9B-Q4_K_M";

    fn both() -> Vec<Spec> {
        specs(Config {
            cloud_key: Some(KEY.into()),
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap()
    }

    fn local_only() -> Vec<Spec> {
        specs(Config {
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap()
    }

    fn cloud_only() -> Vec<Spec> {
        specs(Config {
            cloud_key: Some(KEY.into()),
            ..Config::default()
        })
        .unwrap()
    }

    fn models(specs: Vec<Spec>, stored: Option<&str>) -> Models {
        Models::new(
            &reqwest::Client::new(),
            specs,
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
        assert_eq!(parse_models("solo"), vec!["solo".to_string()]);
        assert!(parse_models(" , ").is_empty());
        assert!(parse_models("").is_empty());
    }

    #[test]
    fn one_cloud_spec_per_listed_model() {
        let specs = specs(Config {
            cloud_key: Some(KEY.into()),
            cloud_models: "grok-4, grok-3".into(),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(
            specs[0],
            Spec {
                server: Server::Cloud,
                base: DEFAULT_CLOUD_API_BASE.into(),
                model: "grok-4".into(),
                key: Some(KEY.into()),
            }
        );
        assert_eq!(specs[1].model, "grok-3");
    }

    #[test]
    fn local_specs_come_first_and_carry_no_key() {
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
        }
        assert_eq!(specs[2].key.as_deref(), Some(KEY));
    }

    #[test]
    fn both_base_urls_lose_a_trailing_slash() {
        let specs = specs(Config {
            cloud_key: Some(KEY.into()),
            cloud_base: "https://api.example.com/v1/".into(),
            local_base: Some(format!("{LOCAL_BASE}/")),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(specs[0].base, LOCAL_BASE);
        assert_eq!(specs[1].base, "https://api.example.com/v1");
    }

    #[test]
    fn a_local_base_without_models_is_refused() {
        let err = specs(Config {
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(" , ".into()),
            ..Config::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains(LOCAL_BASE_ENV), "{err}");
        assert!(err.contains(LOCAL_MODELS_ENV), "{err}");
    }

    #[test]
    fn local_models_without_a_base_are_refused() {
        let err = specs(Config {
            cloud_key: Some(KEY.into()),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains(LOCAL_BASE_ENV), "{err}");
        assert!(err.contains(LOCAL_MODELS_ENV), "{err}");
    }

    #[test]
    fn no_key_and_no_local_base_fails_fast() {
        let err = specs(Config::default()).unwrap_err().to_string();
        assert!(err.contains(CLOUD_KEY_ENV), "{err}");
        assert!(err.contains(LOCAL_BASE_ENV), "{err}");
    }

    #[test]
    fn cloud_profiles_are_listed_unavailable_without_the_key() {
        let specs = specs(Config {
            cloud_models: "grok-4, grok-3".into(),
            local_base: Some(LOCAL_BASE.into()),
            local_models: Some(LOCAL_MODEL.into()),
            ..Config::default()
        })
        .unwrap();
        let view = models(specs, None).view();
        assert_eq!(view["profiles"].as_array().unwrap().len(), 3);
        assert_eq!(view["profiles"][1]["id"], "cloud:grok-4");
        assert_eq!(view["profiles"][1]["available"], false);
        assert_eq!(view["profiles"][2]["available"], false);
    }

    #[test]
    fn an_id_names_its_server_and_a_label_names_both() {
        let view = models(both(), None).view();
        assert_eq!(view["profiles"][0]["id"], LOCAL_ID);
        assert_eq!(view["profiles"][0]["label"], "Qwen3.5-9B-Q4_K_M (local)");
        assert_eq!(view["profiles"][0]["model"], LOCAL_MODEL);
        assert_eq!(view["profiles"][0]["server"], LOCAL_NAME);
        assert_eq!(view["profiles"][1]["id"], CLOUD_ID);
        assert_eq!(
            view["profiles"][1]["label"],
            "grok-4-1-fast-reasoning (cloud)"
        );
        assert_eq!(view["profiles"][1]["server"], CLOUD_NAME);
    }

    #[test]
    fn local_wins_the_default_when_both_are_available() {
        assert_eq!(models(both(), None).active(), LOCAL_ID);
    }

    #[test]
    fn cloud_is_the_default_when_there_is_no_local() {
        assert_eq!(models(cloud_only(), None).active(), CLOUD_ID);
    }

    #[test]
    fn a_stored_choice_is_honoured() {
        assert_eq!(models(both(), Some(CLOUD_ID)).active(), CLOUD_ID);
    }

    /// the key was pulled, so the stored cloud choice cannot be honoured
    #[test]
    fn an_unavailable_stored_choice_falls_back() {
        assert_eq!(models(local_only(), Some(CLOUD_ID)).active(), LOCAL_ID);
    }

    /// the ids before profiles were per model, stored in sqlite by an older build
    #[test]
    fn a_stored_legacy_id_falls_back() {
        for legacy in [CLOUD_NAME, LOCAL_NAME, "nonsense"] {
            assert_eq!(models(both(), Some(legacy)).active(), LOCAL_ID, "{legacy}");
        }
    }

    #[test]
    fn switching_changes_the_client_a_run_would_use() {
        let models = models(both(), None);
        assert_eq!(models.active_client().model(), LOCAL_MODEL);

        models.activate(models.resolve(CLOUD_ID).unwrap());
        assert_eq!(models.active(), CLOUD_ID);
        assert_eq!(models.active_client().model(), DEFAULT_CLOUD_MODELS);
    }

    #[test]
    fn resolve_separates_unknown_from_unavailable() {
        let models = models(local_only(), None);
        assert_eq!(models.resolve("gpt5"), Err(SwitchError::Unknown));
        assert_eq!(models.resolve(LOCAL_NAME), Err(SwitchError::Unknown));
        assert_eq!(models.resolve(CLOUD_ID), Err(SwitchError::Unavailable));
        assert_eq!(models.resolve(LOCAL_ID), Ok(LOCAL_ID));
    }

    /// the whole point of ProfileView: no key, and no base url either
    #[test]
    fn the_view_never_carries_the_key() {
        let body = serde_json::to_string(&models(both(), None).view()).unwrap();
        assert!(!body.contains(KEY), "key leaked into /models: {body}");
        assert!(!body.to_lowercase().contains("authorization"));
        assert!(!body.contains("xai-"));
        assert!(!body.contains(LOCAL_BASE));
        assert!(!body.contains(DEFAULT_CLOUD_API_BASE));
    }

    mod endpoints {
        use super::*;
        use crate::db::testing::TempDb;
        use crate::run::RunLimits;
        use crate::tools::ToolCatalog;
        use std::time::Duration;

        fn state(specs: Vec<Spec>, db: crate::db::Db) -> AppState {
            AppState {
                db: Arc::new(db),
                models: Arc::new(models(specs, None)),
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
            assert_eq!(state.models.active(), LOCAL_ID);

            assert_eq!(put(&state, CLOUD_ID).await, StatusCode::NO_CONTENT);
            assert_eq!(state.models.active(), CLOUD_ID);
            assert_eq!(
                temp.db.get_config(ACTIVE_KEY).unwrap().as_deref(),
                Some(CLOUD_ID)
            );
        }

        #[tokio::test]
        async fn an_unknown_id_is_404_and_an_unavailable_one_is_409() {
            let temp = TempDb::new();
            let state = state(local_only(), temp.reopen());

            assert_eq!(put(&state, "gpt5").await, StatusCode::NOT_FOUND);
            assert_eq!(put(&state, LOCAL_NAME).await, StatusCode::NOT_FOUND);
            assert_eq!(put(&state, CLOUD_ID).await, StatusCode::CONFLICT);
            // neither attempt may move or persist anything
            assert_eq!(state.models.active(), LOCAL_ID);
            assert_eq!(temp.db.get_config(ACTIVE_KEY).unwrap(), None);
        }

        #[tokio::test]
        async fn the_listing_body_holds_no_key() {
            let temp = TempDb::new();
            let state = state(both(), temp.reopen());
            let Json(view) = list(State(state)).await;

            let body = serde_json::to_string(&view).unwrap();
            assert!(!body.contains(KEY), "key leaked into /models: {body}");
            assert_eq!(view["active"], LOCAL_ID);
            assert_eq!(view["profiles"].as_array().unwrap().len(), 2);
        }
    }
}
