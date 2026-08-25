//! Two model profiles built from env at startup, with the active one persisted in
//! sqlite so a restart keeps the operator's choice.

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

pub const DEFAULT_API_BASE: &str = "https://api.x.ai/v1";
pub const DEFAULT_MODEL: &str = "grok-4-1-fast-reasoning";

pub const CLOUD: &str = "cloud";
pub const LOCAL: &str = "local";

/// config key holding the active profile id
pub const ACTIVE_KEY: &str = "active_model";

/// what one profile connects to
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    pub base: String,
    pub model: String,
    pub key: Option<String>,
}

/// the two profiles the env asked for, either of which may be absent
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Specs {
    pub cloud: Option<Spec>,
    pub local: Option<Spec>,
}

/// decides both profiles from config. `base` is `SIBYL_API_BASE` with any trailing
/// slash already stripped: unset, or the x.ai url itself, means no local profile.
/// fails when neither profile can be reached, which keeps a forgotten key loud.
pub fn specs(key: Option<String>, base: Option<String>, model: String) -> Result<Specs> {
    let local = base
        .filter(|base| base != DEFAULT_API_BASE)
        .map(|base| Spec {
            base,
            model: model.clone(),
            // never forward the cloud key to a local server. an authenticated
            // alternate provider would need its own key config, not this one
            key: None,
        });
    let cloud = key.map(|key| Spec {
        base: DEFAULT_API_BASE.to_string(),
        // SIBYL_MODEL names the cloud model until a local server claims it
        model: if local.is_some() {
            DEFAULT_MODEL.to_string()
        } else {
            model
        },
        key: Some(key),
    });
    if cloud.is_none() && local.is_none() {
        bail!("XAI_API_KEY must be set to reach {DEFAULT_API_BASE}");
    }
    Ok(Specs { cloud, local })
}

/// deliberately not `Serialize`: the client holds the api key, and the only way
/// out to the wire is `ProfileView`, which has no field for it
pub struct Profile {
    pub id: &'static str,
    pub label: String,
    pub model: String,
    client: Option<Arc<Client>>,
}

impl Profile {
    fn available(&self) -> bool {
        self.client.is_some()
    }
}

#[derive(Serialize)]
struct ProfileView<'a> {
    id: &'a str,
    label: &'a str,
    model: &'a str,
    available: bool,
}

#[derive(Debug, PartialEq)]
pub enum SwitchError {
    Unknown,
    Unavailable,
}

pub struct Models {
    profiles: Vec<Profile>,
    active: Mutex<&'static str>,
}

impl Models {
    /// `stored` is the persisted choice, ignored when it names a profile that is
    /// no longer available so a pulled key cannot leave sibyl pointing at nothing
    pub fn new(
        http: &reqwest::Client,
        specs: Specs,
        stored: Option<String>,
        max_tokens: Option<u32>,
        thinking: bool,
    ) -> Self {
        let profiles = vec![
            build(
                LOCAL,
                specs.local,
                http,
                max_tokens,
                // thinking only reaches the local llama-server: the cloud api
                // would reject or ignore chat_template_kwargs
                thinking,
                |model| format!("Local ({model})"),
                "Local (not configured)",
            ),
            build(
                CLOUD,
                specs.cloud,
                http,
                max_tokens,
                false,
                |_| "Grok (cloud)".to_string(),
                "Grok (cloud, no API key)",
            ),
        ];
        let fallback = if profiles[0].available() {
            LOCAL
        } else {
            CLOUD
        };
        let active = stored
            .as_deref()
            .and_then(canonical)
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

    pub fn active(&self) -> &'static str {
        *self.active.lock().expect("active model mutex")
    }

    fn profile(&self, id: &str) -> Option<&Profile> {
        self.profiles.iter().find(|profile| profile.id == id)
    }

    /// the client a run should use. every reachable active id has one, so an
    /// unavailable profile can never become active.
    pub fn active_client(&self) -> Arc<Client> {
        self.profile(self.active())
            .and_then(|profile| profile.client.clone())
            .expect("the active profile is always available")
    }

    pub fn active_label(&self) -> String {
        self.profile(self.active())
            .map_or_else(String::new, |profile| profile.label.clone())
    }

    /// checks an id the operator asked for without switching to it yet
    pub fn resolve(&self, id: &str) -> Result<&'static str, SwitchError> {
        let id = canonical(id).ok_or(SwitchError::Unknown)?;
        match self.profile(id) {
            Some(profile) if profile.available() => Ok(id),
            _ => Err(SwitchError::Unavailable),
        }
    }

    /// takes effect on the next run, a run already going keeps its own client
    pub fn activate(&self, id: &'static str) {
        *self.active.lock().expect("active model mutex") = id;
    }

    pub fn view(&self) -> Value {
        let profiles: Vec<ProfileView<'_>> = self
            .profiles
            .iter()
            .map(|profile| ProfileView {
                id: profile.id,
                label: &profile.label,
                model: &profile.model,
                available: profile.available(),
            })
            .collect();
        json!({ "active": self.active(), "profiles": profiles })
    }
}

fn build(
    id: &'static str,
    spec: Option<Spec>,
    http: &reqwest::Client,
    max_tokens: Option<u32>,
    thinking: bool,
    label: impl Fn(&str) -> String,
    missing: &str,
) -> Profile {
    match spec {
        Some(spec) => Profile {
            id,
            label: label(&spec.model),
            model: spec.model.clone(),
            client: Some(Arc::new(Client::new(
                http.clone(),
                spec.base,
                spec.key,
                spec.model,
                max_tokens,
                thinking,
            ))),
        },
        None => Profile {
            id,
            label: missing.to_string(),
            model: String::new(),
            client: None,
        },
    }
}

fn canonical(id: &str) -> Option<&'static str> {
    match id {
        CLOUD => Some(CLOUD),
        LOCAL => Some(LOCAL),
        _ => None,
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

    fn both() -> Specs {
        specs(
            Some(KEY.into()),
            Some(LOCAL_BASE.into()),
            "Qwen3.5-9B".into(),
        )
        .unwrap()
    }

    fn models(specs: Specs, stored: Option<&str>) -> Models {
        Models::new(
            &reqwest::Client::new(),
            specs,
            stored.map(str::to_string),
            None,
            false,
        )
    }

    #[test]
    fn key_alone_gives_the_cloud_profile_todays_model() {
        let specs = specs(Some(KEY.into()), None, "grok-3".into()).unwrap();
        assert_eq!(specs.local, None);
        assert_eq!(
            specs.cloud,
            Some(Spec {
                base: DEFAULT_API_BASE.into(),
                model: "grok-3".into(),
                key: Some(KEY.into()),
            })
        );
    }

    #[test]
    fn neither_configured_still_fails_fast() {
        let err = specs(None, None, DEFAULT_MODEL.into()).unwrap_err();
        assert!(err.to_string().contains("XAI_API_KEY"));
    }

    #[test]
    fn a_custom_base_alone_gives_a_keyless_local_profile() {
        let specs = specs(None, Some(LOCAL_BASE.into()), "Qwen3.5-9B".into()).unwrap();
        assert_eq!(specs.cloud, None);
        assert_eq!(
            specs.local,
            Some(Spec {
                base: LOCAL_BASE.into(),
                model: "Qwen3.5-9B".into(),
                key: None,
            })
        );
    }

    #[test]
    fn both_configured_gives_both_profiles() {
        let specs = both();
        assert_eq!(specs.local.as_ref().unwrap().model, "Qwen3.5-9B");
        // SIBYL_MODEL went to the local server, so cloud falls back to its default
        assert_eq!(specs.cloud.as_ref().unwrap().model, DEFAULT_MODEL);
        assert_eq!(specs.cloud.as_ref().unwrap().base, DEFAULT_API_BASE);
        // the cloud key must never ride along to the local server
        assert_eq!(specs.local.as_ref().unwrap().key, None);
        assert!(specs.cloud.as_ref().unwrap().key.is_some());
    }

    /// a trailing slash must not read as a custom base and invent a local profile
    #[test]
    fn the_cloud_url_is_never_a_local_profile() {
        let normalized = "https://api.x.ai/v1/".trim_end_matches('/').to_string();
        let specs = specs(Some(KEY.into()), Some(normalized), DEFAULT_MODEL.into()).unwrap();
        assert_eq!(specs.local, None);
        assert!(specs.cloud.is_some());
    }

    #[test]
    fn local_wins_the_default_when_both_are_available() {
        assert_eq!(models(both(), None).active(), LOCAL);
    }

    #[test]
    fn cloud_is_the_default_when_there_is_no_local() {
        let specs = specs(Some(KEY.into()), None, DEFAULT_MODEL.into()).unwrap();
        assert_eq!(models(specs, None).active(), CLOUD);
    }

    #[test]
    fn a_stored_choice_is_honoured() {
        assert_eq!(models(both(), Some(CLOUD)).active(), CLOUD);
    }

    /// the key was pulled, so the stored cloud choice cannot be honoured
    #[test]
    fn an_unavailable_stored_choice_falls_back() {
        let specs = specs(None, Some(LOCAL_BASE.into()), "Qwen3.5-9B".into()).unwrap();
        assert_eq!(models(specs, Some(CLOUD)).active(), LOCAL);
    }

    #[test]
    fn a_stored_junk_value_falls_back() {
        assert_eq!(models(both(), Some("nonsense")).active(), LOCAL);
    }

    #[test]
    fn switching_changes_the_client_a_run_would_use() {
        let models = models(both(), None);
        assert_eq!(models.active_client().model(), "Qwen3.5-9B");

        models.activate(models.resolve(CLOUD).unwrap());
        assert_eq!(models.active(), CLOUD);
        assert_eq!(models.active_client().model(), DEFAULT_MODEL);
    }

    #[test]
    fn resolve_separates_unknown_from_unavailable() {
        let specs = specs(None, Some(LOCAL_BASE.into()), "Qwen3.5-9B".into()).unwrap();
        let models = models(specs, None);
        assert_eq!(models.resolve("gpt5"), Err(SwitchError::Unknown));
        assert_eq!(models.resolve(CLOUD), Err(SwitchError::Unavailable));
        assert_eq!(models.resolve(LOCAL), Ok(LOCAL));
    }

    #[test]
    fn the_view_lists_both_profiles_with_availability() {
        let specs = specs(None, Some(LOCAL_BASE.into()), "Qwen3.5-9B".into()).unwrap();
        let view = models(specs, None).view();
        assert_eq!(view["active"], LOCAL);
        assert_eq!(view["profiles"][0]["id"], LOCAL);
        assert_eq!(view["profiles"][0]["label"], "Local (Qwen3.5-9B)");
        assert_eq!(view["profiles"][0]["available"], true);
        assert_eq!(view["profiles"][1]["id"], CLOUD);
        assert_eq!(view["profiles"][1]["available"], false);
    }

    /// the whole point of ProfileView: no key, and no base url either
    #[test]
    fn the_view_never_carries_the_key() {
        let body = serde_json::to_string(&models(both(), None).view()).unwrap();
        assert!(!body.contains(KEY), "key leaked into /models: {body}");
        assert!(!body.to_lowercase().contains("authorization"));
        assert!(!body.contains("xai-"));
        assert!(!body.contains(LOCAL_BASE));
    }

    mod endpoints {
        use super::*;
        use crate::db::testing::TempDb;
        use crate::run::RunLimits;
        use crate::tools::ToolCatalog;
        use std::time::Duration;

        fn state(specs: Specs, db: crate::db::Db) -> AppState {
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
            assert_eq!(state.models.active(), LOCAL);

            assert_eq!(put(&state, CLOUD).await, StatusCode::NO_CONTENT);
            assert_eq!(state.models.active(), CLOUD);
            assert_eq!(
                temp.db.get_config(ACTIVE_KEY).unwrap().as_deref(),
                Some(CLOUD)
            );
        }

        #[tokio::test]
        async fn an_unknown_id_is_404_and_an_unavailable_one_is_409() {
            let temp = TempDb::new();
            let local_only = specs(None, Some(LOCAL_BASE.into()), "Qwen3.5-9B".into()).unwrap();
            let state = state(local_only, temp.reopen());

            assert_eq!(put(&state, "gpt5").await, StatusCode::NOT_FOUND);
            assert_eq!(put(&state, CLOUD).await, StatusCode::CONFLICT);
            // neither attempt may move or persist anything
            assert_eq!(state.models.active(), LOCAL);
            assert_eq!(temp.db.get_config(ACTIVE_KEY).unwrap(), None);
        }

        #[tokio::test]
        async fn the_listing_body_holds_no_key() {
            let temp = TempDb::new();
            let state = state(both(), temp.reopen());
            let Json(view) = list(State(state)).await;

            let body = serde_json::to_string(&view).unwrap();
            assert!(!body.contains(KEY), "key leaked into /models: {body}");
            assert_eq!(view["active"], LOCAL);
            assert_eq!(view["profiles"].as_array().unwrap().len(), 2);
        }
    }
}
