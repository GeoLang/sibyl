use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::AppState;
use crate::auth::{self, AuthError};
use crate::db::{NewMessage, Session};

pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

impl From<AuthError> for ApiError {
    fn from(err: AuthError) -> Self {
        ApiError(StatusCode::UNAUTHORIZED, err.to_string())
    }
}

/// a session of someone else's is answered the same way a missing one is, so a
/// caller cannot learn which ids exist
fn not_found() -> ApiError {
    ApiError(StatusCode::NOT_FOUND, "session not found".into())
}

type ApiResult<T> = Result<T, ApiError>;

fn subject(state: &AppState, headers: &HeaderMap) -> ApiResult<Option<String>> {
    Ok(state.auth.subject(auth::bearer(headers))?)
}

fn id_and_name(session: &Session) -> Value {
    json!({ "id": session.id, "name": session.name })
}

#[derive(Deserialize)]
pub struct NamePayload {
    pub name: String,
}

#[derive(Deserialize)]
pub struct ContentPayload {
    pub content: String,
}

pub async fn list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<Session>>> {
    let subject = subject(&state, &headers)?;
    Ok(Json(state.db.list_sessions(subject.as_deref())?))
}

pub async fn create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<NamePayload>,
) -> ApiResult<Json<Session>> {
    let subject = subject(&state, &headers)?;
    Ok(Json(
        state.db.create_session(&payload.name, subject.as_deref())?,
    ))
}

pub async fn activate(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let subject = subject(&state, &headers)?;
    let session = state
        .db
        .activate_session(&id, subject.as_deref())?
        .ok_or_else(not_found)?;
    Ok(Json(id_and_name(&session)))
}

pub async fn rename(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<NamePayload>,
) -> ApiResult<Json<Value>> {
    let subject = subject(&state, &headers)?;
    let session = state
        .db
        .rename_session(&id, &payload.name, subject.as_deref())?
        .ok_or_else(not_found)?;
    Ok(Json(id_and_name(&session)))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let subject = subject(&state, &headers)?;
    let session = state
        .db
        .get_session(&id, subject.as_deref())?
        .ok_or_else(not_found)?;
    if session.active {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "cannot delete the active session".into(),
        ));
    }
    state.db.delete_session(&id, subject.as_deref())?;
    Ok(Json(json!({ "deleted": id })))
}

pub async fn add_message(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<ContentPayload>,
) -> ApiResult<StatusCode> {
    let subject = subject(&state, &headers)?;
    if state.db.get_session(&id, subject.as_deref())?.is_none() {
        return Err(not_found());
    }
    state
        .db
        .append_message(&id, &NewMessage::user(payload.content))?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use crate::auth::Auth;
    use crate::auth::testing::{SECRET, token_for};
    use crate::db::Db;
    use crate::db::testing::TempDb;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use std::sync::Arc;
    use tower::ServiceExt;

    struct Harness {
        _temp: TempDb,
        db: Arc<Db>,
        router: axum::Router,
    }

    impl Harness {
        fn new(auth: Auth) -> Self {
            let temp = TempDb::new();
            let db = Arc::new(temp.reopen());
            let router = crate::router(crate::testing::state(db.clone(), auth));
            Self {
                _temp: temp,
                db,
                router,
            }
        }

        async fn send(&self, request: Request<Body>) -> (StatusCode, Value) {
            let response = self.router.clone().oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap();
            let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            (status, body)
        }
    }

    fn request(method: &str, path: &str, token: Option<&str>, body: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder().method(method).uri(path);
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        match body {
            Some(body) => builder
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
            None => builder.body(Body::empty()).unwrap(),
        }
    }

    fn gated() -> Harness {
        Harness::new(Auth::new(Some(SECRET.into()), false).unwrap())
    }

    #[tokio::test]
    async fn list_shows_only_the_callers_own_sessions() {
        let harness = gated();
        harness.db.create_session("mine", Some("alice")).unwrap();
        harness.db.create_session("theirs", Some("bob")).unwrap();

        let (status, body) = harness
            .send(request("GET", "/sessions", Some(&token_for("alice")), None))
            .await;

        assert_eq!(status, StatusCode::OK);
        let names: Vec<&str> = body
            .as_array()
            .unwrap()
            .iter()
            .map(|session| session["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["mine"]);
    }

    #[tokio::test]
    async fn a_created_session_belongs_to_the_caller() {
        let harness = gated();
        let (status, body) = harness
            .send(request(
                "POST",
                "/sessions",
                Some(&token_for("alice")),
                Some(r#"{"name":"first"}"#),
            ))
            .await;

        assert_eq!(status, StatusCode::OK);
        let id = body["id"].as_str().unwrap();
        assert_eq!(
            harness.db.session_row(id).unwrap().unwrap().subject,
            Some("alice".to_string())
        );
    }

    #[tokio::test]
    async fn another_subjects_session_is_a_404_everywhere() {
        let harness = gated();
        let mine = harness.db.create_session("mine", Some("alice")).unwrap();
        let bob = token_for("bob");

        for attempt in [
            request(
                "PATCH",
                &format!("/sessions/{}", mine.id),
                Some(&bob),
                Some(r#"{"name":"stolen"}"#),
            ),
            request(
                "DELETE",
                &format!("/sessions/{}", mine.id),
                Some(&bob),
                None,
            ),
            request(
                "POST",
                &format!("/sessions/{}/messages", mine.id),
                Some(&bob),
                Some(r#"{"content":"read this"}"#),
            ),
            request(
                "POST",
                &format!("/sessions/{}/activate", mine.id),
                Some(&bob),
                None,
            ),
        ] {
            let (status, body) = harness.send(attempt).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        }

        let untouched = harness
            .db
            .get_session(&mine.id, Some("alice"))
            .unwrap()
            .unwrap();
        assert_eq!(untouched.name, "mine");
        assert!(
            harness.db.messages_after(&mine.id, 0).unwrap().is_empty(),
            "a foreign message reached the session"
        );
    }

    #[tokio::test]
    async fn a_missing_or_forged_bearer_is_a_401() {
        let harness = gated();
        harness.db.create_session("mine", Some("alice")).unwrap();

        let (status, body) = harness.send(request("GET", "/sessions", None, None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"], "missing bearer token");

        let (status, body) = harness
            .send(request("GET", "/sessions", Some("not-a-jwt"), None))
            .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"], "invalid or expired token");
    }

    #[tokio::test]
    async fn without_a_secret_the_routes_answer_as_they_always_did() {
        let harness = Harness::new(Auth::new(None, true).unwrap());
        harness.db.create_session("first", None).unwrap();

        let (status, body) = harness.send(request("GET", "/sessions", None, None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.as_array().unwrap().len(), 1);

        let (status, body) = harness
            .send(request(
                "POST",
                "/sessions",
                None,
                Some(r#"{"name":"second"}"#),
            ))
            .await;
        assert_eq!(status, StatusCode::OK);
        let id = body["id"].as_str().unwrap().to_string();

        let (status, _) = harness
            .send(request(
                "PATCH",
                &format!("/sessions/{id}"),
                None,
                Some(r#"{"name":"renamed"}"#),
            ))
            .await;
        assert_eq!(status, StatusCode::OK);

        let (status, _) = harness
            .send(request(
                "POST",
                &format!("/sessions/{id}/messages"),
                None,
                Some(r#"{"content":"hello"}"#),
            ))
            .await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert_eq!(harness.db.messages_after(&id, 0).unwrap().len(), 1);
    }
}
