use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::AppState;
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

fn not_found() -> ApiError {
    ApiError(StatusCode::NOT_FOUND, "session not found".into())
}

type ApiResult<T> = Result<T, ApiError>;

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

pub async fn list(State(state): State<AppState>) -> ApiResult<Json<Vec<Session>>> {
    Ok(Json(state.db.list_sessions()?))
}

pub async fn create(
    State(state): State<AppState>,
    Json(payload): Json<NamePayload>,
) -> ApiResult<Json<Session>> {
    Ok(Json(state.db.create_session(&payload.name)?))
}

pub async fn activate(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state.db.activate_session(&id)?.ok_or_else(not_found)?;
    Ok(Json(id_and_name(&session)))
}

pub async fn rename(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(payload): Json<NamePayload>,
) -> ApiResult<Json<Value>> {
    let session = state
        .db
        .rename_session(&id, &payload.name)?
        .ok_or_else(not_found)?;
    Ok(Json(id_and_name(&session)))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state.db.get_session(&id)?.ok_or_else(not_found)?;
    if session.active {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "cannot delete the active session".into(),
        ));
    }
    state.db.delete_session(&id)?;
    Ok(Json(json!({ "deleted": id })))
}

pub async fn add_message(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(payload): Json<ContentPayload>,
) -> ApiResult<StatusCode> {
    if state.db.get_session(&id)?.is_none() {
        return Err(not_found());
    }
    state
        .db
        .append_message(&id, &NewMessage::user(payload.content))?;
    Ok(StatusCode::NO_CONTENT)
}
