//! Who a request acts as.
//!
//! Platform JWTs are HS256 over `{sub, exp, ...}`, the shape ptolemy mints and
//! every other service validates, so one `PLATFORM_JWT_SECRET` covers sibyl too.
//! The tokens carry no `aud`, which is why this validates with
//! `Validation::default()` like the rest of the platform.
//!
//! The verified `sub` is the session owner. With no secret there is no owner and
//! every session is reachable by anyone who can reach the port, which takes
//! `SIBYL_ALLOW_UNAUTHENTICATED` to say so in writing.

use anyhow::{Result, bail};
use axum::Json;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use jsonwebtoken::{DecodingKey, Validation, decode};
use serde::Deserialize;
use serde_json::json;

pub const SECRET_ENV: &str = "PLATFORM_JWT_SECRET";
pub const UNAUTHENTICATED_ENV: &str = "SIBYL_ALLOW_UNAUTHENTICATED";

const TRUTHY: [&str; 4] = ["1", "true", "yes", "on"];

pub fn truthy(value: Option<String>) -> bool {
    value.is_some_and(|value| TRUTHY.contains(&value.trim().to_ascii_lowercase().as_str()))
}

/// geolang's markers for a token that is not a plain platform bearer: a
/// short scoped tool credential, and one minted for its `/mcp` door
#[derive(Deserialize)]
struct TokenClaims {
    sub: String,
    #[serde(default)]
    token_use: Option<String>,
    #[serde(default)]
    geolang_use: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AuthError {
    Missing,
    Invalid,
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // the reason is not echoed back: separating "expired" from "bad
            // signature" helps an attacker more than a caller
            Self::Missing => f.write_str("missing bearer token"),
            Self::Invalid => f.write_str("invalid or expired token"),
        }
    }
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": self.to_string() })),
        )
            .into_response()
    }
}

pub enum Auth {
    Platform(DecodingKey),
    Unauthenticated,
}

/// the key never reaches a formatter
impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Platform(_) => "Auth::Platform",
            Self::Unauthenticated => "Auth::Unauthenticated",
        })
    }
}

impl Auth {
    pub fn new(secret: Option<String>, allow_unauthenticated: bool) -> Result<Self> {
        match secret {
            Some(secret) => Ok(Self::Platform(DecodingKey::from_secret(secret.as_bytes()))),
            None if allow_unauthenticated => Ok(Self::Unauthenticated),
            None => bail!(
                "{SECRET_ENV} is not set. Set it to the shared platform secret, or set \
                 {UNAUTHENTICATED_ENV}=1 to leave every session readable and writable by \
                 anyone who can reach this port."
            ),
        }
    }

    pub fn unauthenticated(&self) -> bool {
        matches!(self, Self::Unauthenticated)
    }

    /// the subject a request acts as. None is the whole population when the
    /// gate is off, not "some caller with no name"
    pub fn subject(&self, token: Option<&str>) -> Result<Option<String>, AuthError> {
        let key = match self {
            Self::Unauthenticated => return Ok(None),
            Self::Platform(key) => key,
        };
        let token = token.ok_or(AuthError::Missing)?;
        let claims = decode::<TokenClaims>(token, key, &Validation::default())
            .map_err(|_| AuthError::Invalid)?
            .claims;
        if claims.sub.is_empty() {
            return Err(AuthError::Invalid);
        }
        // a token minted for another door of geolang is not a session bearer,
        // and the executor holds the tool ones
        if claims.token_use.is_some() || claims.geolang_use.is_some() {
            return Err(AuthError::Invalid);
        }
        Ok(Some(claims.sub))
    }
}

pub fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    Some(token.trim()).filter(|token| !token.is_empty())
}

#[cfg(test)]
pub mod testing {
    use jsonwebtoken::{EncodingKey, Header, encode};
    use serde_json::json;

    pub const SECRET: &str = "test-platform-secret-at-least-32-chars";

    pub fn token_for(subject: &str) -> String {
        token_expiring(subject, 3_000_000_000)
    }

    pub fn token_expiring(subject: &str, exp: i64) -> String {
        signed(json!({ "sub": subject, "exp": exp, "role": "editor" }))
    }

    pub fn signed(claims: serde_json::Value) -> String {
        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(SECRET.as_bytes()),
        )
        .expect("signing a test token")
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{SECRET, signed, token_expiring, token_for};
    use super::*;

    fn platform() -> Auth {
        Auth::new(Some(SECRET.into()), false).unwrap()
    }

    #[test]
    fn a_missing_secret_is_a_startup_error_unless_it_was_waived() {
        let err = Auth::new(None, false).unwrap_err().to_string();
        assert!(err.contains(SECRET_ENV), "{err}");
        assert!(err.contains(UNAUTHENTICATED_ENV), "{err}");
        assert!(Auth::new(None, true).unwrap().unauthenticated());
        assert!(!platform().unauthenticated());
    }

    #[test]
    fn the_waiver_takes_a_written_yes() {
        for yes in ["1", "true", "YES", " on "] {
            assert!(truthy(Some(yes.into())), "{yes}");
        }
        for no in ["", "0", "false", "  "] {
            assert!(!truthy(Some(no.into())), "{no}");
        }
        assert!(!truthy(None));
    }

    #[test]
    fn a_signed_token_names_the_subject() {
        let auth = platform();
        assert_eq!(
            auth.subject(Some(&token_for("user-1"))).unwrap().as_deref(),
            Some("user-1")
        );
    }

    #[test]
    fn a_bad_token_is_refused_and_a_missing_one_says_so() {
        let auth = platform();
        assert_eq!(auth.subject(None), Err(AuthError::Missing));
        assert_eq!(auth.subject(Some("not-a-jwt")), Err(AuthError::Invalid));

        let forged = jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &serde_json::json!({ "sub": "user-1", "exp": 3_000_000_000i64 }),
            &jsonwebtoken::EncodingKey::from_secret(b"another-secret-entirely-not-ours"),
        )
        .unwrap();
        assert_eq!(auth.subject(Some(&forged)), Err(AuthError::Invalid));
    }

    #[test]
    fn an_expired_token_is_refused() {
        let auth = platform();
        let stale = token_expiring("user-1", 1_000_000_000);
        assert_eq!(auth.subject(Some(&stale)), Err(AuthError::Invalid));
    }

    /// a token with no subject owns nothing, so it cannot stand in for a caller
    #[test]
    fn a_subjectless_token_is_refused() {
        let auth = platform();
        let anonymous = jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &serde_json::json!({ "sub": "", "exp": 3_000_000_000i64 }),
            &jsonwebtoken::EncodingKey::from_secret(SECRET.as_bytes()),
        )
        .unwrap();
        assert_eq!(auth.subject(Some(&anonymous)), Err(AuthError::Invalid));
    }

    /// geolang mints these for its own tool boundary and hands them to the
    /// executor, which runs caller-written code
    #[test]
    fn a_scoped_tool_token_is_not_a_session_bearer() {
        let tool = signed(json!({
            "sub": "user-1",
            "exp": 3_000_000_000i64,
            "token_use": "tool",
            "scope": ["ptolemy:write"],
        }));
        assert_eq!(platform().subject(Some(&tool)), Err(AuthError::Invalid));
    }

    /// the marker says the token is for geolang's `/mcp` door, not this one
    #[test]
    fn an_mcp_token_is_not_a_session_bearer() {
        let mcp = signed(json!({
            "sub": "user-1",
            "exp": 3_000_000_000i64,
            "geolang_use": "mcp",
            "source_role": "editor",
        }));
        assert_eq!(platform().subject(Some(&mcp)), Err(AuthError::Invalid));
    }

    #[test]
    fn with_the_gate_off_nobody_owns_anything() {
        let auth = Auth::new(None, true).unwrap();
        assert_eq!(auth.subject(None).unwrap(), None);
        assert_eq!(auth.subject(Some(&token_for("user-1"))).unwrap(), None);
    }

    #[test]
    fn the_bearer_scheme_is_read_case_insensitively() {
        let mut headers = HeaderMap::new();
        assert_eq!(bearer(&headers), None);

        headers.insert(header::AUTHORIZATION, "bearer abc".parse().unwrap());
        assert_eq!(bearer(&headers), Some("abc"));
        headers.insert(header::AUTHORIZATION, "Bearer  abc ".parse().unwrap());
        assert_eq!(bearer(&headers), Some("abc"));
        headers.insert(header::AUTHORIZATION, "Basic abc".parse().unwrap());
        assert_eq!(bearer(&headers), None);
        headers.insert(header::AUTHORIZATION, "Bearer ".parse().unwrap());
        assert_eq!(bearer(&headers), None);
    }
}
