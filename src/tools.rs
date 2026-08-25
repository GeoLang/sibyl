use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

const MANIFEST_TTL: Duration = Duration::from_secs(60);

#[derive(Debug, Deserialize)]
struct Manifest {
    tools: Vec<ToolSpec>,
}

#[derive(Debug, Deserialize)]
struct ToolSpec {
    name: String,
    description: String,
    parameters: Value,
}

/// the caller's bearer token, forwarded to the executor untouched so it can
/// reach the services a tool talks to. it lives in memory for one run: never
/// written to the db, never logged, never sent back to a client.
#[derive(Clone, Deserialize)]
#[serde(transparent)]
pub struct UserToken(String);

impl UserToken {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// redacted, so a `{:?}` on anything holding one cannot put it in a log line
impl std::fmt::Debug for UserToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UserToken(<redacted>)")
    }
}

/// tool manifest fetched from the executor, cached so tool edits show up without a restart
pub struct ToolCatalog {
    http: reqwest::Client,
    base_url: String,
    timeout: Duration,
    cached: Mutex<Option<(Instant, Vec<Value>)>>,
}

impl ToolCatalog {
    pub fn new(http: reqwest::Client, base_url: String, timeout: Duration) -> Self {
        Self {
            http,
            base_url,
            timeout,
            cached: Mutex::new(None),
        }
    }

    pub async fn tools(&self) -> Result<Vec<Value>> {
        if let Some((fetched, tools)) = self.cached.lock().expect("catalog mutex").as_ref()
            && fetched.elapsed() < MANIFEST_TTL
        {
            return Ok(tools.clone());
        }
        let tools = self.fetch().await?;
        *self.cached.lock().expect("catalog mutex") = Some((Instant::now(), tools.clone()));
        Ok(tools)
    }

    async fn fetch(&self) -> Result<Vec<Value>> {
        let response = self
            .http
            .get(format!("{}/tools", self.base_url))
            .send()
            .await
            .context("fetching tool manifest")?;
        let status = response.status();
        let body = response.text().await.context("reading tool manifest")?;
        if !status.is_success() {
            bail!("tool manifest returned {status}");
        }
        let manifest: Manifest = serde_json::from_str(&body).context("decoding tool manifest")?;
        Ok(manifest.tools.iter().map(to_openai_tool).collect())
    }

    /// runs a tool, returning the result string or a failure marker to hand back to the model
    pub async fn execute(&self, name: &str, raw_args: &str, user: Option<&UserToken>) -> String {
        match self.try_execute(name, raw_args, user).await {
            Ok(result) => result,
            Err(err) => format!("❌ Tool execution failed: {err}"),
        }
    }

    async fn try_execute(
        &self,
        name: &str,
        raw_args: &str,
        user: Option<&UserToken>,
    ) -> Result<String> {
        let args: Value = if raw_args.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(raw_args).context("invalid tool arguments")?
        };
        let mut request = self
            .http
            .post(format!("{}/tools/{name}", self.base_url))
            .timeout(self.timeout)
            .json(&json!({ "args": args }));
        // no token means the run is headless: the executor calls services
        // unauthenticated rather than falling back to anything of its own
        if let Some(user) = user.filter(|u| !u.0.is_empty()) {
            request = request.bearer_auth(&user.0);
        }
        let response = request.send().await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            bail!("executor returned {status}: {}", body.trim());
        }
        let parsed: Value = serde_json::from_str(&body).context("invalid executor response")?;
        match parsed.get("result") {
            Some(Value::String(result)) => Ok(result.clone()),
            Some(other) => Ok(other.to_string()),
            None => bail!("executor response has no result"),
        }
    }
}

fn to_openai_tool(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// a stand-in executor that answers /tools/{name} and reports back the
    /// Authorization header it was called with
    async fn executor() -> (String, Arc<Mutex<Option<String>>>) {
        let seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let handler_seen = seen.clone();

        let app = axum::Router::new().route(
            "/tools/{name}",
            axum::routing::post(move |headers: axum::http::HeaderMap| {
                let seen = handler_seen.clone();
                async move {
                    *seen.lock().expect("seen") = headers
                        .get("authorization")
                        .map(|v| v.to_str().expect("header").to_string());
                    axum::Json(json!({"result": "ok"}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), seen)
    }

    fn catalog(base_url: String) -> ToolCatalog {
        ToolCatalog::new(reqwest::Client::new(), base_url, Duration::from_secs(5))
    }

    #[tokio::test]
    async fn the_user_token_rides_along_as_a_bearer_header() {
        let (base_url, seen) = executor().await;
        let token = UserToken("header.payload.signature".into());

        let result = catalog(base_url)
            .execute("geocode_place", "{}", Some(&token))
            .await;
        assert_eq!(result, "ok");
        assert_eq!(
            seen.lock().unwrap().as_deref(),
            Some("Bearer header.payload.signature")
        );
    }

    /// headless runs carry no token, and must not send an empty credential
    #[tokio::test]
    async fn no_token_means_no_header() {
        let (base_url, seen) = executor().await;

        let result = catalog(base_url).execute("geocode_place", "{}", None).await;
        assert_eq!(result, "ok");
        assert_eq!(seen.lock().unwrap().as_deref(), None);
    }

    #[tokio::test]
    async fn an_empty_token_is_not_sent() {
        let (base_url, seen) = executor().await;
        let token = UserToken(String::new());

        catalog(base_url)
            .execute("geocode_place", "{}", Some(&token))
            .await;
        assert_eq!(seen.lock().unwrap().as_deref(), None);
    }

    #[test]
    fn debug_does_not_print_the_token() {
        let token = UserToken("header.payload.signature".into());
        let rendered = format!("{token:?} {:?}", Some(token.clone()));
        assert!(!rendered.contains("signature"), "{rendered}");
    }
}
