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
    pub async fn execute(&self, name: &str, raw_args: &str) -> String {
        match self.try_execute(name, raw_args).await {
            Ok(result) => result,
            Err(err) => format!("❌ Tool execution failed: {err}"),
        }
    }

    async fn try_execute(&self, name: &str, raw_args: &str) -> Result<String> {
        let args: Value = if raw_args.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(raw_args).context("invalid tool arguments")?
        };
        let response = self
            .http
            .post(format!("{}/tools/{name}", self.base_url))
            .timeout(self.timeout)
            .json(&json!({ "args": args }))
            .send()
            .await?;
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
