use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// raw json string exactly as the model sent it
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            ..Default::default()
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            ..Default::default()
        }
    }

    /// rough size estimate, bytes over four
    pub fn estimated_tokens(&self) -> usize {
        let mut bytes = self.role.len();
        bytes += self.content.as_deref().map_or(0, str::len);
        bytes += self.tool_call_id.as_deref().map_or(0, str::len);
        bytes += self.name.as_deref().map_or(0, str::len);
        if let Some(calls) = &self.tool_calls {
            for call in calls {
                bytes += call.id.len() + call.function.name.len() + call.function.arguments.len();
            }
        }
        bytes / 4
    }
}

pub fn estimated_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(ChatMessage::estimated_tokens).sum()
}

/// what one model response asks us to do next
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Turn {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

/// parse a chat-completions body into the first choice's turn
pub fn parse_turn(body: &str) -> Result<Turn> {
    let parsed: ChatResponse =
        serde_json::from_str(body).context("decoding chat completions response")?;
    let Some(choice) = parsed.choices.into_iter().next() else {
        bail!("model returned no choices");
    };
    let text = choice
        .message
        .content
        .filter(|content| !content.trim().is_empty());
    Ok(Turn {
        text,
        tool_calls: choice.message.tool_calls.unwrap_or_default(),
    })
}

pub struct Client {
    http: reqwest::Client,
    api_base: String,
    api_key: String,
    model: String,
}

impl Client {
    pub fn new(http: reqwest::Client, api_base: String, api_key: String, model: String) -> Self {
        Self {
            http,
            api_base,
            api_key,
            model,
        }
    }

    pub async fn chat(&self, messages: &[ChatMessage], tools: &[Value]) -> Result<Turn> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
        }
        let response = self
            .http
            .post(format!("{}/chat/completions", self.api_base))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("calling the model")?;
        let status = response.status();
        let text = response.text().await.context("reading model response")?;
        if !status.is_success() {
            bail!("model returned {status}: {}", text.trim());
        }
        parse_turn(&text)
    }
}
