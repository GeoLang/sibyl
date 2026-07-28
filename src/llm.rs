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
    message: ResponseMessage,
}

/// the assistant message as servers actually send it back, kept separate from
/// `ChatMessage` because llama-server adds `reasoning_content` and sends an
/// empty content string next to tool calls
#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

fn non_blank(value: Option<String>) -> Option<String> {
    value.filter(|text| !text.trim().is_empty())
}

/// parse a chat-completions body into the first choice's turn
pub fn parse_turn(body: &str) -> Result<Turn> {
    let parsed: ChatResponse =
        serde_json::from_str(body).context("decoding chat completions response")?;
    let Some(choice) = parsed.choices.into_iter().next() else {
        bail!("model returned no choices");
    };
    let message = choice.message;
    let tool_calls = message.tool_calls.unwrap_or_default();
    let text = match non_blank(message.content) {
        Some(text) => Some(text),
        // a thinking model that spent its whole budget on thoughts leaves content
        // empty, so fall back to the thoughts rather than reporting nothing at all
        None if tool_calls.is_empty() => non_blank(message.reasoning_content),
        None => None,
    };
    Ok(Turn { text, tool_calls })
}

pub struct Client {
    http: reqwest::Client,
    api_base: String,
    /// none when running against a keyless server, no auth header is sent then
    api_key: Option<String>,
    model: String,
}

impl Client {
    pub fn new(
        http: reqwest::Client,
        api_base: String,
        api_key: Option<String>,
        model: String,
    ) -> Self {
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
        let mut request = self
            .http
            .post(format!("{}/chat/completions", self.api_base))
            .json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.context("calling the model")?;
        let status = response.status();
        let text = response.text().await.context("reading model response")?;
        if !status.is_success() {
            bail!("model returned {status}: {}", text.trim());
        }
        parse_turn(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama-server b10052 answering without tools: real content plus the thoughts
    /// it split out into reasoning_content, and no tool_calls key at all
    #[test]
    fn parses_llama_server_text_response() {
        let body = r#"{"choices":[{"finish_reason":"stop","index":0,"message":{
            "role":"assistant","content":"PONG",
            "reasoning_content":"The user is asking for a specific word.\n"}}],
            "model":"/models/Qwen3.5-9B-Q4_K_M.gguf","object":"chat.completion",
            "system_fingerprint":"b10052-b2dd28a3b"}"#;
        let turn = parse_turn(body).expect("parses");
        assert_eq!(turn.text.as_deref(), Some("PONG"));
        assert!(turn.tool_calls.is_empty());
    }

    /// llama-server sends an empty content string next to tool_calls, and puts the
    /// id after the function object. x.ai sends null content and no reasoning.
    #[test]
    fn parses_llama_server_tool_call_with_empty_content() {
        let body = r#"{"choices":[{"finish_reason":"tool_calls","index":0,"message":{
            "role":"assistant","content":"",
            "reasoning_content":"I should call get_weather with Paris.\n",
            "tool_calls":[{"type":"function","function":{
                "name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
                "id":"KpuUpnfE7rnh4OJlomSmc9sWEl7ffQa2"}]}}]}"#;
        let turn = parse_turn(body).expect("parses");
        assert_eq!(turn.text, None, "thoughts must not become assistant text");
        assert_eq!(
            turn.tool_calls,
            vec![ToolCall {
                id: "KpuUpnfE7rnh4OJlomSmc9sWEl7ffQa2".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Paris"}"#.into(),
                },
            }]
        );
    }

    /// a thinking model can burn its whole budget before writing any content,
    /// which would otherwise make the run finish with nothing to show
    #[test]
    fn falls_back_to_reasoning_when_nothing_else_came_back() {
        let body = r#"{"choices":[{"finish_reason":"length","index":0,"message":{
            "role":"assistant","content":"",
            "reasoning_content":"Thinking Process:\n1. Analyze the request"}}]}"#;
        let turn = parse_turn(body).expect("parses");
        assert_eq!(
            turn.text.as_deref(),
            Some("Thinking Process:\n1. Analyze the request")
        );
    }

    /// the cloud shape stays exactly as it was: null content beside tool calls
    #[test]
    fn parses_cloud_tool_call_with_null_content() {
        let body = r#"{"choices":[{"index":0,"finish_reason":"tool_calls","message":{
            "role":"assistant","content":null,
            "tool_calls":[{"id":"call_123","type":"function","function":{
                "name":"list_datasets","arguments":"{}"}}]}}]}"#;
        let turn = parse_turn(body).expect("parses");
        assert_eq!(turn.text, None);
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.name, "list_datasets");
    }

    #[test]
    fn blank_response_yields_an_empty_turn() {
        let body = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"  "}}]}"#;
        assert_eq!(parse_turn(body).expect("parses"), Turn::default());
    }

    #[test]
    fn no_choices_is_an_error() {
        assert!(parse_turn(r#"{"choices":[]}"#).is_err());
    }
}
