use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::StreamExt;
use uuid::Uuid;

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

/// ceilings on what a streamed response may make us allocate
const MAX_STREAM_TOOL_CALLS: usize = 64;
const MAX_STREAM_LINE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    /// llama-server can report a failure mid-stream, after a 200
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
}

/// one streamed slice of the assistant message. llama-server sends `content` as
/// an explicit null on the opening chunk, splits thinking into
/// `reasoning_content`, and dribbles tool arguments across many chunks.
#[derive(Debug, Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

fn non_blank(value: String) -> Option<String> {
    Some(value).filter(|text| !text.trim().is_empty())
}

/// collects sse chunks into the one turn the loop acts on
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    content: String,
    reasoning: String,
    calls: Vec<PartialCall>,
}

impl StreamAccumulator {
    /// feeds one raw sse line, returning false once the stream says it is done
    pub fn push(&mut self, line: &str) -> Result<bool> {
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            // blank separators and comment lines carry nothing
            return Ok(true);
        };
        let data = data.trim();
        if data == "[DONE]" {
            return Ok(false);
        }
        let chunk: StreamChunk =
            serde_json::from_str(data).context("decoding a chat completions chunk")?;
        if let Some(error) = chunk.error {
            let detail = error
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| error.to_string(), str::to_string);
            bail!("model stream failed: {detail}");
        }
        for choice in chunk.choices {
            self.apply(choice.delta)?;
        }
        Ok(true)
    }

    fn apply(&mut self, delta: Delta) -> Result<()> {
        if let Some(content) = delta.content {
            self.content.push_str(&content);
        }
        if let Some(reasoning) = delta.reasoning_content {
            self.reasoning.push_str(&reasoning);
        }
        for call in delta.tool_calls.unwrap_or_default() {
            // a chunk without an index continues the call already in flight
            let index = call
                .index
                .unwrap_or_else(|| self.calls.len().saturating_sub(1));
            if index >= MAX_STREAM_TOOL_CALLS {
                bail!("model streamed more than {MAX_STREAM_TOOL_CALLS} tool calls");
            }
            if self.calls.len() <= index {
                self.calls.resize_with(index + 1, PartialCall::default);
            }
            let slot = &mut self.calls[index];
            if let Some(id) = call.id {
                slot.id = id;
            }
            if let Some(function) = call.function {
                if let Some(name) = function.name {
                    slot.name = name;
                }
                if let Some(arguments) = function.arguments {
                    slot.arguments.push_str(&arguments);
                }
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Turn {
        let tool_calls: Vec<ToolCall> = self
            .calls
            .into_iter()
            // a call that never got a name cannot be dispatched anywhere
            .filter(|call| !call.name.is_empty())
            .map(|call| ToolCall {
                id: if call.id.is_empty() {
                    format!("stream_{}", Uuid::new_v4())
                } else {
                    call.id
                },
                kind: "function".to_string(),
                function: FunctionCall {
                    name: call.name,
                    arguments: call.arguments,
                },
            })
            .collect();
        let text = match non_blank(self.content) {
            Some(text) => Some(text),
            // a thinking model that spent its whole budget on thoughts leaves content
            // empty, so fall back to the thoughts rather than reporting nothing at all
            None if tool_calls.is_empty() => non_blank(self.reasoning),
            None => None,
        };
        Turn { text, tool_calls }
    }
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

    pub fn model(&self) -> &str {
        &self.model
    }

    /// streams the completion and accumulates it server side. streaming is what
    /// lets a dropped run stop generation at the server instead of paying for the
    /// whole answer nobody is waiting for.
    pub async fn chat(&self, messages: &[ChatMessage], tools: &[Value]) -> Result<Turn> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
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
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!("model returned {status}: {}", text.trim());
        }

        let mut stream = std::pin::pin!(response.bytes_stream());
        let mut accumulator = StreamAccumulator::default();
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(&chunk.context("reading the model stream")?);
            if buffer.len() > MAX_STREAM_LINE_BYTES {
                bail!("model streamed an oversized line");
            }
            while let Some(end) = buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = buffer.drain(..=end).collect();
                // a full line is complete json, so lossy decoding cannot split a char
                if !accumulator.push(&String::from_utf8_lossy(&line))? {
                    return Ok(accumulator.finish());
                }
            }
        }
        Ok(accumulator.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// feeds a whole sse transcript, stopping at [DONE] like the client does
    fn accumulate(lines: &[&str]) -> Result<Turn> {
        let mut accumulator = StreamAccumulator::default();
        for line in lines {
            if !accumulator.push(line)? {
                break;
            }
        }
        Ok(accumulator.finish())
    }

    /// llama-server b10052 answering without tools: an opening chunk with a null
    /// content, thinking split into reasoning_content, then the real content
    #[test]
    fn accumulates_a_llama_server_text_response() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"object":"chat.completion.chunk"}"#,
            "",
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"The user wants"}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":" one word.\n"}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"PO"}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"NG"}}]}"#,
            r#"data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}"#,
            "data: [DONE]",
        ])
        .expect("accumulates");
        assert_eq!(turn.text.as_deref(), Some("PONG"));
        assert!(turn.tool_calls.is_empty());
    }

    /// the exact delta sequence llama-server streams for a tool call: id and name
    /// on the first chunk, then the arguments one fragment at a time
    #[test]
    fn assembles_tool_call_arguments_from_fragments() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"Call it.\n"}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"id":"WdRbfwqmabBQiIPAe4wSLKYDakX0iL2E","type":"function","function":{"name":"get_weather","arguments":"{"}}]}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"city\":\""}}]}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris"}}]}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\""}}]}}]}"#,
            r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}"#,
            r#"data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{}}]}"#,
            "data: [DONE]",
        ])
        .expect("accumulates");
        assert_eq!(turn.text, None, "thoughts must not become assistant text");
        assert_eq!(
            turn.tool_calls,
            vec![ToolCall {
                id: "WdRbfwqmabBQiIPAe4wSLKYDakX0iL2E".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Paris"}"#.into(),
                },
            }]
        );
    }

    /// a thinking model can burn its whole budget before writing any content
    #[test]
    fn falls_back_to_reasoning_when_nothing_else_came_back() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":"Thinking Process:"}}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":" analyze"}}]}"#,
            r#"data: {"choices":[{"finish_reason":"length","index":0,"delta":{}}]}"#,
            "data: [DONE]",
        ])
        .expect("accumulates");
        assert_eq!(turn.text.as_deref(), Some("Thinking Process: analyze"));
    }

    /// the cloud dialect: no reasoning field, whole tool call in one delta
    #[test]
    fn accumulates_a_cloud_shaped_tool_call() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"list_datasets","arguments":"{}"}}]}}]}"#,
            r#"data: {"choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}"#,
            "data: [DONE]",
        ])
        .expect("accumulates");
        assert_eq!(turn.text, None);
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.name, "list_datasets");
        assert_eq!(turn.tool_calls[0].function.arguments, "{}");
    }

    #[test]
    fn keeps_two_tool_calls_apart_by_index() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"first","arguments":"{\"x\":"}}]}}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"second","arguments":"{\"y\":"}}]}}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}"#,
            "data: [DONE]",
        ])
        .expect("accumulates");
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].function.arguments, r#"{"x":1}"#);
        assert_eq!(turn.tool_calls[1].function.arguments, r#"{"y":2}"#);
    }

    /// llama-server reports a mid-stream failure after already sending a 200
    #[test]
    fn an_error_chunk_fails_the_call() {
        let err = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}"#,
            r#"data: {"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}"#,
        ])
        .unwrap_err();
        assert!(
            err.to_string().contains("Context size has been exceeded"),
            "unhelpful error: {err}"
        );
    }

    #[test]
    fn nothing_after_done_is_read() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"content":"kept"}}]}"#,
            "data: [DONE]",
            r#"data: {"choices":[{"index":0,"delta":{"content":" dropped"}}]}"#,
        ])
        .expect("accumulates");
        assert_eq!(turn.text.as_deref(), Some("kept"));
    }

    #[test]
    fn a_blank_stream_yields_an_empty_turn() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"content":"   "}}]}"#,
            "data: [DONE]",
        ])
        .expect("accumulates");
        assert_eq!(turn, Turn::default());
    }

    #[test]
    fn a_runaway_tool_call_index_is_refused() {
        let err = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":100000,"id":"x","function":{"name":"boom","arguments":"{}"}}]}}]}"#,
        ])
        .unwrap_err();
        assert!(err.to_string().contains("tool calls"), "wrong error: {err}");
    }

    #[test]
    fn a_nameless_tool_call_is_dropped() {
        let turn = accumulate(&[
            r#"data: {"choices":[{"index":0,"delta":{"content":"hi","tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
            "data: [DONE]",
        ])
        .expect("accumulates");
        assert!(turn.tool_calls.is_empty());
        assert_eq!(turn.text.as_deref(), Some("hi"));
    }
}
