use std::collections::HashSet;
use std::convert::Infallible;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::AppState;
use crate::db::{Db, NewMessage, StoredMessage};
use crate::llm::{ChatMessage, Client, ToolCall, Turn, estimated_tokens};

pub const DEFAULT_MAX_MODEL_CALLS: usize = 30;
pub const DEFAULT_RUN_BUDGET_SECS: u64 = 900;
pub const SUMMARIZE_THRESHOLD_TOKENS: usize = 100_000;
pub const KEEP_RECENT_MESSAGES: usize = 20;
pub const MAX_TOOL_OUTPUT_CHARS: usize = 20_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Text {
        content: String,
    },
    ToolCall {
        name: String,
        /// raw json string of the arguments as the model sent them
        args: String,
    },
    ToolReturn {
        name: String,
        content: String,
    },
    Error {
        message: String,
    },
    Done,
}

#[derive(Clone)]
pub struct EventSink(mpsc::Sender<Result<String, Infallible>>);

impl EventSink {
    pub fn new(tx: mpsc::Sender<Result<String, Infallible>>) -> Self {
        Self(tx)
    }

    pub async fn send(&self, event: Event) {
        let mut line = serde_json::to_string(&event).expect("event serialization");
        line.push('\n');
        let _ = self.0.send(Ok(line)).await;
    }

    async fn fail(&self, message: impl std::fmt::Display) {
        // alternate display so anyhow prints the whole context chain
        self.send(Event::Error {
            message: format!("{message:#}"),
        })
        .await;
        self.send(Event::Done).await;
    }
}

#[derive(Debug, Deserialize)]
pub struct RunRequest {
    pub system_prompt: String,
    pub message: String,
}

/// per-run ceilings, both operator-tunable. the wall clock one exists because a
/// slow local model can burn a long time without ever reaching the call cap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunLimits {
    pub max_model_calls: usize,
    pub budget: Duration,
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_model_calls: DEFAULT_MAX_MODEL_CALLS,
            budget: Duration::from_secs(DEFAULT_RUN_BUDGET_SECS),
        }
    }
}

/// which ceiling a run hit, so the error event can say which one
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Exhausted {
    Calls(usize),
    Time(u64),
}

impl std::fmt::Display for Exhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exhausted::Calls(calls) => write!(f, "run exceeded {calls} model calls"),
            Exhausted::Time(secs) => write!(f, "run exceeded its {secs}s time budget"),
        }
    }
}

impl RunLimits {
    /// checked between model calls, never mid-call
    pub fn exhausted(&self, calls: usize, elapsed: Duration) -> Option<Exhausted> {
        if calls >= self.max_model_calls {
            Some(Exhausted::Calls(self.max_model_calls))
        } else if elapsed >= self.budget {
            Some(Exhausted::Time(self.budget.as_secs()))
        } else {
            None
        }
    }
}

pub fn truncate_tool_output(output: &str) -> String {
    if output.chars().count() <= MAX_TOOL_OUTPUT_CHARS {
        return output.to_string();
    }
    let mut truncated: String = output.chars().take(MAX_TOOL_OUTPUT_CHARS).collect();
    truncated.push_str("\n[truncated]");
    truncated
}

fn to_chat_message(stored: &StoredMessage) -> ChatMessage {
    let tool_calls = stored
        .tool_calls
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Vec<ToolCall>>(raw).ok())
        .filter(|calls| !calls.is_empty());
    ChatMessage {
        role: stored.role.clone(),
        content: stored.content.clone(),
        tool_calls,
        tool_call_id: stored.tool_call_id.clone(),
        name: stored.name.clone(),
    }
}

fn build_request(
    system_prompt: &str,
    summary: Option<&str>,
    history: &[StoredMessage],
) -> Vec<ChatMessage> {
    let mut messages = vec![ChatMessage::system(system_prompt)];
    if let Some(summary) = summary.filter(|s| !s.trim().is_empty()) {
        messages.push(ChatMessage::system(format!(
            "Previous conversation summary: {summary}"
        )));
    }
    messages.extend(history.iter().map(to_chat_message));
    messages
}

/// index of the first message to keep verbatim, never leaving a tool result
/// without the assistant message that requested it
fn keep_boundary(history: &[StoredMessage]) -> usize {
    let mut split = history.len().saturating_sub(KEEP_RECENT_MESSAGES);
    while split < history.len() && history[split].role == "tool" {
        split += 1;
    }
    split
}

/// renders older messages as plain text for the summarizer
pub fn flatten_for_summary(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&msg.role);
        out.push_str(": ");
        if let Some(content) = &msg.content {
            out.push_str(content);
        }
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                out.push_str(&format!(
                    "\n[tool call] {}({})",
                    call.function.name, call.function.arguments
                ));
            }
        }
        out.push('\n');
    }
    out
}

/// builds the request message list, summarizing older history when it gets too big.
/// the summarizer is injected so the loop and the tests share this code path.
pub async fn assemble<F, Fut>(
    db: &Db,
    session_id: &str,
    system_prompt: &str,
    summarize: F,
) -> Result<Vec<ChatMessage>>
where
    F: FnOnce(Vec<ChatMessage>) -> Fut,
    Fut: Future<Output = Result<String>>,
{
    let session = db
        .get_session(session_id)?
        .context("session disappeared mid run")?;
    let history = db.messages_after(session_id, session.summary_watermark)?;
    let messages = build_request(system_prompt, session.summary.as_deref(), &history);

    if estimated_tokens(&messages) <= SUMMARIZE_THRESHOLD_TOKENS {
        return Ok(messages);
    }
    let split = keep_boundary(&history);
    if split == 0 {
        return Ok(messages);
    }
    let (older, keep) = history.split_at(split);
    let older_chat: Vec<ChatMessage> = older.iter().map(to_chat_message).collect();

    match summarize(older_chat).await {
        Ok(fresh) => {
            let merged = match session.summary.as_deref() {
                Some(previous) if !previous.trim().is_empty() => format!("{previous}\n{fresh}"),
                _ => fresh,
            };
            let watermark = older.last().map_or(session.summary_watermark, |msg| msg.id);
            db.set_summary(session_id, &merged, watermark)?;
            Ok(build_request(system_prompt, Some(&merged), keep))
        }
        Err(err) => {
            // summarizing failed, drop the oldest messages for this request only
            warn!("summarization failed, dropping oldest messages: {err}");
            Ok(build_request(
                system_prompt,
                session.summary.as_deref(),
                keep,
            ))
        }
    }
}

/// emits the events for one model response and persists it. the tool executor is
/// injected so the event sequence can be tested without http. returns true when
/// the model asked for tools and the loop should continue.
pub async fn execute_turn<F, Fut>(
    db: &Db,
    session_id: &str,
    turn: &Turn,
    execute: F,
    sink: &EventSink,
) -> Result<bool>
where
    F: Fn(String, String) -> Fut,
    Fut: Future<Output = String>,
{
    if let Some(text) = &turn.text {
        sink.send(Event::Text {
            content: text.clone(),
        })
        .await;
    }
    let tool_calls_json = if turn.tool_calls.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&turn.tool_calls)?)
    };
    if turn.text.is_some() || tool_calls_json.is_some() {
        db.append_message(
            session_id,
            &NewMessage::assistant(turn.text.clone(), tool_calls_json),
        )?;
    }

    for call in &turn.tool_calls {
        sink.send(Event::ToolCall {
            name: call.function.name.clone(),
            args: call.function.arguments.clone(),
        })
        .await;
        let result = execute(call.function.name.clone(), call.function.arguments.clone()).await;
        let result = truncate_tool_output(&result);
        sink.send(Event::ToolReturn {
            name: call.function.name.clone(),
            content: result.clone(),
        })
        .await;
        db.append_message(
            session_id,
            &NewMessage::tool(call.id.clone(), call.function.name.clone(), result),
        )?;
    }
    Ok(!turn.tool_calls.is_empty())
}

/// the names the model was actually offered, so salvage can refuse anything else
fn tool_names(tools: &[Value]) -> HashSet<&str> {
    tools
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str())
        .collect()
}

const SUMMARY_INSTRUCTION: &str = "Summarize the conversation below into a compact briefing. \
Preserve dataset names, file paths, key results, and open threads. Keep it factual, no preamble.";

async fn summarize_with_model(client: &Client, older: Vec<ChatMessage>) -> Result<String> {
    let messages = vec![
        ChatMessage::system(SUMMARY_INSTRUCTION),
        ChatMessage::user(flatten_for_summary(&older)),
    ];
    let turn = client.chat(&messages, &[]).await?;
    turn.text.context("summarizer returned no text")
}

async fn agent_loop(state: &AppState, session_id: &str, req: &RunRequest, sink: &EventSink) {
    let tools = match state.catalog.tools().await {
        Ok(tools) => tools,
        Err(err) => return sink.fail(err).await,
    };
    let names = tool_names(&tools);
    // captured once, so switching profiles mid-run cannot swap the client underneath
    let client = state.models.active_client();
    let started = Instant::now();
    let mut calls = 0;

    loop {
        if let Some(exhausted) = state.limits.exhausted(calls, started.elapsed()) {
            return sink.fail(exhausted).await;
        }
        calls += 1;
        let messages = match assemble(&state.db, session_id, &req.system_prompt, |older| {
            summarize_with_model(&client, older)
        })
        .await
        {
            Ok(messages) => messages,
            Err(err) => return sink.fail(err).await,
        };
        let turn = match client.chat(&messages, &tools).await {
            Ok(turn) => turn,
            Err(err) => return sink.fail(err).await,
        };
        let turn = crate::salvage::salvage_turn(turn, &names);
        let executor = |name: String, args: String| {
            let catalog = state.catalog.clone();
            async move { catalog.execute(&name, &args).await }
        };
        match execute_turn(&state.db, session_id, &turn, executor, sink).await {
            Ok(true) => continue,
            Ok(false) => return sink.send(Event::Done).await,
            Err(err) => return sink.fail(err).await,
        }
    }
}

pub async fn post_run(State(state): State<AppState>, Json(req): Json<RunRequest>) -> Response {
    let (tx, rx) = mpsc::channel::<Result<String, Infallible>>(64);
    let sink = EventSink::new(tx);

    tokio::spawn(async move {
        let _guard = state.run_lock.clone().lock_owned().await;
        let session = match state.db.active_session() {
            Ok(Some(session)) => Some(session),
            Ok(None) => match state.db.create_session("Default") {
                Ok(session) => Some(session),
                Err(err) => {
                    sink.fail(err).await;
                    None
                }
            },
            Err(err) => {
                sink.fail(err).await;
                None
            }
        };
        let Some(session) = session else { return };
        if let Err(err) = state
            .db
            .append_message(&session.id, &NewMessage::user(req.message.clone()))
        {
            return sink.fail(err).await;
        }
        agent_loop(&state, &session.id, &req, &sink).await;
    });

    (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::testing::TempDb;
    use crate::llm::parse_turn;

    fn sink_and_events() -> (EventSink, mpsc::Receiver<Result<String, Infallible>>) {
        let (tx, rx) = mpsc::channel(64);
        (EventSink::new(tx), rx)
    }

    async fn drain(
        sink: EventSink,
        mut rx: mpsc::Receiver<Result<String, Infallible>>,
    ) -> Vec<Event> {
        drop(sink);
        let mut events = Vec::new();
        while let Some(Ok(line)) = rx.recv().await {
            events.push(serde_json::from_str(line.trim_end()).expect("event json"));
        }
        events
    }

    #[test]
    fn the_call_cap_trips_before_the_time_budget() {
        let limits = RunLimits {
            max_model_calls: 3,
            budget: Duration::from_secs(900),
        };
        assert_eq!(limits.exhausted(0, Duration::ZERO), None);
        assert_eq!(limits.exhausted(2, Duration::from_secs(899)), None);
        assert_eq!(
            limits.exhausted(3, Duration::ZERO),
            Some(Exhausted::Calls(3))
        );
    }

    #[test]
    fn the_time_budget_trips_while_calls_are_still_spare() {
        let limits = RunLimits {
            max_model_calls: 30,
            budget: Duration::from_secs(120),
        };
        assert_eq!(limits.exhausted(1, Duration::from_secs(119)), None);
        assert_eq!(
            limits.exhausted(1, Duration::from_secs(120)),
            Some(Exhausted::Time(120))
        );
    }

    #[test]
    fn each_budget_names_itself() {
        assert_eq!(
            Exhausted::Calls(30).to_string(),
            "run exceeded 30 model calls"
        );
        assert_eq!(
            Exhausted::Time(900).to_string(),
            "run exceeded its 900s time budget"
        );
    }

    /// the ui only sees the stream, so a tripped budget must arrive as an error event
    #[tokio::test]
    async fn a_tripped_budget_ends_the_stream_with_an_error_event() {
        let (sink, rx) = sink_and_events();
        sink.fail(Exhausted::Time(120)).await;
        assert_eq!(
            drain(sink, rx).await,
            vec![
                Event::Error {
                    message: "run exceeded its 120s time budget".into()
                },
                Event::Done,
            ]
        );
    }

    #[test]
    fn truncates_long_tool_output_once_past_the_limit() {
        let short = "x".repeat(MAX_TOOL_OUTPUT_CHARS);
        assert_eq!(truncate_tool_output(&short), short);

        let long = "y".repeat(MAX_TOOL_OUTPUT_CHARS + 500);
        let cut = truncate_tool_output(&long);
        assert!(cut.ends_with("\n[truncated]"));
        assert_eq!(cut.chars().count(), MAX_TOOL_OUTPUT_CHARS + 12);
    }

    #[tokio::test]
    async fn content_only_response_emits_one_text_event() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        let turn = parse_turn(
            r#"{"choices":[{"message":{"role":"assistant","content":"all done","tool_calls":null}}]}"#,
        )
        .unwrap();

        let (sink, rx) = sink_and_events();
        let more = execute_turn(
            &temp.db,
            &session.id,
            &turn,
            |_name, _args| async { unreachable!("no tools in this response") },
            &sink,
        )
        .await
        .unwrap();

        assert!(!more);
        assert_eq!(
            drain(sink, rx).await,
            vec![Event::Text {
                content: "all done".into()
            }]
        );
        let stored = temp.db.messages_after(&session.id, 0).unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].role, "assistant");
        assert_eq!(stored[0].content.as_deref(), Some("all done"));
    }

    #[tokio::test]
    async fn tool_calls_emit_call_then_return_in_order() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        let turn = parse_turn(
            r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"list_files","arguments":"{\"path\":\"/data\"}"}},
                {"id":"c2","type":"function","function":{"name":"stat","arguments":"{}"}}
            ]}}]}"#,
        )
        .unwrap();

        let (sink, rx) = sink_and_events();
        let more = execute_turn(
            &temp.db,
            &session.id,
            &turn,
            |name, args| async move { format!("{name}:{args}") },
            &sink,
        )
        .await
        .unwrap();

        assert!(more);
        assert_eq!(
            drain(sink, rx).await,
            vec![
                Event::ToolCall {
                    name: "list_files".into(),
                    args: r#"{"path":"/data"}"#.into()
                },
                Event::ToolReturn {
                    name: "list_files".into(),
                    content: r#"list_files:{"path":"/data"}"#.into()
                },
                Event::ToolCall {
                    name: "stat".into(),
                    args: "{}".into()
                },
                Event::ToolReturn {
                    name: "stat".into(),
                    content: "stat:{}".into()
                },
            ]
        );

        let stored = temp.db.messages_after(&session.id, 0).unwrap();
        assert_eq!(
            stored.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            vec!["assistant", "tool", "tool"]
        );
        assert!(stored[0].content.is_none());
        assert_eq!(stored[1].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(stored[2].tool_call_id.as_deref(), Some("c2"));
        // the stored tool calls must replay as valid request messages
        let replayed = to_chat_message(&stored[0]);
        assert_eq!(replayed.tool_calls.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn mixed_response_emits_text_before_tool_events() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        let turn = parse_turn(
            r#"{"choices":[{"message":{"role":"assistant","content":"looking it up","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"search","arguments":"{\"q\":\"roads\"}"}}
            ]}}]}"#,
        )
        .unwrap();

        let (sink, rx) = sink_and_events();
        execute_turn(
            &temp.db,
            &session.id,
            &turn,
            |_name, _args| async { "found nothing".to_string() },
            &sink,
        )
        .await
        .unwrap();

        let events = drain(sink, rx).await;
        assert_eq!(
            events[0],
            Event::Text {
                content: "looking it up".into()
            }
        );
        assert!(matches!(events[1], Event::ToolCall { .. }));
        assert!(matches!(events[2], Event::ToolReturn { .. }));
        assert_eq!(events.len(), 3);
    }

    fn seed_overflow(db: &Db, session_id: &str, count: usize) {
        let bulk = "z".repeat(10_000);
        for i in 0..count {
            db.append_message(session_id, &NewMessage::user(format!("{i} {bulk}")))
                .unwrap();
        }
    }

    #[tokio::test]
    async fn overflow_summarizes_and_keeps_the_recent_window() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        seed_overflow(&temp.db, &session.id, 60);

        let messages = assemble(&temp.db, &session.id, "be useful", |older| async move {
            assert_eq!(older.len(), 40);
            Ok("canned summary".to_string())
        })
        .await
        .unwrap();

        assert_eq!(messages[0].content.as_deref(), Some("be useful"));
        assert_eq!(
            messages[1].content.as_deref(),
            Some("Previous conversation summary: canned summary")
        );
        assert_eq!(messages.len(), 2 + KEEP_RECENT_MESSAGES);
        assert!(messages[2].content.as_deref().unwrap().starts_with("40 "));

        let stored = temp.db.get_session(&session.id).unwrap().unwrap();
        assert_eq!(stored.summary.as_deref(), Some("canned summary"));
        let history = temp.db.messages_after(&session.id, 0).unwrap();
        assert_eq!(stored.summary_watermark, history[39].id);

        // the next assembly reads from the watermark and needs no further summarizing
        let again = assemble(&temp.db, &session.id, "be useful", |_older| async move {
            panic!("should not summarize again")
        })
        .await
        .unwrap();
        assert_eq!(again.len(), 2 + KEEP_RECENT_MESSAGES);
    }

    #[tokio::test]
    async fn summary_merges_with_the_existing_one() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        temp.db.set_summary(&session.id, "earlier", 0).unwrap();
        seed_overflow(&temp.db, &session.id, 60);

        assemble(&temp.db, &session.id, "be useful", |_older| async move {
            Ok("later".to_string())
        })
        .await
        .unwrap();

        let stored = temp.db.get_session(&session.id).unwrap().unwrap();
        assert_eq!(stored.summary.as_deref(), Some("earlier\nlater"));
    }

    #[tokio::test]
    async fn failed_summary_drops_oldest_without_persisting() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        seed_overflow(&temp.db, &session.id, 60);

        let messages = assemble(&temp.db, &session.id, "be useful", |_older| async move {
            Err(anyhow::anyhow!("summarizer down"))
        })
        .await
        .unwrap();

        assert_eq!(messages.len(), 1 + KEEP_RECENT_MESSAGES);
        let stored = temp.db.get_session(&session.id).unwrap().unwrap();
        assert!(stored.summary.is_none());
        assert_eq!(stored.summary_watermark, 0);
    }

    #[tokio::test]
    async fn short_history_is_passed_through_untouched() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        temp.db
            .append_message(&session.id, &NewMessage::user("hi"))
            .unwrap();

        let messages = assemble(&temp.db, &session.id, "be useful", |_older| async move {
            panic!("no summarization for a short history")
        })
        .await
        .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, "user");
    }

    #[tokio::test]
    async fn the_kept_window_never_starts_with_an_orphan_tool_result() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        seed_overflow(&temp.db, &session.id, 40);
        // the boundary lands on this tool result, whose assistant message gets summarized away
        temp.db
            .append_message(
                &session.id,
                &NewMessage::tool("c1".into(), "search".into(), "result".into()),
            )
            .unwrap();
        seed_overflow(&temp.db, &session.id, 19);

        let messages = assemble(&temp.db, &session.id, "be useful", |_older| async move {
            Ok("canned".to_string())
        })
        .await
        .unwrap();

        assert!(messages[2..].iter().all(|m| m.role != "tool"));
        assert_eq!(messages.len(), 2 + (KEEP_RECENT_MESSAGES - 1));
    }
}
