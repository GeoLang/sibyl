use std::collections::HashSet;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
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
use tokio_stream::Stream;
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

    /// true once the client stopped reading, which drops the receiving half
    pub fn disconnected(&self) -> bool {
        self.0.is_closed()
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

/// a model that repeats the same failing tool call verbatim has stopped
/// adapting (grok once burned a whole run on 30 identical emit_ui_spec
/// errors), so the run aborts after this many identical failures in a row
pub const MAX_IDENTICAL_FAILURES: usize = 3;

#[derive(Default)]
pub struct RepeatGuard {
    last: Option<(String, String)>,
    failures: usize,
}

impl RepeatGuard {
    /// records one tool result; true when the same call just failed
    /// MAX_IDENTICAL_FAILURES times in a row
    fn tripped(&mut self, name: &str, args: &str, result: &str) -> bool {
        if !(result.starts_with("❌") || result.starts_with("ERROR")) {
            self.last = None;
            self.failures = 0;
            return false;
        }
        let same = self
            .last
            .as_ref()
            .is_some_and(|(n, a)| n == name && a == args);
        if same {
            self.failures += 1;
        } else {
            self.last = Some((name.to_string(), args.to_string()));
            self.failures = 1;
        }
        self.failures >= MAX_IDENTICAL_FAILURES
    }
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
    guard: &mut RepeatGuard,
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
            &NewMessage::tool(call.id.clone(), call.function.name.clone(), result.clone()),
        )?;
        if guard.tripped(&call.function.name, &call.function.arguments, &result) {
            anyhow::bail!(
                "tool '{}' failed {MAX_IDENTICAL_FAILURES} times in a row with identical \
                 arguments; aborting the run instead of burning the rest of the budget",
                call.function.name
            );
        }
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

/// everything one run needs that is not a call out to the model or a tool
struct Cycle<'a> {
    db: &'a Db,
    session_id: &'a str,
    system_prompt: &'a str,
    limits: RunLimits,
    names: &'a HashSet<&'a str>,
    sink: &'a EventSink,
}

impl Cycle<'_> {
    /// the agent loop, with the model and tool calls injected so tests can count
    /// model calls and cut the client off partway through
    async fn drive<M, MFut, S, SFut, E, EFut>(&self, call_model: M, summarize: S, execute: E)
    where
        M: Fn(Vec<ChatMessage>) -> MFut,
        MFut: Future<Output = Result<Turn>>,
        S: Fn(Vec<ChatMessage>) -> SFut,
        SFut: Future<Output = Result<String>>,
        E: Fn(String, String) -> EFut,
        EFut: Future<Output = String>,
    {
        let started = Instant::now();
        let mut calls = 0;
        let mut guard = RepeatGuard::default();

        loop {
            // nobody is reading, so another model call would only burn tokens
            if self.sink.disconnected() {
                warn!("client left, stopping the run after {calls} model calls");
                return;
            }
            if let Some(exhausted) = self.limits.exhausted(calls, started.elapsed()) {
                return self.sink.fail(exhausted).await;
            }
            calls += 1;
            let messages =
                match assemble(self.db, self.session_id, self.system_prompt, &summarize).await {
                    Ok(messages) => messages,
                    Err(err) => return self.sink.fail(err).await,
                };
            let turn = match call_model(messages).await {
                Ok(turn) => turn,
                Err(err) => return self.sink.fail(err).await,
            };
            let turn = crate::salvage::salvage_turn(turn, self.names);
            match execute_turn(
                self.db,
                self.session_id,
                &turn,
                &execute,
                self.sink,
                &mut guard,
            )
            .await
            {
                Ok(true) => continue,
                Ok(false) => return self.sink.send(Event::Done).await,
                Err(err) => return self.sink.fail(err).await,
            }
        }
    }
}

async fn agent_loop(state: &AppState, session_id: &str, req: &RunRequest, sink: &EventSink) {
    let tools = match state.catalog.tools().await {
        Ok(tools) => tools,
        Err(err) => return sink.fail(err).await,
    };
    let tools = Arc::new(tools);
    let names = tool_names(&tools);
    // captured once, so switching profiles mid-run cannot swap the client underneath
    let client = state.models.active_client();

    let cycle = Cycle {
        db: &state.db,
        session_id,
        system_prompt: &req.system_prompt,
        limits: state.limits,
        names: &names,
        sink,
    };
    cycle
        .drive(
            |messages| {
                let client = client.clone();
                let tools = tools.clone();
                async move { client.chat(&messages, &tools).await }
            },
            |older| {
                let client = client.clone();
                async move { summarize_with_model(&client, older).await }
            },
            |name, args| {
                let catalog = state.catalog.clone();
                async move { catalog.execute(&name, &args).await }
            },
        )
        .await;
}

/// kills the run task when the response body is dropped. a client disconnect
/// drops the body, and dropping the task drops the in-flight model request, so
/// the server stops generating instead of finishing an answer nobody wants.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

struct GuardedStream {
    inner: ReceiverStream<Result<String, Infallible>>,
    _guard: AbortOnDrop,
}

impl Stream for GuardedStream {
    type Item = Result<String, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

pub async fn post_run(State(state): State<AppState>, Json(req): Json<RunRequest>) -> Response {
    let (tx, rx) = mpsc::channel::<Result<String, Infallible>>(64);
    let sink = EventSink::new(tx);

    let task = tokio::spawn(async move {
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
        Body::from_stream(GuardedStream {
            inner: ReceiverStream::new(rx),
            _guard: AbortOnDrop(task),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::testing::TempDb;
    use crate::llm::FunctionCall;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

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

    #[tokio::test]
    async fn the_sink_notices_the_client_leaving() {
        let (sink, rx) = sink_and_events();
        assert!(!sink.disconnected());
        drop(rx);
        assert!(sink.disconnected());
    }

    /// the viewer's abort button drops the ndjson stream. the loop must not keep
    /// calling the model after that, which is what burned minutes of generation.
    #[tokio::test]
    async fn a_client_that_leaves_stops_the_run_before_the_next_model_call() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        let (sink, rx) = sink_and_events();
        let names: HashSet<&str> = ["echo"].into_iter().collect();

        let calls = Arc::new(AtomicUsize::new(0));
        let client = Arc::new(std::sync::Mutex::new(Some(rx)));
        let cycle = Cycle {
            db: &temp.db,
            session_id: &session.id,
            system_prompt: "you are a test",
            limits: RunLimits::default(),
            names: &names,
            sink: &sink,
        };

        cycle
            .drive(
                |_messages| {
                    let calls = calls.clone();
                    let client = client.clone();
                    async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        // the viewer goes away while the model is answering
                        client.lock().expect("client mutex").take();
                        // a tool call, so the loop would otherwise come round again
                        Ok(Turn {
                            text: None,
                            tool_calls: vec![call("c1", "echo", "{}")],
                        })
                    }
                },
                |_older| async { unreachable!("history is too short to summarize") },
                |_name, _args| async { "echoed".to_string() },
            )
            .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the loop kept calling the model after the client left"
        );
    }

    /// the same loop keeps going while the client is still reading
    #[tokio::test]
    async fn a_connected_client_runs_until_the_call_cap() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        let (sink, _rx) = sink_and_events();
        let names: HashSet<&str> = ["echo"].into_iter().collect();

        let calls = Arc::new(AtomicUsize::new(0));
        let cycle = Cycle {
            db: &temp.db,
            session_id: &session.id,
            system_prompt: "you are a test",
            limits: RunLimits {
                max_model_calls: 3,
                budget: Duration::from_secs(900),
            },
            names: &names,
            sink: &sink,
        };

        cycle
            .drive(
                |_messages| {
                    let calls = calls.clone();
                    async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        Ok(Turn {
                            text: None,
                            tool_calls: vec![call("c1", "echo", "{}")],
                        })
                    }
                },
                |_older| async { unreachable!("history is too short to summarize") },
                |_name, _args| async { "echoed".to_string() },
            )
            .await;

        assert_eq!(calls.load(Ordering::SeqCst), 3);
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
        let turn = Turn {
            text: Some("all done".into()),
            tool_calls: Vec::new(),
        };

        let (sink, rx) = sink_and_events();
        let more = execute_turn(
            &temp.db,
            &session.id,
            &turn,
            |_name, _args| async { unreachable!("no tools in this response") },
            &sink,
            &mut RepeatGuard::default(),
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
        let turn = Turn {
            text: None,
            tool_calls: vec![
                call("c1", "list_files", r#"{"path":"/data"}"#),
                call("c2", "stat", "{}"),
            ],
        };

        let (sink, rx) = sink_and_events();
        let more = execute_turn(
            &temp.db,
            &session.id,
            &turn,
            |name, args| async move { format!("{name}:{args}") },
            &sink,
            &mut RepeatGuard::default(),
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
        let turn = Turn {
            text: Some("looking it up".into()),
            tool_calls: vec![call("c1", "search", r#"{"q":"roads"}"#)],
        };

        let (sink, rx) = sink_and_events();
        execute_turn(
            &temp.db,
            &session.id,
            &turn,
            |_name, _args| async { "found nothing".to_string() },
            &sink,
            &mut RepeatGuard::default(),
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

    #[tokio::test]
    async fn a_thrice_repeated_failing_call_aborts_the_run() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        let turn = Turn {
            text: None,
            tool_calls: vec![
                call("c1", "emit_ui_spec", r#"{"ui_type":"map"}"#),
                call("c2", "emit_ui_spec", r#"{"ui_type":"map"}"#),
                call("c3", "emit_ui_spec", r#"{"ui_type":"map"}"#),
            ],
        };

        let (sink, rx) = sink_and_events();
        let err = execute_turn(
            &temp.db,
            &session.id,
            &turn,
            |_name, _args| async { "ERROR: a map spec needs at least one layer".to_string() },
            &sink,
            &mut RepeatGuard::default(),
        )
        .await
        .unwrap_err();
        drop(rx);

        assert!(err.to_string().contains("3 times in a row"), "{err}");
        // the failing returns are still in history so the next run can see them
        let stored = temp.db.messages_after(&session.id, 0).unwrap();
        assert_eq!(stored.iter().filter(|m| m.role == "tool").count(), 3);
    }

    #[test]
    fn the_guard_resets_on_success_or_different_arguments() {
        let mut guard = RepeatGuard::default();
        // success in between resets the streak
        assert!(!guard.tripped("t", "{}", "ERROR: x"));
        assert!(!guard.tripped("t", "{}", "ok"));
        assert!(!guard.tripped("t", "{}", "ERROR: x"));
        // changing arguments is adapting, not looping
        assert!(!guard.tripped("t", r#"{"a":1}"#, "ERROR: x"));
        assert!(!guard.tripped("t", r#"{"a":2}"#, "ERROR: x"));
        assert!(!guard.tripped("t", r#"{"a":2}"#, "ERROR: x"));
        assert!(guard.tripped("t", r#"{"a":2}"#, "ERROR: x"));
        // emoji-style tool failures count too
        let mut guard = RepeatGuard::default();
        assert!(!guard.tripped("t", "{}", "❌ boom"));
        assert!(!guard.tripped("t", "{}", "❌ boom"));
        assert!(guard.tripped("t", "{}", "❌ boom"));
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
