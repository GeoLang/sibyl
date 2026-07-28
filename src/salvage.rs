//! Small models sometimes print a tool call as literal text instead of emitting a
//! structured one, leaving raw markup on the user's screen. This recovers those
//! calls, but only from complete, unambiguous blocks: anything else stays text.

use std::collections::HashSet;

use serde_json::{Map, Value};
use tracing::warn;
use uuid::Uuid;

use crate::llm::{FunctionCall, ToolCall, Turn};

/// caps so a pathological response cannot wedge the loop
const MAX_CONTENT_BYTES: usize = 64 * 1024;
const MAX_BLOCKS: usize = 8;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

struct Salvaged {
    name: String,
    arguments: String,
    dialect: &'static str,
}

/// recovers tool calls a model wrote as text. a turn that already carries
/// structured tool calls is returned untouched.
pub fn salvage_turn(turn: Turn, known: &HashSet<&str>) -> Turn {
    if !turn.tool_calls.is_empty() {
        return turn;
    }
    let Some(content) = turn.text.as_deref() else {
        return turn;
    };
    if !content.contains(OPEN) {
        return turn;
    }
    if content.len() > MAX_CONTENT_BYTES {
        warn!("model text carries tool-call markup but is too big to salvage");
        return turn;
    }
    let Some(blocks) = find_blocks(content) else {
        warn!("model text carries more than {MAX_BLOCKS} tool-call blocks, leaving it as text");
        return turn;
    };

    let mut calls = Vec::new();
    let mut text = String::new();
    let mut cursor = 0;
    for (start, end, body) in blocks {
        // an unparseable block, or one naming a tool we were not offered, stays
        // put: we never execute a name outside the manifest
        let Some(call) = parse_block(body).filter(|call| known.contains(call.name.as_str())) else {
            continue;
        };
        warn!(
            "salvaged a {} tool call for {} out of model text",
            call.dialect, call.name
        );
        text.push_str(&content[cursor..start]);
        calls.push(ToolCall {
            id: format!("salvaged_{}", Uuid::new_v4()),
            kind: "function".to_string(),
            function: FunctionCall {
                name: call.name,
                arguments: call.arguments,
            },
        });
        cursor = end;
    }
    if calls.is_empty() {
        return turn;
    }
    text.push_str(&content[cursor..]);
    let text = text.trim();
    Turn {
        text: (!text.is_empty()).then(|| text.to_string()),
        tool_calls: calls,
    }
}

/// spans of every complete `<tool_call>...</tool_call>`, or none when there are
/// too many to be worth trusting. an unterminated block ends the scan, so a
/// half-written call is left as text.
fn find_blocks(content: &str) -> Option<Vec<(usize, usize, &str)>> {
    let mut blocks = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = content[cursor..].find(OPEN) {
        let start = cursor + offset;
        let body_start = start + OPEN.len();
        let Some(offset) = content[body_start..].find(CLOSE) else {
            break;
        };
        let body_end = body_start + offset;
        if blocks.len() == MAX_BLOCKS {
            return None;
        }
        blocks.push((
            start,
            body_end + CLOSE.len(),
            &content[body_start..body_end],
        ));
        cursor = body_end + CLOSE.len();
    }
    Some(blocks)
}

fn parse_block(body: &str) -> Option<Salvaged> {
    let body = body.trim();
    if body.starts_with('{') {
        parse_json(body)
    } else if body.starts_with(FUNCTION_OPEN) {
        parse_xml(body)
    } else {
        None
    }
}

/// hermes style: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
fn parse_json(body: &str) -> Option<Salvaged> {
    let Value::Object(call) = serde_json::from_str::<Value>(body).ok()? else {
        return None;
    };
    let name = call.get("name")?.as_str()?;
    let arguments = match call.get("arguments") {
        None => "{}".to_string(),
        Some(Value::Object(args)) => Value::Object(args.clone()).to_string(),
        // anything else is ambiguous, so leave the block alone
        Some(_) => return None,
    };
    Some(Salvaged {
        name: name.to_string(),
        arguments,
        dialect: "hermes json",
    })
}

/// qwen-coder style: `<function=name><parameter=key>value</parameter></function>`
fn parse_xml(body: &str) -> Option<Salvaged> {
    let (name, mut rest) = body.strip_prefix(FUNCTION_OPEN)?.split_once('>')?;
    if !is_plain_name(name) {
        return None;
    }
    let mut args = Map::new();
    loop {
        let head = rest.trim_start();
        if head.is_empty() {
            break;
        }
        if let Some(tail) = head.strip_prefix(FUNCTION_CLOSE) {
            rest = tail;
            break;
        }
        let (key, tail) = head.strip_prefix(PARAM_OPEN)?.split_once('>')?;
        if !is_plain_name(key) {
            return None;
        }
        let (raw, tail) = tail.split_once(PARAM_CLOSE)?;
        // a repeated key means we misread the block, so don't guess which wins
        if args.insert(key.to_string(), scalar(raw.trim())).is_some() {
            return None;
        }
        rest = tail;
    }
    if !rest.trim().is_empty() {
        return None;
    }
    Some(Salvaged {
        name: name.to_string(),
        arguments: Value::Object(args).to_string(),
        dialect: "qwen xml",
    })
}

fn is_plain_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['<', '>', '\n'])
}

/// xml parameters arrive as text, so recover the scalars that round-trip cleanly
fn scalar(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(value @ (Value::Number(_) | Value::Bool(_))) => value,
        _ => Value::String(raw.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known(names: &[&'static str]) -> HashSet<&'static str> {
        names.iter().copied().collect()
    }

    fn text_turn(content: &str) -> Turn {
        Turn {
            text: Some(content.to_string()),
            tool_calls: Vec::new(),
        }
    }

    fn args(turn: &Turn, index: usize) -> Value {
        serde_json::from_str(&turn.tool_calls[index].function.arguments).expect("valid args")
    }

    /// the exact dialect seen leaking onto a viewtopia screen
    #[test]
    fn salvages_the_qwen_xml_block_from_live_drift() {
        let turn = text_turn(
            "I'll render that as a map.\n\n\
             <tool_call>\n<function=emit_ui_spec>\n\
             <parameter=ui_type>\nmap\n</parameter>\n\
             <parameter=title>\nRainfall by county\n</parameter>\n\
             </function>\n</tool_call>",
        );
        let turn = salvage_turn(turn, &known(&["emit_ui_spec"]));
        assert_eq!(turn.text.as_deref(), Some("I'll render that as a map."));
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.name, "emit_ui_spec");
        assert_eq!(turn.tool_calls[0].kind, "function");
        assert_eq!(
            args(&turn, 0),
            serde_json::json!({"ui_type": "map", "title": "Rainfall by county"})
        );
    }

    #[test]
    fn salvages_the_hermes_json_dialect() {
        let turn = text_turn(
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#,
        );
        let turn = salvage_turn(turn, &known(&["get_weather"]));
        assert_eq!(turn.text, None);
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(args(&turn, 0), serde_json::json!({"city": "Paris"}));
    }

    #[test]
    fn keeps_prose_from_both_sides_of_a_block() {
        let turn = text_turn(
            "Let me look that up.\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>\n\
             I'll summarize once it lands.",
        );
        let turn = salvage_turn(turn, &known(&["get_weather"]));
        assert_eq!(
            turn.text.as_deref(),
            Some("Let me look that up.\n\nI'll summarize once it lands.")
        );
        assert_eq!(args(&turn, 0), serde_json::json!({}));
    }

    #[test]
    fn salvages_several_blocks_in_one_turn() {
        let turn = text_turn(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}</tool_call>",
        );
        let turn = salvage_turn(turn, &known(&["get_weather"]));
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(args(&turn, 0), serde_json::json!({"city": "Paris"}));
        assert_eq!(args(&turn, 1), serde_json::json!({"city": "Berlin"}));
        assert_eq!(turn.text, None);
    }

    #[test]
    fn ids_are_unique_per_salvaged_call() {
        let turn = text_turn(
            "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\
             <tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>",
        );
        let turn = salvage_turn(turn, &known(&["a"]));
        assert_ne!(turn.tool_calls[0].id, turn.tool_calls[1].id);
    }

    #[test]
    fn coerces_only_clean_numbers_and_bools() {
        let turn = text_turn(
            "<tool_call>\n<function=plot>\n\
             <parameter=zoom>\n12\n</parameter>\n\
             <parameter=opacity>\n0.5\n</parameter>\n\
             <parameter=labels>\ntrue\n</parameter>\n\
             <parameter=title>\nmap\n</parameter>\n\
             <parameter=zip>\n007\n</parameter>\n\
             <parameter=version>\n1.2.3\n</parameter>\n\
             </function>\n</tool_call>",
        );
        let turn = salvage_turn(turn, &known(&["plot"]));
        assert_eq!(
            args(&turn, 0),
            serde_json::json!({
                "zoom": 12,
                "opacity": 0.5,
                "labels": true,
                "title": "map",
                "zip": "007",
                "version": "1.2.3",
            })
        );
    }

    #[test]
    fn a_malformed_block_stays_as_text() {
        // no closing marker, so the call was never finished
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>";
        let turn = salvage_turn(text_turn(raw), &known(&["get_weather"]));
        assert_eq!(turn.text.as_deref(), Some(raw));
        assert!(turn.tool_calls.is_empty());
    }

    #[test]
    fn an_unclosed_parameter_stays_as_text() {
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</function>\n</tool_call>";
        let turn = salvage_turn(text_turn(raw), &known(&["get_weather"]));
        assert_eq!(turn.text.as_deref(), Some(raw));
        assert!(turn.tool_calls.is_empty());
    }

    /// a half-written trailing call must not cost us the complete one before it
    #[test]
    fn a_complete_block_survives_an_unterminated_one_after_it() {
        let turn = text_turn(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>\n\
             <tool_call>{\"name\": \"get_weather\", \"argum",
        );
        let turn = salvage_turn(turn, &known(&["get_weather"]));
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(args(&turn, 0), serde_json::json!({"city": "Paris"}));
        assert_eq!(
            turn.text.as_deref(),
            Some(r#"<tool_call>{"name": "get_weather", "argum"#)
        );
    }

    #[test]
    fn broken_json_stays_as_text() {
        let raw = r#"<tool_call>{"name": "get_weather", "arguments": {"city":}</tool_call>"#;
        let turn = salvage_turn(text_turn(raw), &known(&["get_weather"]));
        assert_eq!(turn.text.as_deref(), Some(raw));
        assert!(turn.tool_calls.is_empty());
    }

    #[test]
    fn a_tool_outside_the_manifest_stays_as_text() {
        let raw = r#"<tool_call>{"name": "rm_rf", "arguments": {"path": "/"}}</tool_call>"#;
        let turn = salvage_turn(text_turn(raw), &known(&["get_weather"]));
        assert_eq!(turn.text.as_deref(), Some(raw));
        assert!(turn.tool_calls.is_empty());
    }

    /// one good block still salvages while an unknown one is left on screen
    #[test]
    fn mixes_a_salvaged_block_with_an_unknown_one() {
        let turn = text_turn(
            "<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
        );
        let turn = salvage_turn(turn, &known(&["get_weather"]));
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            turn.text.as_deref(),
            Some(r#"<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>"#)
        );
    }

    /// the cloud path: a structured call must come through completely untouched
    #[test]
    fn a_structured_turn_is_never_rewritten() {
        let original = Turn {
            text: Some(
                "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>".into(),
            ),
            tool_calls: vec![ToolCall {
                id: "call_123".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "list_datasets".into(),
                    arguments: "{}".into(),
                },
            }],
        };
        assert_eq!(
            salvage_turn(original.clone(), &known(&["get_weather", "list_datasets"])),
            original
        );
    }

    #[test]
    fn ordinary_prose_is_left_alone() {
        let turn = text_turn("The weather in Paris is 11 degrees C.");
        assert_eq!(
            salvage_turn(turn.clone(), &known(&["get_weather"])),
            turn.clone()
        );
    }

    /// prose that merely talks about the markup must not be eaten
    #[test]
    fn prose_without_a_complete_block_is_left_alone() {
        let turn = text_turn("Wrap the call in <tool_call> markers when you emit it.");
        assert_eq!(salvage_turn(turn.clone(), &known(&["get_weather"])), turn);
    }

    #[test]
    fn too_many_blocks_are_left_as_text() {
        let block = r#"<tool_call>{"name": "a", "arguments": {}}</tool_call>"#;
        let flood = block.repeat(MAX_BLOCKS + 1);
        let turn = salvage_turn(text_turn(&flood), &known(&["a"]));
        assert_eq!(turn.text.as_deref(), Some(flood.as_str()));
        assert!(turn.tool_calls.is_empty());

        let at_cap = block.repeat(MAX_BLOCKS);
        let turn = salvage_turn(text_turn(&at_cap), &known(&["a"]));
        assert_eq!(turn.tool_calls.len(), MAX_BLOCKS);
    }

    #[test]
    fn oversized_content_is_left_as_text() {
        let huge = format!(
            "{}<tool_call>{{\"name\": \"a\", \"arguments\": {{}}}}</tool_call>",
            "x".repeat(MAX_CONTENT_BYTES)
        );
        let turn = salvage_turn(text_turn(&huge), &known(&["a"]));
        assert!(turn.tool_calls.is_empty());
        assert_eq!(turn.text.as_deref(), Some(huge.as_str()));
    }
}
