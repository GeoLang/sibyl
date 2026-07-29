//! cross-session memory: a small fact store the model edits through two
//! native tools. facts are injected into every run's system context, so
//! recall needs no tool call and works on the first turn of a new session.

use serde_json::{Value, json};

use crate::db::Db;

pub const SAVE_MEMORY: &str = "save_memory";
pub const FORGET_MEMORY: &str = "forget_memory";

/// bound on injected facts so memory can never crowd out the conversation
pub const MAX_INJECTED_MEMORIES: usize = 50;

pub fn tool_defs() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": SAVE_MEMORY,
                "description": "Save a lasting fact about the user or their work (a preference, region of interest, or project context) so future sessions remember it. Use when the user says 'remember ...' or states a durable preference. Not for facts about the current analysis only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "One short sentence stating the fact."
                        }
                    },
                    "required": ["content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": FORGET_MEMORY,
                "description": "Delete saved memories containing the given text. Use when the user asks to forget something.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "matching": {
                            "type": "string",
                            "description": "Text contained in the memory to delete."
                        }
                    },
                    "required": ["matching"]
                }
            }
        }),
    ]
}

/// handles memory tools locally; None means the call is for the executor
pub fn execute(db: &Db, name: &str, raw_args: &str) -> Option<String> {
    match name {
        SAVE_MEMORY => Some(save(db, raw_args)),
        FORGET_MEMORY => Some(forget(db, raw_args)),
        _ => None,
    }
}

fn string_arg(raw: &str, key: &str) -> Option<String> {
    serde_json::from_str::<Value>(raw)
        .ok()?
        .get(key)?
        .as_str()
        .map(str::to_string)
}

fn save(db: &Db, raw_args: &str) -> String {
    match string_arg(raw_args, "content").filter(|c| !c.trim().is_empty()) {
        Some(content) => match db.add_memory(content.trim()) {
            Ok(_) => format!("Saved to memory: {}", content.trim()),
            Err(err) => format!("❌ could not save memory: {err}"),
        },
        None => "❌ save_memory needs a non-empty 'content' string".into(),
    }
}

fn forget(db: &Db, raw_args: &str) -> String {
    match string_arg(raw_args, "matching").filter(|m| !m.trim().is_empty()) {
        Some(needle) => match db.delete_memories_matching(needle.trim()) {
            Ok(0) => format!("No saved memory contains '{}'.", needle.trim()),
            Ok(n) => format!("Forgot {n} memor{}.", if n == 1 { "y" } else { "ies" }),
            Err(err) => format!("❌ could not delete memories: {err}"),
        },
        None => "❌ forget_memory needs a non-empty 'matching' string".into(),
    }
}

/// the system message carrying saved facts, None when there are none
pub fn context_block(db: &Db) -> Option<String> {
    let memories = db.list_memories(MAX_INJECTED_MEMORIES).ok()?;
    if memories.is_empty() {
        return None;
    }
    let lines: Vec<String> = memories
        .iter()
        .map(|m| format!("- {}", m.content))
        .collect();
    Some(format!(
        "Persistent memory (facts saved in earlier sessions):\n{}",
        lines.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::testing::TempDb;

    #[test]
    fn save_then_inject_then_forget_round_trip() {
        let temp = TempDb::new();

        let saved = execute(
            &temp.db,
            SAVE_MEMORY,
            r#"{"content":"user works in Lisbon"}"#,
        );
        assert_eq!(
            saved.as_deref(),
            Some("Saved to memory: user works in Lisbon")
        );

        let block = context_block(&temp.db).expect("memories should inject");
        assert!(block.contains("- user works in Lisbon"));

        let forgot = execute(&temp.db, FORGET_MEMORY, r#"{"matching":"Lisbon"}"#);
        assert_eq!(forgot.as_deref(), Some("Forgot 1 memory."));
        assert!(context_block(&temp.db).is_none());
    }

    #[test]
    fn bad_arguments_return_errors_not_panics() {
        let temp = TempDb::new();
        assert!(
            execute(&temp.db, SAVE_MEMORY, "{}")
                .unwrap()
                .starts_with("❌")
        );
        assert!(
            execute(&temp.db, SAVE_MEMORY, "not json")
                .unwrap()
                .starts_with("❌")
        );
        assert!(
            execute(&temp.db, FORGET_MEMORY, "{}")
                .unwrap()
                .starts_with("❌")
        );
    }

    #[test]
    fn unknown_tools_pass_through() {
        let temp = TempDb::new();
        assert!(execute(&temp.db, "geocode_place", "{}").is_none());
    }

    #[test]
    fn forgetting_something_never_saved_is_not_an_error() {
        let temp = TempDb::new();
        let result = execute(&temp.db, FORGET_MEMORY, r#"{"matching":"unicorns"}"#).unwrap();
        assert!(result.starts_with("No saved memory"));
    }
}
