//! per-session memory: a small fact store the model edits through two
//! native tools. facts are injected into that session's system context, so
//! recall needs no tool call on later turns.

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
                "description": "Save a lasting fact about the user or their work (a preference, region of interest, or project context) so later turns in this session remember it. Use when the user says 'remember ...' or states a durable preference. Not for facts about the current analysis only.",
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
                "description": "Delete saved memories containing the given text from this session. Use when the user asks to forget something.",
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
pub fn execute(db: &Db, session_id: &str, name: &str, raw_args: &str) -> Option<String> {
    match name {
        SAVE_MEMORY => Some(save(db, session_id, raw_args)),
        FORGET_MEMORY => Some(forget(db, session_id, raw_args)),
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

fn save(db: &Db, session_id: &str, raw_args: &str) -> String {
    match string_arg(raw_args, "content").filter(|c| !c.trim().is_empty()) {
        Some(content) => match db.add_memory(session_id, content.trim()) {
            Ok(_) => format!("Saved to memory: {}", content.trim()),
            Err(err) => format!("❌ could not save memory: {err}"),
        },
        None => "❌ save_memory needs a non-empty 'content' string".into(),
    }
}

fn forget(db: &Db, session_id: &str, raw_args: &str) -> String {
    match string_arg(raw_args, "matching").filter(|m| !m.trim().is_empty()) {
        Some(needle) => match db.delete_memories_matching(session_id, needle.trim()) {
            Ok(0) => format!("No saved memory contains '{}'.", needle.trim()),
            Ok(n) => format!("Forgot {n} memor{}.", if n == 1 { "y" } else { "ies" }),
            Err(err) => format!("❌ could not delete memories: {err}"),
        },
        None => "❌ forget_memory needs a non-empty 'matching' string".into(),
    }
}

/// the system message carrying saved facts, None when there are none
pub fn context_block(db: &Db, session_id: &str) -> Option<String> {
    let memories = db.list_memories(session_id, MAX_INJECTED_MEMORIES).ok()?;
    if memories.is_empty() {
        return None;
    }
    let lines: Vec<String> = memories
        .iter()
        .map(|m| format!("- {}", m.content))
        .collect();
    Some(format!(
        "Persistent memory (facts saved earlier in this session):\n{}",
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
        let session = temp.db.create_session("chat").unwrap();

        let saved = execute(
            &temp.db,
            &session.id,
            SAVE_MEMORY,
            r#"{"content":"user works in Lisbon"}"#,
        );
        assert_eq!(
            saved.as_deref(),
            Some("Saved to memory: user works in Lisbon")
        );

        let block = context_block(&temp.db, &session.id).expect("memories should inject");
        assert!(block.contains("- user works in Lisbon"));

        let forgot = execute(
            &temp.db,
            &session.id,
            FORGET_MEMORY,
            r#"{"matching":"Lisbon"}"#,
        );
        assert_eq!(forgot.as_deref(), Some("Forgot 1 memory."));
        assert!(context_block(&temp.db, &session.id).is_none());
    }

    #[test]
    fn memories_stay_inside_their_session() {
        let temp = TempDb::new();
        let a = temp.db.create_session("a").unwrap();
        let b = temp.db.create_session("b").unwrap();

        execute(
            &temp.db,
            &a.id,
            SAVE_MEMORY,
            r#"{"content":"user's study area is Lisbon"}"#,
        );
        execute(
            &temp.db,
            &b.id,
            SAVE_MEMORY,
            r#"{"content":"user's study area is Porto"}"#,
        );

        let block_a = context_block(&temp.db, &a.id).expect("session a should inject");
        assert!(block_a.contains("Lisbon"));
        assert!(!block_a.contains("Porto"));

        let block_b = context_block(&temp.db, &b.id).expect("session b should inject");
        assert!(block_b.contains("Porto"));
        assert!(!block_b.contains("Lisbon"));

        let forgot = execute(&temp.db, &a.id, FORGET_MEMORY, r#"{"matching":"Lisbon"}"#);
        assert_eq!(forgot.as_deref(), Some("Forgot 1 memory."));
        assert!(context_block(&temp.db, &a.id).is_none());
        let block_b = context_block(&temp.db, &b.id).expect("session b should keep its memory");
        assert!(block_b.contains("Porto"));
    }

    #[test]
    fn bad_arguments_return_errors_not_panics() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        assert!(
            execute(&temp.db, &session.id, SAVE_MEMORY, "{}")
                .unwrap()
                .starts_with("❌")
        );
        assert!(
            execute(&temp.db, &session.id, SAVE_MEMORY, "not json")
                .unwrap()
                .starts_with("❌")
        );
        assert!(
            execute(&temp.db, &session.id, FORGET_MEMORY, "{}")
                .unwrap()
                .starts_with("❌")
        );
    }

    #[test]
    fn unknown_tools_pass_through() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        assert!(execute(&temp.db, &session.id, "geocode_place", "{}").is_none());
    }

    #[test]
    fn forgetting_something_never_saved_is_not_an_error() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        let result = execute(
            &temp.db,
            &session.id,
            FORGET_MEMORY,
            r#"{"matching":"unicorns"}"#,
        )
        .unwrap();
        assert!(result.starts_with("No saved memory"));
    }
}
