use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub active: bool,
    #[serde(skip)]
    pub summary: Option<String>,
    #[serde(skip)]
    pub summary_watermark: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredMessage {
    pub id: i64,
    pub role: String,
    pub content: Option<String>,
    /// raw json array of tool calls as the model sent them
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

/// message to append, without the row id
#[derive(Debug, Clone, Default)]
pub struct NewMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

impl NewMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            ..Default::default()
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Option<String>) -> Self {
        Self {
            role: "assistant".into(),
            content,
            tool_calls,
            ..Default::default()
        }
    }

    pub fn tool(tool_call_id: String, name: String, content: String) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content),
            tool_call_id: Some(tool_call_id),
            name: Some(name),
            ..Default::default()
        }
    }
}

pub struct Db {
    conn: Mutex<Connection>,
}

pub fn now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("rfc3339 formatting")
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0,
                summary TEXT,
                summary_watermark INTEGER
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                name TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("db mutex poisoned")
    }

    pub fn list_sessions(&self) -> Result<Vec<Session>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT id, name, created_at, active, summary, summary_watermark
             FROM sessions ORDER BY created_at DESC, rowid DESC",
        )?;
        let rows = stmt
            .query_map([], row_to_session)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn get_session(&self, id: &str) -> Result<Option<Session>> {
        let conn = self.conn();
        let session = conn
            .query_row(
                "SELECT id, name, created_at, active, summary, summary_watermark
                 FROM sessions WHERE id = ?1",
                params![id],
                row_to_session,
            )
            .optional()?;
        Ok(session)
    }

    pub fn active_session(&self) -> Result<Option<Session>> {
        let conn = self.conn();
        let session = conn
            .query_row(
                "SELECT id, name, created_at, active, summary, summary_watermark
                 FROM sessions WHERE active = 1 LIMIT 1",
                [],
                row_to_session,
            )
            .optional()?;
        Ok(session)
    }

    /// creates a session and makes it the only active one
    pub fn create_session(&self, name: &str) -> Result<Session> {
        let session = Session {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            created_at: now(),
            active: true,
            summary: None,
            summary_watermark: 0,
        };
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        tx.execute("UPDATE sessions SET active = 0", [])?;
        tx.execute(
            "INSERT INTO sessions (id, name, created_at, active, summary, summary_watermark)
             VALUES (?1, ?2, ?3, 1, NULL, 0)",
            params![session.id, session.name, session.created_at],
        )?;
        tx.commit()?;
        Ok(session)
    }

    pub fn activate_session(&self, id: &str) -> Result<Option<Session>> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        let updated = tx.execute("UPDATE sessions SET active = 1 WHERE id = ?1", params![id])?;
        if updated == 0 {
            return Ok(None);
        }
        tx.execute("UPDATE sessions SET active = 0 WHERE id != ?1", params![id])?;
        tx.commit()?;
        drop(conn);
        self.get_session(id)
    }

    pub fn rename_session(&self, id: &str, name: &str) -> Result<Option<Session>> {
        let updated = self.conn().execute(
            "UPDATE sessions SET name = ?2 WHERE id = ?1",
            params![id, name],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        self.get_session(id)
    }

    pub fn delete_session(&self, id: &str) -> Result<bool> {
        let deleted = self
            .conn()
            .execute("DELETE FROM sessions WHERE id = ?1", params![id])?;
        Ok(deleted > 0)
    }

    pub fn append_message(&self, session_id: &str, msg: &NewMessage) -> Result<i64> {
        let conn = self.conn();
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, name, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                msg.role,
                msg.content,
                msg.tool_calls,
                msg.tool_call_id,
                msg.name,
                now()
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// history after the summary watermark, oldest first
    pub fn messages_after(&self, session_id: &str, watermark: i64) -> Result<Vec<StoredMessage>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT id, role, content, tool_calls, tool_call_id, name
             FROM messages WHERE session_id = ?1 AND id > ?2 ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![session_id, watermark], |row| {
                Ok(StoredMessage {
                    id: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    tool_calls: row.get(3)?,
                    tool_call_id: row.get(4)?,
                    name: row.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn set_summary(&self, session_id: &str, summary: &str, watermark: i64) -> Result<()> {
        self.conn().execute(
            "UPDATE sessions SET summary = ?2, summary_watermark = ?3 WHERE id = ?1",
            params![session_id, summary, watermark],
        )?;
        Ok(())
    }

    pub fn get_config(&self, key: &str) -> Result<Option<String>> {
        let value = self
            .conn()
            .query_row(
                "SELECT value FROM config WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value)
    }

    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO config (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn add_memory(&self, content: &str) -> Result<i64> {
        let conn = self.conn();
        conn.execute(
            "INSERT INTO memories (content, created_at) VALUES (?1, ?2)",
            params![content, now()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// oldest first, capped so injected context stays bounded
    pub fn list_memories(&self, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT id, content, created_at FROM memories
             ORDER BY id DESC LIMIT ?1",
        )?;
        let mut memories = stmt
            .query_map(params![limit as i64], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    created_at: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        memories.reverse();
        Ok(memories)
    }

    pub fn delete_memories_matching(&self, needle: &str) -> Result<usize> {
        let deleted = self.conn().execute(
            "DELETE FROM memories WHERE content LIKE '%' || ?1 || '%'",
            params![needle],
        )?;
        Ok(deleted)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub created_at: String,
}

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    Ok(Session {
        id: row.get(0)?,
        name: row.get(1)?,
        created_at: row.get(2)?,
        active: row.get::<_, i64>(3)? != 0,
        summary: row.get(4)?,
        summary_watermark: row.get::<_, Option<i64>>(5)?.unwrap_or(0),
    })
}

#[cfg(test)]
pub mod testing {
    use super::Db;
    use std::path::PathBuf;

    /// temp sqlite file removed when the test ends
    pub struct TempDb {
        pub db: Db,
        path: PathBuf,
    }

    impl TempDb {
        pub fn new() -> Self {
            let path = std::env::temp_dir()
                .join("sibyl-tests")
                .join(format!("{}.db", uuid::Uuid::new_v4()));
            let db = Db::open(&path).expect("opening temp db");
            Self { db, path }
        }

        /// opens the same file again, standing in for a restart
        pub fn reopen(&self) -> Db {
            Db::open(&self.path).expect("reopening temp db")
        }
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::TempDb;
    use super::*;

    #[test]
    fn create_activates_and_deactivates_others() {
        let temp = TempDb::new();
        let first = temp.db.create_session("first").unwrap();
        let second = temp.db.create_session("second").unwrap();

        let active = temp.db.active_session().unwrap().unwrap();
        assert_eq!(active.id, second.id);
        assert!(!temp.db.get_session(&first.id).unwrap().unwrap().active);

        let activated = temp.db.activate_session(&first.id).unwrap().unwrap();
        assert!(activated.active);
        assert_eq!(temp.db.active_session().unwrap().unwrap().id, first.id);
        assert!(temp.db.activate_session("missing").unwrap().is_none());
    }

    #[test]
    fn list_is_newest_first_and_rename_sticks() {
        let temp = TempDb::new();
        let first = temp.db.create_session("first").unwrap();
        let second = temp.db.create_session("second").unwrap();

        let listed = temp.db.list_sessions().unwrap();
        assert_eq!(
            listed.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            vec![second.id.as_str(), first.id.as_str()]
        );

        temp.db.rename_session(&first.id, "renamed").unwrap();
        assert_eq!(
            temp.db.get_session(&first.id).unwrap().unwrap().name,
            "renamed"
        );
        assert!(temp.db.rename_session("missing", "x").unwrap().is_none());
    }

    #[test]
    fn delete_removes_session_and_its_messages() {
        let temp = TempDb::new();
        let session = temp.db.create_session("gone").unwrap();
        temp.db
            .append_message(&session.id, &NewMessage::user("hello"))
            .unwrap();

        assert!(temp.db.delete_session(&session.id).unwrap());
        assert!(temp.db.get_session(&session.id).unwrap().is_none());
        assert!(temp.db.messages_after(&session.id, 0).unwrap().is_empty());
        assert!(!temp.db.delete_session(&session.id).unwrap());
    }

    #[test]
    fn messages_append_in_order_and_respect_the_watermark() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        temp.db
            .append_message(&session.id, &NewMessage::user("one"))
            .unwrap();
        let second = temp
            .db
            .append_message(
                &session.id,
                &NewMessage::assistant(Some("two".into()), None),
            )
            .unwrap();
        temp.db
            .append_message(
                &session.id,
                &NewMessage::tool("call-1".into(), "search".into(), "three".into()),
            )
            .unwrap();

        let all = temp.db.messages_after(&session.id, 0).unwrap();
        assert_eq!(
            all.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            vec!["user", "assistant", "tool"]
        );
        assert_eq!(all[2].tool_call_id.as_deref(), Some("call-1"));

        let after = temp.db.messages_after(&session.id, second).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].content.as_deref(), Some("three"));
    }

    #[test]
    fn config_survives_a_restart() {
        let temp = TempDb::new();
        assert_eq!(temp.db.get_config("active_model").unwrap(), None);

        temp.db.set_config("active_model", "local").unwrap();
        temp.db.set_config("active_model", "cloud").unwrap();

        let restarted = temp.reopen();
        assert_eq!(
            restarted.get_config("active_model").unwrap().as_deref(),
            Some("cloud")
        );
        assert_eq!(restarted.get_config("missing").unwrap(), None);
    }

    #[test]
    fn summary_round_trips() {
        let temp = TempDb::new();
        let session = temp.db.create_session("chat").unwrap();
        temp.db.set_summary(&session.id, "earlier work", 7).unwrap();

        let stored = temp.db.get_session(&session.id).unwrap().unwrap();
        assert_eq!(stored.summary.as_deref(), Some("earlier work"));
        assert_eq!(stored.summary_watermark, 7);
    }
}
