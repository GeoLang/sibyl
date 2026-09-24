use std::sync::Arc;

use anyhow::{Result, bail};
use time::OffsetDateTime;

use crate::db::Db;
use crate::llm::Usage;

pub const RUNS_ENV: &str = "SIBYL_RUNS_PER_USER_PER_DAY";
pub const TOKENS_ENV: &str = "SIBYL_TOKENS_PER_USER_PER_DAY";

pub const RUNS_USED_MESSAGE: &str = "You have used today's runs. Try again tomorrow.";
pub const TOKENS_USED_MESSAGE: &str = "Today's model budget is used up. Try again tomorrow.";

fn current_day() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}-{:02}-{:02}",
        now.year(),
        u8::from(now.month()),
        now.day()
    )
}

pub struct DailyLimits {
    db: Arc<Db>,
    runs_per_day: Option<u64>,
    tokens_per_day: Option<u64>,
    day: fn() -> String,
}

impl DailyLimits {
    pub fn new(
        db: Arc<Db>,
        runs_per_day: Option<u64>,
        tokens_per_day: Option<u64>,
    ) -> Option<Arc<Self>> {
        if runs_per_day.is_none() && tokens_per_day.is_none() {
            return None;
        }
        Some(Arc::new(Self {
            db,
            runs_per_day,
            tokens_per_day,
            day: current_day,
        }))
    }

    pub fn count_run(&self, subject: &str) -> Result<()> {
        let Some(runs_per_day) = self.runs_per_day else {
            return Ok(());
        };
        if !self.db.count_run(subject, &(self.day)(), runs_per_day)? {
            bail!(RUNS_USED_MESSAGE);
        }
        Ok(())
    }

    pub fn tokens_for(self: &Arc<Self>, subject: &str) -> Option<UserTokens> {
        let tokens_per_day = self.tokens_per_day?;
        Some(UserTokens {
            limits: self.clone(),
            subject: subject.to_string(),
            tokens_per_day,
        })
    }
}

#[derive(Debug)]
pub struct TokenCharge {
    day: String,
    tokens: i64,
}

pub struct UserTokens {
    limits: Arc<DailyLimits>,
    subject: String,
    tokens_per_day: u64,
}

impl UserTokens {
    // charged before the call so a call cut off by the client leaving still counts
    pub fn charge_estimate(&self, estimated_input_tokens: u64) -> Result<TokenCharge> {
        let day = (self.limits.day)();
        if self.limits.db.day_tokens(&self.subject, &day)? >= self.tokens_per_day as i64 {
            bail!(TOKENS_USED_MESSAGE);
        }
        let tokens = estimated_input_tokens as i64;
        self.limits.db.add_tokens(&self.subject, &day, tokens)?;
        Ok(TokenCharge { day, tokens })
    }

    pub fn settle(&self, charge: TokenCharge, usage: Usage) -> Result<()> {
        let actual = (usage.prompt_tokens + usage.completion_tokens) as i64;
        self.limits
            .db
            .add_tokens(&self.subject, &charge.day, actual - charge.tokens)
    }
}

#[cfg(test)]
pub mod testing {
    use super::*;

    pub const DAY: &str = "2026-09-23";

    pub fn limits(
        db: Arc<Db>,
        runs_per_day: Option<u64>,
        tokens_per_day: Option<u64>,
    ) -> Arc<DailyLimits> {
        Arc::new(DailyLimits {
            db,
            runs_per_day,
            tokens_per_day,
            day: || DAY.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{DAY, limits};
    use super::*;
    use crate::db::testing::TempDb;

    #[test]
    fn no_limit_set_means_no_limits_at_all() {
        let temp = TempDb::new();
        assert!(DailyLimits::new(Arc::new(temp.reopen()), None, None).is_none());
        let runs_only = DailyLimits::new(Arc::new(temp.reopen()), Some(40), None).unwrap();
        assert!(runs_only.tokens_for("alice").is_none());
    }

    #[test]
    fn a_refused_run_does_not_count() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        let limits = limits(db.clone(), Some(1), None);
        limits.count_run("alice").unwrap();
        let err = limits.count_run("alice").unwrap_err();
        assert_eq!(err.to_string(), RUNS_USED_MESSAGE);
        assert!(
            db.count_run("alice", DAY, 2).unwrap(),
            "the refused run took a slot"
        );
    }

    #[test]
    fn each_user_has_their_own_runs() {
        let temp = TempDb::new();
        let limits = limits(Arc::new(temp.reopen()), Some(1), None);
        limits.count_run("alice").unwrap();
        limits.count_run("bob").unwrap();
        assert!(limits.count_run("alice").is_err());
    }

    #[test]
    fn the_run_count_survives_a_restart() {
        let temp = TempDb::new();
        limits(Arc::new(temp.reopen()), Some(1), None)
            .count_run("alice")
            .unwrap();
        let restarted = limits(Arc::new(temp.reopen()), Some(1), None);
        assert_eq!(
            restarted.count_run("alice").unwrap_err().to_string(),
            RUNS_USED_MESSAGE
        );
    }

    #[test]
    fn the_token_estimate_is_corrected_to_the_reported_usage() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        let tokens = limits(db.clone(), None, Some(2_000_000))
            .tokens_for("alice")
            .unwrap();
        let charge = tokens.charge_estimate(1_000).unwrap();
        assert_eq!(db.day_tokens("alice", DAY).unwrap(), 1_000);
        tokens
            .settle(
                charge,
                Usage {
                    prompt_tokens: 1_500,
                    completion_tokens: 200,
                },
            )
            .unwrap();
        assert_eq!(db.day_tokens("alice", DAY).unwrap(), 1_700);
        assert_eq!(db.day_tokens("bob", DAY).unwrap(), 0);
    }

    #[test]
    fn a_used_up_day_refuses_the_next_call_without_charging_it() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        db.add_tokens("alice", DAY, 2_000_000).unwrap();
        let err = limits(db.clone(), None, Some(2_000_000))
            .tokens_for("alice")
            .unwrap()
            .charge_estimate(10)
            .unwrap_err();
        assert_eq!(err.to_string(), TOKENS_USED_MESSAGE);
        assert_eq!(db.day_tokens("alice", DAY).unwrap(), 2_000_000);
    }
}
