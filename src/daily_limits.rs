use std::sync::Arc;

use anyhow::{Result, bail};
use time::OffsetDateTime;

use crate::db::Db;
use crate::llm::Usage;

pub const RUNS_ENV: &str = "SIBYL_RUNS_PER_USER_PER_DAY";
pub const TOKENS_ENV: &str = "SIBYL_TOKENS_PER_USER_PER_DAY";
pub const ADMIN_RUNS_ENV: &str = "SIBYL_RUNS_PER_ADMIN_PER_DAY";
pub const ADMIN_TOKENS_ENV: &str = "SIBYL_TOKENS_PER_ADMIN_PER_DAY";

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

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Allowance {
    pub runs_per_day: Option<u64>,
    pub tokens_per_day: Option<u64>,
}

impl Allowance {
    fn is_unlimited(&self) -> bool {
        self.runs_per_day.is_none() && self.tokens_per_day.is_none()
    }
}

pub struct DailyLimits {
    db: Arc<Db>,
    users: Allowance,
    admins: Allowance,
    day: fn() -> String,
}

impl DailyLimits {
    pub fn new(db: Arc<Db>, users: Allowance, admins: Allowance) -> Option<Arc<Self>> {
        if users.is_unlimited() && admins.is_unlimited() {
            return None;
        }
        Some(Arc::new(Self {
            db,
            users,
            admins,
            day: current_day,
        }))
    }

    fn allowance(&self, admin: bool) -> Allowance {
        if admin { self.admins } else { self.users }
    }

    pub fn count_run(&self, subject: &str, admin: bool) -> Result<()> {
        let Some(runs_per_day) = self.allowance(admin).runs_per_day else {
            return Ok(());
        };
        if !self.db.count_run(subject, &(self.day)(), runs_per_day)? {
            bail!(RUNS_USED_MESSAGE);
        }
        Ok(())
    }

    pub fn tokens_for(self: &Arc<Self>, subject: &str, admin: bool) -> Option<UserTokens> {
        let tokens_per_day = self.allowance(admin).tokens_per_day?;
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
        let allowance = Allowance {
            runs_per_day,
            tokens_per_day,
        };
        tiered_limits(db, allowance, allowance)
    }

    pub fn tiered_limits(db: Arc<Db>, users: Allowance, admins: Allowance) -> Arc<DailyLimits> {
        Arc::new(DailyLimits {
            db,
            users,
            admins,
            day: || DAY.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{DAY, limits, tiered_limits};
    use super::*;
    use crate::db::testing::TempDb;

    #[test]
    fn no_limit_set_means_no_limits_at_all() {
        let temp = TempDb::new();
        let unlimited = Allowance::default();
        assert!(DailyLimits::new(Arc::new(temp.reopen()), unlimited, unlimited).is_none());
        let runs_only = Allowance {
            runs_per_day: Some(40),
            tokens_per_day: None,
        };
        let limits = DailyLimits::new(Arc::new(temp.reopen()), runs_only, runs_only).unwrap();
        assert!(limits.tokens_for("alice", false).is_none());
    }

    #[test]
    fn a_refused_run_does_not_count() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        let limits = limits(db.clone(), Some(1), None);
        limits.count_run("alice", false).unwrap();
        let err = limits.count_run("alice", false).unwrap_err();
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
        limits.count_run("alice", false).unwrap();
        limits.count_run("bob", false).unwrap();
        assert!(limits.count_run("alice", false).is_err());
    }

    #[test]
    fn the_run_count_survives_a_restart() {
        let temp = TempDb::new();
        limits(Arc::new(temp.reopen()), Some(1), None)
            .count_run("alice", false)
            .unwrap();
        let restarted = limits(Arc::new(temp.reopen()), Some(1), None);
        assert_eq!(
            restarted.count_run("alice", false).unwrap_err().to_string(),
            RUNS_USED_MESSAGE
        );
    }

    #[test]
    fn the_token_estimate_is_corrected_to_the_reported_usage() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        let tokens = limits(db.clone(), None, Some(2_000_000))
            .tokens_for("alice", false)
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
            .tokens_for("alice", false)
            .unwrap()
            .charge_estimate(10)
            .unwrap_err();
        assert_eq!(err.to_string(), TOKENS_USED_MESSAGE);
        assert_eq!(db.day_tokens("alice", DAY).unwrap(), 2_000_000);
    }

    #[test]
    fn an_admin_runs_on_the_admin_allowance() {
        let temp = TempDb::new();
        let limits = tiered_limits(
            Arc::new(temp.reopen()),
            Allowance {
                runs_per_day: Some(1),
                tokens_per_day: Some(10),
            },
            Allowance {
                runs_per_day: Some(3),
                tokens_per_day: Some(1_000),
            },
        );
        for _ in 0..3 {
            limits.count_run("owner", true).unwrap();
        }
        assert!(limits.count_run("owner", true).is_err());
        limits.count_run("alice", false).unwrap();
        assert!(limits.count_run("alice", false).is_err());
        let admin_tokens = limits.tokens_for("owner", true).unwrap();
        admin_tokens.charge_estimate(500).unwrap();
        admin_tokens.charge_estimate(400).unwrap();
    }
}
