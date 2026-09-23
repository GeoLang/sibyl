use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use time::OffsetDateTime;

use crate::db::Db;
use crate::llm::Usage;

pub const LIMIT_ENV: &str = "SIBYL_MONTHLY_SPEND_LIMIT_USD";
pub const PRICES_ENV: &str = "SIBYL_MODEL_PRICES";

pub const SPENT_MESSAGE: &str = "The model budget for this month is used up. Try again next month.";

const TOKENS_PER_MILLION: f64 = 1_000_000.0;
const ENTRY_SEPARATOR: char = ',';
const MODEL_SEPARATOR: char = '=';
const INPUT_OUTPUT_SEPARATOR: char = '/';

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    pub input_usd_per_million: f64,
    pub output_usd_per_million: f64,
}

impl Price {
    fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_usd_per_million
            + output_tokens as f64 * self.output_usd_per_million)
            / TOKENS_PER_MILLION
    }
}

pub fn parse_prices(raw: &str) -> Result<HashMap<String, Price>> {
    raw.split(ENTRY_SEPARATOR)
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let shape = || format!("{PRICES_ENV} entry {entry:?} is not model=input/output");
            let (model, rates) = entry.split_once(MODEL_SEPARATOR).with_context(shape)?;
            let (input, output) = rates
                .split_once(INPUT_OUTPUT_SEPARATOR)
                .with_context(shape)?;
            let rate = |raw: &str| raw.trim().parse::<f64>().with_context(shape);
            let price = Price {
                input_usd_per_million: rate(input)?,
                output_usd_per_million: rate(output)?,
            };
            Ok((model.trim().to_string(), price))
        })
        .collect()
}

fn current_month() -> String {
    let now = OffsetDateTime::now_utc();
    format!("{:04}-{:02}", now.year(), u8::from(now.month()))
}

#[derive(Debug)]
pub struct Charge {
    month: String,
    usd: f64,
}

pub struct SpendCap {
    db: Arc<Db>,
    monthly_limit_usd: f64,
    prices: HashMap<String, Price>,
    month: fn() -> String,
}

impl SpendCap {
    pub fn from_env(
        db: Arc<Db>,
        limit: Option<String>,
        prices: Option<String>,
    ) -> Result<Option<Arc<Self>>> {
        let (limit, prices) = match (limit, prices) {
            (None, None) => return Ok(None),
            (Some(limit), Some(prices)) => (limit, prices),
            (Some(_), None) => {
                bail!("{LIMIT_ENV} is set, so {PRICES_ENV} must price the models it caps")
            }
            (None, Some(_)) => bail!("{PRICES_ENV} is set without {LIMIT_ENV}"),
        };
        let monthly_limit_usd = limit
            .parse()
            .with_context(|| format!("{LIMIT_ENV} must be a number of dollars"))?;
        Ok(Some(Arc::new(Self {
            db,
            monthly_limit_usd,
            prices: parse_prices(&prices)?,
            month: current_month,
        })))
    }

    fn price(&self, model: &str) -> Result<Price> {
        self.prices.get(model).copied().with_context(|| {
            format!("{model} has no price in {PRICES_ENV}, so the spend cap refuses it")
        })
    }

    // charged before the call so a call cut off by the client leaving still counts
    pub fn charge_estimate(&self, model: &str, estimated_input_tokens: u64) -> Result<Charge> {
        let price = self.price(model)?;
        let month = (self.month)();
        if self.db.month_spend(&month)? >= self.monthly_limit_usd {
            bail!(SPENT_MESSAGE);
        }
        let usd = price.cost(estimated_input_tokens, 0);
        self.db.add_spend(&month, usd)?;
        Ok(Charge { month, usd })
    }

    pub fn settle(&self, model: &str, charge: Charge, usage: Usage) -> Result<()> {
        let actual = self
            .price(model)?
            .cost(usage.prompt_tokens, usage.completion_tokens);
        self.db.add_spend(&charge.month, actual - charge.usd)
    }
}

#[cfg(test)]
pub mod testing {
    use super::*;

    pub const MODEL: &str = "openai.gpt-oss-120b";
    pub const MONTH: &str = "2026-09";

    pub fn cap(db: Arc<Db>, monthly_limit_usd: f64) -> Arc<SpendCap> {
        Arc::new(SpendCap {
            db,
            monthly_limit_usd,
            prices: parse_prices(&format!("{MODEL}=0.15/0.60")).unwrap(),
            month: || MONTH.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{MODEL, MONTH, cap};
    use super::*;
    use crate::db::testing::TempDb;

    const CLOSE_ENOUGH: f64 = 1e-12;

    #[test]
    fn prices_parse_per_model() {
        let prices =
            parse_prices(" openai.gpt-oss-120b=0.15/0.60, qwen.qwen3-235b-a22b-2507=0.22/0.88 ")
                .unwrap();
        assert_eq!(
            prices["qwen.qwen3-235b-a22b-2507"],
            Price {
                input_usd_per_million: 0.22,
                output_usd_per_million: 0.88,
            }
        );
        assert_eq!(prices.len(), 2);
    }

    #[test]
    fn a_malformed_price_names_the_entry() {
        for raw in ["gpt=0.15", "gpt", "gpt=cheap/0.6"] {
            let err = parse_prices(raw).unwrap_err().to_string();
            assert!(err.contains(PRICES_ENV), "{raw}: {err}");
        }
    }

    #[test]
    fn the_limit_and_the_prices_come_as_a_pair() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        assert!(
            SpendCap::from_env(db.clone(), None, None)
                .unwrap()
                .is_none()
        );
        assert!(SpendCap::from_env(db.clone(), Some("50".into()), None).is_err());
        assert!(SpendCap::from_env(db.clone(), None, Some("m=1/1".into())).is_err());
        assert!(SpendCap::from_env(db, Some("fifty".into()), Some("m=1/1".into())).is_err());
    }

    #[test]
    fn the_estimate_is_corrected_to_the_reported_usage() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        let cap = cap(db.clone(), 50.0);
        let charge = cap.charge_estimate(MODEL, 1_000_000).unwrap();
        assert!((db.month_spend(MONTH).unwrap() - 0.15).abs() < CLOSE_ENOUGH);
        cap.settle(
            MODEL,
            charge,
            Usage {
                prompt_tokens: 2_000_000,
                completion_tokens: 1_000_000,
            },
        )
        .unwrap();
        assert!((db.month_spend(MONTH).unwrap() - 0.90).abs() < CLOSE_ENOUGH);
    }

    #[test]
    fn a_spent_month_refuses_the_next_call() {
        let temp = TempDb::new();
        let db = Arc::new(temp.reopen());
        let cap = cap(db.clone(), 1.0);
        db.add_spend(MONTH, 1.0).unwrap();
        let err = cap.charge_estimate(MODEL, 10).unwrap_err();
        assert_eq!(err.to_string(), SPENT_MESSAGE);
        assert!((db.month_spend(MONTH).unwrap() - 1.0).abs() < CLOSE_ENOUGH);
    }

    #[test]
    fn the_spend_survives_a_restart() {
        let temp = TempDb::new();
        cap(Arc::new(temp.reopen()), 50.0)
            .charge_estimate(MODEL, 1_000_000)
            .unwrap();
        assert!((temp.reopen().month_spend(MONTH).unwrap() - 0.15).abs() < CLOSE_ENOUGH);
    }

    #[test]
    fn an_unpriced_model_is_refused() {
        let temp = TempDb::new();
        let err = cap(Arc::new(temp.reopen()), 50.0)
            .charge_estimate("grok-4", 10)
            .unwrap_err();
        assert!(err.to_string().contains("grok-4"), "{err}");
    }
}
