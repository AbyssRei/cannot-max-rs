use crate::error::AppResult;
use crate::engine::battle_simulator::{BattleSimulator, IBattleSimulator};
use crate::types::{MonsterDef, MonteCarloResult, Winner};

pub trait IMonteCarloEngine: Send + Sync {
    fn simulate(
        &self,
        left: &[MonsterDef],
        right: &[MonsterDef],
        samples: u64,
        threads: usize,
    ) -> AppResult<MonteCarloResult>;
}

pub struct MonteCarloEngine;

impl MonteCarloEngine {
    pub fn new() -> Self {
        Self
    }
}

impl IMonteCarloEngine for MonteCarloEngine {
    fn simulate(
        &self,
        left: &[MonsterDef],
        right: &[MonsterDef],
        samples: u64,
        threads: usize,
    ) -> AppResult<MonteCarloResult> {
        let effective_samples = if samples == 0 { 100 } else { samples };
        let effective_threads = if threads == 0 { 1 } else { threads };
        let per_thread = effective_samples / effective_threads as u64;
        let remainder = effective_samples % effective_threads as u64;
        let results: Vec<Winner> = (0..effective_threads)
            .flat_map(|t| {
                let count = per_thread + if (t as u64) < remainder { 1 } else { 0 };
                (0..count)
                    .filter_map(|_| {
                        let mut sim = BattleSimulator::new();
                        sim.initialize(left, right).ok()?;
                        let result = sim.run_until_end(5000).ok()?;
                        Some(result.winner)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let left_wins = results.iter().filter(|w| **w == Winner::Left).count() as f64;
        let right_wins = results.iter().filter(|w| **w == Winner::Right).count() as f64;
        let total = results.len() as f64;
        if total < 1e-10 {
            return Ok(MonteCarloResult {
                left_win_rate: 0.0,
                right_win_rate: 0.0,
                total_samples: 0,
            });
        }
        Ok(MonteCarloResult {
            left_win_rate: left_wins / total,
            right_win_rate: right_wins / total,
            total_samples: results.len() as u64,
        })
    }
}
