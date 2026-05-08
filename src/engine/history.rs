use std::collections::HashMap;
use std::sync::Arc;

use crate::error::AppResult;
use crate::infra::storage_manager::IStorageManager;
use crate::types::{FetchRecord, HistoryMatchResult};

pub trait IHistoryMatcher: Send + Sync {
    fn match_history(&self, query: &[(String, i32)], top_k: usize) -> AppResult<Vec<HistoryMatchResult>>;
}

pub struct HistoryMatcher {
    storage: Arc<dyn IStorageManager>,
    monster_index_map: HashMap<String, usize>,
}

impl HistoryMatcher {
    pub fn new(storage: Arc<dyn IStorageManager>, monster_names: &[String]) -> Self {
        let monster_index_map: HashMap<String, usize> = monster_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        Self {
            storage,
            monster_index_map,
        }
    }

    fn encode_sparse_vector(&self, monsters: &[(String, i32)]) -> Vec<(usize, f64)> {
        let mut vec: Vec<(usize, f64)> = Vec::new();
        for (name, count) in monsters {
            if let Some(&idx) = self.monster_index_map.get(name) {
                vec.push((idx, *count as f64));
            }
        }
        vec.sort_by_key(|(idx, _)| *idx);
        vec
    }

    fn cosine_similarity(a: &[(usize, f64)], b: &[(usize, f64)]) -> f64 {
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for &(_, v) in a {
            norm_a += v * v;
        }
        for &(_, v) in b {
            norm_b += v * v;
        }
        let mut i = 0usize;
        let mut j = 0usize;
        while i < a.len() && j < b.len() {
            if a[i].0 == b[j].0 {
                dot += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            } else if a[i].0 < b[j].0 {
                i += 1;
            } else {
                j += 1;
            }
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            dot / denom
        }
    }

    fn record_to_monsters(record: &FetchRecord) -> Vec<(String, i32)> {
        let mut monsters = record.left_monsters.clone();
        monsters.extend(record.right_monsters.clone());
        monsters
    }
}

impl IHistoryMatcher for HistoryMatcher {
    fn match_history(&self, query: &[(String, i32)], top_k: usize) -> AppResult<Vec<HistoryMatchResult>> {
        let records = self.storage.load_history().unwrap_or_default();
        if records.is_empty() {
            tracing::info!("No history data available");
            return Ok(Vec::new());
        }
        let query_vec = self.encode_sparse_vector(query);
        let mut scored: Vec<(f64, &FetchRecord)> = Vec::new();
        for record in &records {
            let record_monsters = Self::record_to_monsters(record);
            let record_vec = self.encode_sparse_vector(&record_monsters);
            let sim = Self::cosine_similarity(&query_vec, &record_vec);
            scored.push((sim, record));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(sim, record)| HistoryMatchResult {
                left_monsters: record.left_monsters.clone(),
                right_monsters: record.right_monsters.clone(),
                terrain: record.left_terrain.clone(),
                winner: record.winner.clone(),
                similarity: sim,
            })
            .collect())
    }
}
