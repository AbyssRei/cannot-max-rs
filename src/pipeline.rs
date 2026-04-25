use crate::capture::capture_frame;
use crate::config::AppConfig;
use crate::core::{AnalysisOutput, CaptureCatalog, CaptureSource, Side};
use crate::history_match::HistoryMatch;
use crate::prediction::{CandlePredictor, Predictor};
use crate::recognition::analyze_frame;
use crate::resources::ResourceStore;
use crate::special_monster::SpecialMonsterHandler;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AnalysisPipeline {
    model_path: PathBuf,
    history_csv: Option<PathBuf>,
}

impl AnalysisPipeline {
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path, history_csv: None }
    }

    pub fn with_history(model_path: PathBuf, history_csv: PathBuf) -> Self {
        Self { model_path, history_csv: Some(history_csv) }
    }

    pub fn run(
        &self,
        source: &CaptureSource,
        catalog: &CaptureCatalog,
        config: &AppConfig,
    ) -> Result<AnalysisOutput, String> {
        let resources = ResourceStore::load(config)?;
        let frame = capture_frame(source, config, catalog)?;
        let snapshot = analyze_frame(source, &frame, config.roi, &resources, config);
        let predictor = CandlePredictor::new(self.model_path.clone());
        let prediction = predictor.predict(&snapshot)?;

        // 特殊怪物检测
        let handler = SpecialMonsterHandler::new();
        let left_ids: Vec<String> = snapshot.units.iter()
            .filter(|u| u.side == Side::Left).map(|u| u.unit_id.clone()).collect();
        let right_ids: Vec<String> = snapshot.units.iter()
            .filter(|u| u.side == Side::Right).map(|u| u.unit_id.clone()).collect();
        let special_messages = handler.check_special_monsters(&left_ids, &right_ids, &prediction.winner);

        // 历史匹配
        let history_results = if let Some(ref csv) = self.history_csv {
            match HistoryMatch::new(csv) {
                Ok(hm) => {
                    let mc = config.monster_count;
                    let mut lc = vec![0.0f32; mc];
                    let mut rc = vec![0.0f32; mc];
                    for unit in &snapshot.units {
                        let id: usize = unit.unit_id.parse().unwrap_or(0);
                        if id == 0 || id > mc { continue; }
                        match unit.side {
                            Side::Left => lc[id - 1] = unit.count as f32,
                            Side::Right => rc[id - 1] = unit.count as f32,
                        }
                    }
                    let (indices, win_rate, sim) = hm.render_similar_matches(&lc, &rc);
                    let mut results = Vec::new();
                    if !indices.is_empty() {
                        results.push(format!("历史匹配: {} 条相似对局, 胜率 {:.1}%, 平均相似度 {:.3}", indices.len(), win_rate * 100.0, sim));
                        for (i, &idx) in indices.iter().enumerate().take(5) {
                            results.push(format!("  #{}: 历史记录索引 {}", i + 1, idx));
                        }
                    }
                    results
                }
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };

        Ok(AnalysisOutput {
            frame,
            snapshot,
            prediction,
            special_messages,
            history_results,
        })
    }
}
