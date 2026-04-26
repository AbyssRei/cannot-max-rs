use crate::capture::capture_frame;
use crate::config::AppConfig;
use crate::core::{AnalysisOutput, BattleSnapshot, CaptureCatalog, CaptureSource, CapturedFrame, Side};
use crate::field_recognition::FieldRecognizer;
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
        let mut snapshot = analyze_frame(source, &frame, config.roi, &resources, config);

        // 场地识别：如果配置了场地特征数，调用FieldRecognizer识别场地元素
        if config.field_feature_count > 0 {
            let field_recognizer = FieldRecognizer::new(&config.field_model_path);
            if field_recognizer.is_ready() {
                let field_data = field_recognizer.recognize_field_elements(&frame.image);
                let feature_columns = field_recognizer.get_feature_columns();
                let mut terrain_features = vec![0.0f32; config.field_feature_count];
                for (i, col) in feature_columns.iter().enumerate() {
                    if i >= config.field_feature_count { break; }
                    if let Some(&val) = field_data.get(col) {
                        terrain_features[i] = val;
                    }
                }
                snapshot.terrain_features = terrain_features;
                // 根据场地特征推断地形名称
                if !field_data.is_empty() {
                    let detected: Vec<&str> = field_data.iter()
                        .filter(|(_, v)| **v > 0.0)
                        .map(|(k, _)| k.as_str())
                        .collect();
                    snapshot.terrain_name = if detected.is_empty() {
                        None
                    } else {
                        Some(detected.join("+"))
                    };
                }
            }
        }

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

    /// 从已有的 BattleSnapshot 进行预测（用于手动阵容输入）
    pub fn predict_from_snapshot(
        &self,
        snapshot: &BattleSnapshot,
        config: &AppConfig,
    ) -> Result<AnalysisOutput, String> {
        let predictor = CandlePredictor::new(self.model_path.clone());
        let prediction = predictor.predict(snapshot)?;

        let handler = SpecialMonsterHandler::new();
        let left_ids: Vec<String> = snapshot.units.iter()
            .filter(|u| u.side == Side::Left).map(|u| u.unit_id.clone()).collect();
        let right_ids: Vec<String> = snapshot.units.iter()
            .filter(|u| u.side == Side::Right).map(|u| u.unit_id.clone()).collect();
        let special_messages = handler.check_special_monsters(&left_ids, &right_ids, &prediction.winner);

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

        // 创建一个空的 CapturedFrame 用于输出
        let frame = CapturedFrame {
            image: image::RgbaImage::from_pixel(
                snapshot.frame_size.0,
                snapshot.frame_size.1,
                image::Rgba([0, 0, 0, 255]),
            ),
            note: "手动阵容输入".to_string(),
        };

        Ok(AnalysisOutput {
            frame,
            snapshot: snapshot.clone(),
            prediction,
            special_messages,
            history_results,
        })
    }
}
