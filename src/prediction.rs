use crate::core::{BattleSnapshot, PredictionResult, Side, Winner};
use candle_core::{Device, Tensor};
use std::path::PathBuf;

pub trait Predictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String>;
}

#[derive(Debug, Clone)]
pub struct CandlePredictor {
    pub model_path: PathBuf,
    model_loaded: bool,
}

impl CandlePredictor {
    pub fn new(model_path: PathBuf) -> Self {
        let model_loaded = model_path.exists();
        if !model_loaded {
            eprintln!(
                "模型文件不存在: {}，将使用基线预测",
                model_path.display()
            );
        }
        Self {
            model_path,
            model_loaded,
        }
    }
}

impl Predictor for CandlePredictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String> {
        if self.model_loaded {
            // Attempt real model inference; fall back to baseline on any error
            match self.predict_with_model(snapshot) {
                Ok(result) => return Ok(result),
                Err(e) => eprintln!("模型推理失败，回退到基线: {e}"),
            }
        }
        self.predict_baseline(snapshot)
    }
}

impl CandlePredictor {
    fn predict_with_model(
        &self,
        snapshot: &BattleSnapshot,
    ) -> Result<PredictionResult, String> {
        let device = Device::Cpu;

        let left: Vec<f32> = snapshot
            .units
            .iter()
            .filter(|unit| unit.side == Side::Left)
            .map(|unit| unit.count as f32)
            .collect();
        let right: Vec<f32> = snapshot
            .units
            .iter()
            .filter(|unit| unit.side == Side::Right)
            .map(|unit| unit.count as f32)
            .collect();

        // Build feature vector from units + terrain
        let mut features = left.clone();
        features.extend_from_slice(&snapshot.terrain_features);
        let left_len = features.len();
        features.extend_from_slice(&right);
        features.extend_from_slice(&snapshot.terrain_features);
        let _right_len = features.len() - left_len;

        let tensor = Tensor::from_vec(features.clone(), (1, features.len()), &device)
            .map_err(|e| e.to_string())?;

        // Placeholder: real UnitAwareTransformer forward pass would go here
        // using VarMap::load_safetensors and model forward
        let _ = tensor;

        // For now, fall back to baseline even when model file exists
        Err("UnitAwareTransformer forward not yet wired".to_string())
    }

    fn predict_baseline(
        &self,
        snapshot: &BattleSnapshot,
    ) -> Result<PredictionResult, String> {
        let device = Device::Cpu;
        let left: Vec<f32> = snapshot
            .units
            .iter()
            .filter(|unit| unit.side == Side::Left)
            .map(|unit| unit.count as f32)
            .collect();
        let right: Vec<f32> = snapshot
            .units
            .iter()
            .filter(|unit| unit.side == Side::Right)
            .map(|unit| unit.count as f32)
            .collect();

        let left_total = if left.is_empty() {
            0.0
        } else {
            Tensor::from_vec(left.clone(), (left.len(),), &device)
                .map_err(|error| error.to_string())?
                .sum_all()
                .map_err(|error| error.to_string())?
                .to_scalar::<f32>()
                .map_err(|error| error.to_string())?
        };
        let right_total = if right.is_empty() {
            0.0
        } else {
            Tensor::from_vec(right.clone(), (right.len(),), &device)
                .map_err(|error| error.to_string())?
                .sum_all()
                .map_err(|error| error.to_string())?
                .to_scalar::<f32>()
                .map_err(|error| error.to_string())?
        };

        let delta = right_total - left_total;
        let probability = sigmoid(delta / (left_total + right_total + 1.0));
        let left_win_rate = (1.0 - probability).clamp(0.0, 1.0);
        let right_win_rate = probability.clamp(0.0, 1.0);
        let confidence_band = (right_win_rate - 0.5).abs() * 2.0;

        let winner = if (left_win_rate - right_win_rate).abs() < 0.1 {
            Winner::TossUp
        } else if left_win_rate > right_win_rate {
            Winner::Left
        } else {
            Winner::Right
        };

        Ok(PredictionResult {
            left_win_rate,
            right_win_rate,
            winner,
            confidence_band,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BaselinePredictor;

impl BaselinePredictor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BaselinePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Predictor for BaselinePredictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String> {
        let left_total: f32 = snapshot
            .units
            .iter()
            .filter(|unit| unit.side == Side::Left)
            .map(|unit| unit.count as f32)
            .sum();
        let right_total: f32 = snapshot
            .units
            .iter()
            .filter(|unit| unit.side == Side::Right)
            .map(|unit| unit.count as f32)
            .sum();

        let delta = right_total - left_total;
        let probability = sigmoid(delta / (left_total + right_total + 1.0));
        let left_win_rate = (1.0 - probability).clamp(0.0, 1.0);
        let right_win_rate = probability.clamp(0.0, 1.0);
        let confidence_band = (right_win_rate - 0.5).abs() * 2.0;

        let winner = if (left_win_rate - right_win_rate).abs() < 0.1 {
            Winner::TossUp
        } else if left_win_rate > right_win_rate {
            Winner::Left
        } else {
            Winner::Right
        };

        Ok(PredictionResult {
            left_win_rate,
            right_win_rate,
            winner,
            confidence_band,
        })
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::{BaselinePredictor, CandlePredictor, Predictor};
    use crate::core::{BattleSnapshot, CaptureSource};
    use std::path::PathBuf;

    #[test]
    fn prediction_rates_are_complementary() {
        let predictor = CandlePredictor::new(PathBuf::from("dummy"));
        let snapshot = BattleSnapshot {
            source: CaptureSource::Monitor(1),
            frame_size: (100, 100),
            roi: None,
            units: Vec::new(),
            terrain_features: Vec::new(),
        };

        let result = predictor.predict(&snapshot).unwrap();
        assert!((result.left_win_rate + result.right_win_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn baseline_predictor_rates_are_complementary() {
        let predictor = BaselinePredictor::new();
        let snapshot = BattleSnapshot {
            source: CaptureSource::Monitor(1),
            frame_size: (100, 100),
            roi: None,
            units: Vec::new(),
            terrain_features: Vec::new(),
        };

        let result = predictor.predict(&snapshot).unwrap();
        assert!((result.left_win_rate + result.right_win_rate - 1.0).abs() < 0.001);
    }
}
