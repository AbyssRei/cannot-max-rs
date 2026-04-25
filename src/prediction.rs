use crate::core::{BattleSnapshot, PredictionResult, Side, Winner};
use candle_core::{Device, Tensor};
use std::path::PathBuf;

pub trait Predictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String>;
}

#[derive(Debug, Clone)]
pub struct CandlePredictor {
    pub model_path: PathBuf,
}

impl CandlePredictor {
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }
}

impl Predictor for CandlePredictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String> {
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

        let _ = &self.model_path;

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
    use super::{CandlePredictor, Predictor};
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
}
