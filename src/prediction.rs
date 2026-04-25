use crate::core::{BattleSnapshot, PredictionResult, Side, Winner};
use candle_core::{Device, Tensor};
use std::path::PathBuf;

pub trait Predictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String>;
    fn is_model_loaded(&self) -> bool;
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

    fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }
}

impl CandlePredictor {
    fn predict_with_model(
        &self,
        snapshot: &BattleSnapshot,
    ) -> Result<PredictionResult, String> {
        let device = Device::Cpu;

        // 构建左右单位特征向量
        let monster_count = 60; // Greenvine 赛季
        let field_feature_count = 0; // 当前禁用场地特征

        let mut left_features = vec![0.0f32; monster_count];
        let mut right_features = vec![0.0f32; monster_count];

        for unit in &snapshot.units {
            let id: usize = unit.unit_id.parse().unwrap_or(0);
            if id == 0 || id > monster_count {
                continue;
            }
            match unit.side {
                Side::Left => left_features[id - 1] = unit.count as f32,
                Side::Right => right_features[id - 1] = unit.count as f32,
            }
        }

        // 构建完整特征向量: [left_monster, left_field, right_monster, right_field]
        let mut features = left_features.clone();
        features.extend_from_slice(&vec![0.0f32; field_feature_count]);
        features.extend_from_slice(&right_features);
        features.extend_from_slice(&vec![0.0f32; field_feature_count]);

        let total_features = monster_count + field_feature_count;

        // 构建输入张量: (1, total_features * 2)
        let tensor = Tensor::from_vec(features.clone(), (1, total_features * 2), &device)
            .map_err(|e| e.to_string())?;

        // 尝试加载 safetensors 模型并执行前向传播
        if let Ok(tensors) = candle_core::safetensors::load(&self.model_path, &device) {
            // 尝试从加载的张量中获取线性层权重
            if let (Some(w_tensor), Some(b_tensor)) = (tensors.get("w"), tensors.get("b")) {
                // 线性层前向传播: sigmoid(x * w + b)
                let logits = tensor.matmul(w_tensor).map_err(|e| e.to_string())?;
                let logits = logits.broadcast_add(b_tensor).map_err(|e| e.to_string())?;
                let output = sigmoid_tensor(&logits)?;

                let probability = output
                    .to_scalar::<f32>()
                    .map_err(|e| e.to_string())?;

                let probability = if probability.is_nan() || probability.is_infinite() {
                    0.5f32
                } else {
                    probability.clamp(0.0, 1.0)
                };

                return Ok(build_prediction_result(probability));
            }
        }

        // 模型加载或推理失败，回退到基线
        Err("模型权重加载或前向传播失败".to_string())
    }

    fn predict_baseline(
        &self,
        snapshot: &BattleSnapshot,
    ) -> Result<PredictionResult, String> {
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

        Ok(build_prediction_result(probability))
    }
}

fn sigmoid_tensor(t: &Tensor) -> Result<Tensor, String> {
    let neg = t.neg().map_err(|e| e.to_string())?;
    let exp_neg = neg.exp().map_err(|e| e.to_string())?;
    let one = Tensor::ones(exp_neg.shape(), exp_neg.dtype(), exp_neg.device()).map_err(|e| e.to_string())?;
    let denom = one.add(&exp_neg).map_err(|e| e.to_string())?;
    let one2 = Tensor::ones(denom.shape(), denom.dtype(), denom.device()).map_err(|e| e.to_string())?;
    one2.div(&denom).map_err(|e| e.to_string())
}

fn build_prediction_result(probability: f32) -> PredictionResult {
    let probability = probability.clamp(0.0, 1.0);
    let left_win_rate = 1.0 - probability;
    let right_win_rate = probability;
    let confidence_band = (right_win_rate - 0.5).abs() * 2.0;

    let winner = if (left_win_rate - right_win_rate).abs() < 0.1 {
        Winner::TossUp
    } else if left_win_rate > right_win_rate {
        Winner::Left
    } else {
        Winner::Right
    };

    PredictionResult {
        left_win_rate,
        right_win_rate,
        winner,
        confidence_band,
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

        Ok(build_prediction_result(probability))
    }

    fn is_model_loaded(&self) -> bool {
        false
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

    #[test]
    fn baseline_predictor_is_not_model_loaded() {
        let predictor = BaselinePredictor::new();
        assert!(!predictor.is_model_loaded());
    }

    #[test]
    fn candle_predictor_nonexistent_model_not_loaded() {
        let predictor = CandlePredictor::new(PathBuf::from("nonexistent.safetensors"));
        assert!(!predictor.is_model_loaded());
    }
}
