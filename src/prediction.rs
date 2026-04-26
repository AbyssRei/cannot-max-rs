use crate::core::{BattleSnapshot, PredictionResult, Side, Winner};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::path::PathBuf;

const DEFAULT_MONSTER_COUNT: usize = 60;
const DEFAULT_FIELD_FEATURE_COUNT: usize = 0;

fn se(e: candle_core::Error) -> String { e.to_string() }

pub trait Predictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String>;
    fn is_model_loaded(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct CandlePredictor {
    pub model_path: PathBuf,
    model_loaded: bool,
    monster_count: usize,
    field_feature_count: usize,
}

impl CandlePredictor {
    pub fn new(model_path: PathBuf) -> Self {
        Self::with_counts(model_path, DEFAULT_MONSTER_COUNT, DEFAULT_FIELD_FEATURE_COUNT)
    }

    pub fn with_counts(model_path: PathBuf, monster_count: usize, field_feature_count: usize) -> Self {
        let model_loaded = model_path.exists();
        if !model_loaded {
            eprintln!("模型文件不存在: {}，将使用基线预测", model_path.display());
        }
        Self { model_path, model_loaded, monster_count, field_feature_count }
    }
}

impl Predictor for CandlePredictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String> {
        if self.model_loaded {
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
    fn build_features(&self, snapshot: &BattleSnapshot) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mc = self.monster_count;
        let fc = self.field_feature_count;

        let mut left_signs = vec![0.0f32; mc + fc];
        let mut left_counts = vec![0.0f32; mc + fc];
        let mut right_signs = vec![0.0f32; mc + fc];
        let mut right_counts = vec![0.0f32; mc + fc];

        for unit in &snapshot.units {
            let id: usize = unit.unit_id.parse().unwrap_or(0);
            if id == 0 || id > mc { continue; }
            let count = unit.count as f32;
            match unit.side {
                Side::Left => {
                    left_signs[id - 1] = count.signum();
                    left_counts[id - 1] = count.abs();
                }
                Side::Right => {
                    right_signs[id - 1] = count.signum();
                    right_counts[id - 1] = count.abs();
                }
            }
        }
        // 场地特征部分 sign=1, count=原值（当前禁用，保持0）
        (left_signs, left_counts, right_signs, right_counts)
    }

    fn predict_with_model(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String> {
        let device = Device::Cpu;
        let (ls, lc, rs, rc) = self.build_features(snapshot);
        let tfc = self.monster_count + self.field_feature_count;

        let ls_t = Tensor::from_vec(ls, (1, tfc), &device).map_err(se)?;
        let lc_t = Tensor::from_vec(lc, (1, tfc), &device).map_err(se)?;
        let rs_t = Tensor::from_vec(rs, (1, tfc), &device).map_err(se)?;
        let rc_t = Tensor::from_vec(rc, (1, tfc), &device).map_err(se)?;

        // 加载 safetensors 并检测模型类型
        let tensors = candle_core::safetensors::load(&self.model_path, &device)
            .map_err(|e| format!("加载模型失败: {e}"))?;

        // 检测是否为 Transformer 模型（有 uew 键）
        if tensors.contains_key("uew") {
            return self.predict_transformer(&tensors, &ls_t, &lc_t, &rs_t, &rc_t, tfc);
        }

        // 回退到线性模型（有 w/b 键）
        if let (Some(w), Some(b)) = (tensors.get("w"), tensors.get("b")) {
            let x = Tensor::cat(&[&lc_t, &rc_t], 1).map_err(se)?;
            let logits = x.matmul(w).map_err(se)?.broadcast_add(b).map_err(se)?;
            let prob = sigmoid_tensor(&logits)?.to_scalar::<f32>().map_err(se)?;
            let prob = if prob.is_nan() || prob.is_infinite() { 0.5f32 } else { prob.clamp(0.0, 1.0) };
            return Ok(build_prediction_result(prob));
        }

        Err("无法识别模型格式".to_string())
    }

    fn predict_transformer(
        &self,
        tensors: &std::collections::HashMap<String, Tensor>,
        ls: &Tensor, lc: &Tensor, rs: &Tensor, rc: &Tensor,
        tfc: usize,
    ) -> Result<PredictionResult, String> {
        let device = Device::Cpu;
        // 从 safetensors 构建 VarMap + VarBuilder，再构建模型
        let mut varmap = VarMap::new();
        for (key, tensor) in tensors {
            varmap.set_one(key, tensor).map_err(|e| e.to_string())?;
        }
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // 推断模型参数
        let uew = tensors.get("uew").ok_or("缺少 uew")?;
        let ed = uew.dims()[1];
        // 从键名推断层数
        let nl = (0..10).take_while(|i| tensors.contains_key(&format!("l{i}.eq.weight"))).count();
        let nh = 16; // 默认头数，与训练配置一致

        let model = crate::training::UnitAwareTransformer::new(vb, tfc, ed, nh, nl)?;
        let output = model.forward(ls, lc, rs, rc)?;
        let prob = output.to_scalar::<f32>().map_err(se)?;
        let prob = if prob.is_nan() || prob.is_infinite() { 0.5f32 } else { prob.clamp(0.0, 1.0) };
        Ok(build_prediction_result(prob))
    }

    fn predict_baseline(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String> {
        let left_total: f32 = snapshot.units.iter()
            .filter(|u| u.side == Side::Left).map(|u| u.count as f32).sum();
        let right_total: f32 = snapshot.units.iter()
            .filter(|u| u.side == Side::Right).map(|u| u.count as f32).sum();
        let delta = right_total - left_total;
        let probability = sigmoid(delta / (left_total + right_total + 1.0));
        Ok(build_prediction_result(probability))
    }
}

fn sigmoid_tensor(t: &Tensor) -> Result<Tensor, String> {
    let en = t.neg().map_err(se)?.exp().map_err(se)?;
    let one = Tensor::ones(en.shape(), en.dtype(), en.device()).map_err(se)?;
    one.add(&en).map_err(se)?.recip().map_err(se)
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
    PredictionResult { left_win_rate, right_win_rate, winner, confidence_band }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ── BaselinePredictor ──

#[derive(Debug, Clone)]
pub struct BaselinePredictor;

impl BaselinePredictor { pub fn new() -> Self { Self } }
impl Default for BaselinePredictor { fn default() -> Self { Self::new() } }

impl Predictor for BaselinePredictor {
    fn predict(&self, snapshot: &BattleSnapshot) -> Result<PredictionResult, String> {
        let left_total: f32 = snapshot.units.iter()
            .filter(|u| u.side == Side::Left).map(|u| u.count as f32).sum();
        let right_total: f32 = snapshot.units.iter()
            .filter(|u| u.side == Side::Right).map(|u| u.count as f32).sum();
        let delta = right_total - left_total;
        let probability = sigmoid(delta / (left_total + right_total + 1.0));
        Ok(build_prediction_result(probability))
    }
    fn is_model_loaded(&self) -> bool { false }
}

#[cfg(test)]
mod tests {
    use super::{BaselinePredictor, CandlePredictor, Predictor};
    use crate::core::{BattleSnapshot, CaptureSource};
    use std::path::PathBuf;

    fn empty_snapshot() -> BattleSnapshot {
        BattleSnapshot { source: CaptureSource::Monitor(1), frame_size: (100, 100), roi: None, units: Vec::new(), terrain_features: Vec::new() }
    }

    #[test]
    fn prediction_rates_are_complementary() {
        let r = CandlePredictor::new(PathBuf::from("dummy")).predict(&empty_snapshot()).unwrap();
        assert!((r.left_win_rate + r.right_win_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn baseline_predictor_rates_are_complementary() {
        let r = BaselinePredictor::new().predict(&empty_snapshot()).unwrap();
        assert!((r.left_win_rate + r.right_win_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn baseline_predictor_is_not_model_loaded() {
        assert!(!BaselinePredictor::new().is_model_loaded());
    }

    #[test]
    fn candle_predictor_nonexistent_model_not_loaded() {
        assert!(!CandlePredictor::new(PathBuf::from("nonexistent.safetensors")).is_model_loaded());
    }
}
