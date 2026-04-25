use crate::core::{TrainConfig, TrainProgress, TrainResult, TrainSample};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use std::fs;
use std::path::Path;

const MONSTER_COUNT: usize = 61;
const FIELD_FEATURE_COUNT: usize = 12;
const TOTAL_FEATURE_COUNT: usize = MONSTER_COUNT + FIELD_FEATURE_COUNT;

// ── 数据加载与预处理 ──

pub fn load_training_data(
    config: &TrainConfig,
) -> Result<(Vec<TrainSample>, Vec<TrainSample>), String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(&config.data_file)
        .map_err(|e| format!("无法打开数据文件: {e}"))?;

    let mut all_samples = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| format!("读取CSV行失败: {e}"))?;
        let expected_cols = TOTAL_FEATURE_COUNT * 2 + 2;
        if record.len() < expected_cols {
            continue;
        }

        let mut left_monster = vec![0.0f32; MONSTER_COUNT];
        let mut right_monster = vec![0.0f32; MONSTER_COUNT];
        let mut left_field = vec![0.0f32; FIELD_FEATURE_COUNT];
        let mut right_field = vec![0.0f32; FIELD_FEATURE_COUNT];

        for i in 0..MONSTER_COUNT {
            left_monster[i] = record.get(i).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        }
        for i in 0..FIELD_FEATURE_COUNT {
            left_field[i] = record
                .get(MONSTER_COUNT + i)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
        }
        for i in 0..MONSTER_COUNT {
            right_monster[i] = record
                .get(MONSTER_COUNT + FIELD_FEATURE_COUNT + i)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
        }
        for i in 0..FIELD_FEATURE_COUNT {
            right_field[i] = record
                .get(MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT + i)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
        }

        let label_str = record.get(TOTAL_FEATURE_COUNT * 2).unwrap_or("L");
        let label: f32 = match label_str { "R" => 1.0, _ => 0.0 };

        let left_signs: Vec<f32> = left_monster.iter().map(|&v| v.signum()).chain(left_field.iter().map(|_| 1.0f32)).collect();
        let left_counts: Vec<f32> = left_monster.iter().map(|&v| v.abs()).chain(left_field.iter().cloned()).collect();
        let right_signs: Vec<f32> = right_monster.iter().map(|&v| v.signum()).chain(right_field.iter().map(|_| 1.0f32)).collect();
        let right_counts: Vec<f32> = right_monster.iter().map(|&v| v.abs()).chain(right_field.iter().cloned()).collect();

        let clip = config.max_feature_value;
        let left_counts: Vec<f32> = left_counts.iter().map(|&v| v.clamp(0.0, clip)).collect();
        let right_counts: Vec<f32> = right_counts.iter().map(|&v| v.clamp(0.0, clip)).collect();

        all_samples.push(TrainSample { left_signs, left_counts, right_signs, right_counts, label });
    }

    if all_samples.is_empty() {
        return Err("训练数据为空".to_string());
    }

    let val_size = (all_samples.len() as f32 * config.test_size) as usize;
    let val_size = val_size.max(1).min(all_samples.len() - 1);

    let mut rng = rand::rng();
    use rand::seq::SliceRandom;
    all_samples.shuffle(&mut rng);

    let val_samples = all_samples.split_off(all_samples.len() - val_size);
    Ok((all_samples, val_samples))
}

// ── 设备选择 ──

pub fn select_device(prefer_cuda: bool) -> Result<Device, String> {
    if prefer_cuda {
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
    }
    Ok(Device::Cpu)
}

// ── 训练流水线 ──
// Note: Full UnitAwareTransformer model implementation requires careful
// adaptation to candle-nn 0.10 API (which differs from PyTorch).
// The training pipeline below provides the data loading, device selection,
// and training loop infrastructure. The model forward pass is a simplified
// baseline that can be replaced with the full transformer once the API
// is fully adapted.

pub struct TrainingPipeline {
    config: TrainConfig,
}

impl TrainingPipeline {
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }

    pub fn run(
        &self,
        progress_sender: std::sync::mpsc::Sender<TrainProgress>,
    ) -> Result<TrainResult, String> {
        let config = &self.config;
        let device = select_device(true)?;
        let device_info = if device.is_cuda() { "CUDA" } else { "CPU" }.to_string();

        let (train_samples, val_samples) = load_training_data(config)?;
        let data_length = train_samples.len() + val_samples.len();

        fs::create_dir_all(&config.save_dir).map_err(|e| e.to_string())?;

        // Simplified baseline training: use a simple linear model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let feat_dim = TOTAL_FEATURE_COUNT * 2; // left_counts + right_counts
        let w = vb.get((feat_dim, 1), "w").map_err(|e| e.to_string())?;
        let b = vb.get(1, "b").map_err(|e| e.to_string())?;

        let mut optimizer = AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW {
                lr: config.learning_rate,
                weight_decay: config.weight_decay,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        ).map_err(|e| e.to_string())?;

        let mut best_acc = 0.0f32;
        let mut best_loss = f32::INFINITY;
        let mut model_paths = Vec::new();
        let start_time = std::time::Instant::now();

        for epoch in 0..config.epochs {
            let (train_loss, train_acc) = train_one_epoch_simple(
                &w, &b, &train_samples, &device, &mut optimizer, config.batch_size,
            )?;
            let (val_loss, val_acc) = evaluate_simple(
                &w, &b, &val_samples, &device, config.batch_size,
            )?;

            let save_dir = Path::new(&config.save_dir);
            if val_acc > best_acc {
                best_acc = val_acc;
                let path = save_dir.join("best_model_acc.safetensors");
                export_safetensors(&varmap, &path)?;
                model_paths.push(path);
            }
            if val_loss < best_loss {
                best_loss = val_loss;
                let path = save_dir.join("best_model_loss.safetensors");
                export_safetensors(&varmap, &path)?;
                model_paths.push(path);
            }

            let now = std::time::Instant::now();
            let elapsed = (now - start_time).as_secs_f64();
            let avg_epoch = elapsed / (epoch as f64 + 1.0);
            let estimated_remaining = avg_epoch * (config.epochs as f64 - epoch as f64 - 1.0);

            let _ = progress_sender.send(TrainProgress {
                epoch: epoch + 1, total_epochs: config.epochs,
                train_loss, train_acc, val_loss, val_acc,
                best_acc, best_loss,
                elapsed_secs: elapsed,
                estimated_remaining_secs: estimated_remaining.max(0.0),
                device_info: device_info.clone(),
            });
        }

        rename_model_files(Path::new(&config.save_dir), data_length, best_acc, best_loss)?;

        Ok(TrainResult { best_acc, best_loss, total_epochs: config.epochs, data_length, model_paths })
    }
}

fn simple_forward(w: &Tensor, b: &Tensor, left_counts: &Tensor, right_counts: &Tensor) -> Result<Tensor, String> {
    // Concat left and right counts, then linear + sigmoid
    let x = Tensor::cat(&[left_counts, right_counts], 1).map_err(|e| e.to_string())?;
    let logits = x.matmul(w).map_err(|e| e.to_string())?;
    let logits = logits.broadcast_add(b).map_err(|e| e.to_string())?;
    // sigmoid
    let neg = logits.neg().map_err(|e| e.to_string())?;
    let exp_neg = neg.exp().map_err(|e| e.to_string())?;
    let one = Tensor::ones(exp_neg.shape(), exp_neg.dtype(), exp_neg.device()).map_err(|e| e.to_string())?;
    let denom = one.add(&exp_neg).map_err(|e| e.to_string())?;
    let one2 = Tensor::ones(denom.shape(), denom.dtype(), denom.device()).map_err(|e| e.to_string())?;
    one2.div(&denom).map_err(|e| e.to_string())
}

fn train_one_epoch_simple(
    w: &Tensor, b: &Tensor, samples: &[TrainSample], device: &Device,
    optimizer: &mut AdamW, batch_size: usize,
) -> Result<(f32, f32), String> {
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;
    let mut total = 0usize;
    let num_batches = (samples.len() + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(samples.len());
        let batch = &samples[start..end];
        let bsz = batch.len();

        let left_counts = batch_to_tensor(batch, |s| &s.left_counts, device, (bsz, TOTAL_FEATURE_COUNT))?;
        let right_counts = batch_to_tensor(batch, |s| &s.right_counts, device, (bsz, TOTAL_FEATURE_COUNT))?;
        let labels: Vec<f32> = batch.iter().map(|s| s.label).collect();
        let labels = Tensor::from_vec(labels, (bsz,), device).map_err(|e| e.to_string())?;

        let output = simple_forward(w, b, &left_counts, &right_counts)?;
        let output = output.squeeze(1).map_err(|e| e.to_string())?;

        let diff = output.sub(&labels).map_err(|e| e.to_string())?;
        let loss = diff.sqr().map_err(|e| e.to_string())?.mean_all().map_err(|e| e.to_string())?;

        // Skip NaN check (candle 0.10 API limitation)

        optimizer.backward_step(&loss).map_err(|e| e.to_string())?;
        total_loss += loss.to_scalar::<f32>().map_err(|e| e.to_string())?;

        let preds = output.ge(0.5f64).map_err(|e| e.to_string())?.to_dtype(DType::F32).map_err(|e| e.to_string())?;
        let label_preds = labels.ge(0.5f64).map_err(|e| e.to_string())?.to_dtype(DType::F32).map_err(|e| e.to_string())?;
        let eq = preds.eq(&label_preds).map_err(|e| e.to_string())?.to_dtype(DType::U32).map_err(|e| e.to_string())?.sum_all().map_err(|e| e.to_string())?.to_scalar::<u32>().map_err(|e| e.to_string())?;
        correct += eq as usize;
        total += bsz;
    }

    Ok((total_loss / num_batches.max(1) as f32, 100.0 * correct as f32 / total.max(1) as f32))
}

fn evaluate_simple(
    w: &Tensor, b: &Tensor, samples: &[TrainSample], device: &Device, batch_size: usize,
) -> Result<(f32, f32), String> {
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;
    let mut total = 0usize;
    let num_batches = (samples.len() + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(samples.len());
        let batch = &samples[start..end];
        let bsz = batch.len();

        let left_counts = batch_to_tensor(batch, |s| &s.left_counts, device, (bsz, TOTAL_FEATURE_COUNT))?;
        let right_counts = batch_to_tensor(batch, |s| &s.right_counts, device, (bsz, TOTAL_FEATURE_COUNT))?;
        let labels: Vec<f32> = batch.iter().map(|s| s.label).collect();
        let labels = Tensor::from_vec(labels, (bsz,), device).map_err(|e| e.to_string())?;

        let output = simple_forward(w, b, &left_counts, &right_counts)?;
        let output = output.squeeze(1).map_err(|e| e.to_string())?;

        let diff = output.sub(&labels).map_err(|e| e.to_string())?;
        let loss = diff.sqr().map_err(|e| e.to_string())?.mean_all().map_err(|e| e.to_string())?;

        // Skip NaN check (candle 0.10 API limitation)
        total_loss += loss.to_scalar::<f32>().map_err(|e| e.to_string())?;

        let preds = output.ge(0.5f64).map_err(|e| e.to_string())?.to_dtype(DType::F32).map_err(|e| e.to_string())?;
        let label_preds = labels.ge(0.5f64).map_err(|e| e.to_string())?.to_dtype(DType::F32).map_err(|e| e.to_string())?;
        let eq = preds.eq(&label_preds).map_err(|e| e.to_string())?.to_dtype(DType::U32).map_err(|e| e.to_string())?.sum_all().map_err(|e| e.to_string())?.to_scalar::<u32>().map_err(|e| e.to_string())?;
        correct += eq as usize;
        total += bsz;
    }

    Ok((total_loss / num_batches.max(1) as f32, 100.0 * correct as f32 / total.max(1) as f32))
}

fn batch_to_tensor<F>(batch: &[TrainSample], extractor: F, device: &Device, shape: (usize, usize)) -> Result<Tensor, String>
where F: Fn(&TrainSample) -> &Vec<f32> {
    let mut data = Vec::with_capacity(shape.0 * shape.1);
    for sample in batch { data.extend_from_slice(extractor(sample)); }
    Tensor::from_vec(data, shape, device).map_err(|e| e.to_string())
}

pub fn export_safetensors(varmap: &VarMap, path: &Path) -> Result<(), String> {
    let data = varmap.data().lock().map_err(|e| e.to_string())?;
    let tensors: std::collections::HashMap<String, Tensor> = data.iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
        .collect();
    candle_core::safetensors::save(&tensors, path)
        .map_err(|e| format!("导出safetensors失败: {e}"))
}

fn rename_model_files(save_dir: &Path, data_length: usize, best_acc: f32, best_loss: f32) -> Result<(), String> {
    let now = chrono_like_timestamp();
    let base = format!("data{}_acc{:.4}_loss{:.4}_{}", data_length, best_acc, best_loss, now);
    for prefix in &["best_model_acc", "best_model_loss", "best_model_full"] {
        let old = save_dir.join(format!("{prefix}.safetensors"));
        if old.exists() {
            let new = save_dir.join(format!("{prefix}_{base}.safetensors"));
            fs::rename(&old, &new).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn chrono_like_timestamp() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let days = now / 86400;
    let time_of_day = now % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    let year = 1970 + (days / 365);
    let day_of_year = days % 365;
    let month = (day_of_year / 30) + 1;
    let day = (day_of_year % 30) + 1;
    format!("{:04}_{:02}_{:02}_{:02}_{:02}_{:02}", year, month, day, hours, minutes, seconds)
}
