use crate::core::{TrainConfig, TrainProgress, TrainResult, TrainSample};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, VarBuilder, VarMap};
use std::fs;
use std::path::Path;

// ── Dropout 工具函数 ──

/// 在训练时应用 Dropout，推理时直接返回输入
fn apply_dropout(x: &Tensor, dropout: f32, train: bool) -> Result<Tensor, String> {
    if !train || dropout <= 0.0 {
        return Ok(x.clone());
    }
    // 使用 candle_nn::Dropout
    let d = candle_nn::Dropout::new(dropout);
    d.forward(x, train).map_err(|e| e.to_string())
}

const DEFAULT_MONSTER_COUNT: usize = 60;
const DEFAULT_FIELD_FEATURE_COUNT: usize = 0;
const TOPK: usize = 8;

fn se(e: candle_core::Error) -> String { e.to_string() }

// ── 数据加载与预处理 ──

pub fn load_training_data(
    config: &TrainConfig,
    monster_count: usize,
    field_feature_count: usize,
) -> Result<(Vec<TrainSample>, Vec<TrainSample>), String> {
    let total_feature_count = monster_count + field_feature_count;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true).flexible(true)
        .from_path(&config.data_file)
        .map_err(|e| format!("无法打开数据文件: {e}"))?;

    let mut all_samples = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| format!("读取CSV行失败: {e}"))?;
        let expected_cols = total_feature_count * 2 + 2;
        if record.len() < expected_cols { continue; }

        let mut left_monster = vec![0.0f32; monster_count];
        let mut right_monster = vec![0.0f32; monster_count];
        let mut left_field = vec![0.0f32; field_feature_count];
        let mut right_field = vec![0.0f32; field_feature_count];

        for i in 0..monster_count {
            left_monster[i] = record.get(i).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        }
        for i in 0..field_feature_count {
            left_field[i] = record.get(monster_count + i).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        }
        for i in 0..monster_count {
            right_monster[i] = record.get(monster_count + field_feature_count + i).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        }
        for i in 0..field_feature_count {
            right_field[i] = record.get(monster_count + field_feature_count + monster_count + i).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        }

        let label_str = record.get(total_feature_count * 2).unwrap_or("L");
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

    if all_samples.is_empty() { return Err("训练数据为空".to_string()); }

    // 分层随机划分：按胜负标签分组，每组内随机shuffle后按比例划分
    let mut win_samples: Vec<TrainSample> = Vec::new();
    let mut lose_samples: Vec<TrainSample> = Vec::new();
    for sample in all_samples {
        if sample.label > 0.5 {
            win_samples.push(sample);
        } else {
            lose_samples.push(sample);
        }
    }

    let mut rng = rand::rng();
    use rand::seq::SliceRandom;
    win_samples.shuffle(&mut rng);
    lose_samples.shuffle(&mut rng);

    let win_val_size = (win_samples.len() as f32 * config.test_size) as usize;
    let lose_val_size = (lose_samples.len() as f32 * config.test_size) as usize;
    let win_val_size = win_val_size.max(1).min(win_samples.len().saturating_sub(1));
    let lose_val_size = lose_val_size.max(1).min(lose_samples.len().saturating_sub(1));

    let win_val = win_samples.split_off(win_samples.len() - win_val_size);
    let lose_val = lose_samples.split_off(lose_samples.len() - lose_val_size);

    let mut train_samples = win_samples;
    train_samples.extend(lose_samples);
    let mut val_samples = win_val;
    val_samples.extend(lose_val);

    // 整体shuffle训练集
    train_samples.shuffle(&mut rng);

    Ok((train_samples, val_samples))
}

pub fn select_device(prefer_cuda: bool) -> Result<Device, String> {
    if prefer_cuda {
        if let Ok(device) = Device::new_cuda(0) { return Ok(device); }
    }
    Ok(Device::Cpu)
}

// ── TopK via sort_last_dim ──
// candle-core 0.10 没有 topk，用 sort_last_dim(asc=false) + narrow 模拟

fn topk_dim1(tensor: &Tensor, k: usize) -> Result<(Tensor, Tensor), String> {
    // sort_last_dim 返回 (values, indices)，asc=false 为降序
    let (sorted, indices) = tensor.sort_last_dim(false).map_err(se)?;
    let values = sorted.narrow(1, 0, k).map_err(se)?;
    let idx = indices.narrow(1, 0, k).map_err(se)?;
    Ok((values, idx))
}

// ── UnitAwareTransformer ──

struct TransformerLayer {
    eq: Linear, ek: Linear, ev: Linear, eo: Linear, ef1: Linear, ef2: Linear,
    fq: Linear, fk: Linear, fv: Linear, fo: Linear, ff1: Linear, ff2: Linear,
    nw: Tensor, nb: Tensor,
    ed: usize, nh: usize,
}

impl TransformerLayer {
    fn new(vb: VarBuilder, ed: usize, nh: usize) -> Result<Self, String> {
        let mk = |vb: &VarBuilder, n: &str| -> Result<Linear, String> {
            Ok(Linear::new(vb.pp(n).get((ed, ed), "weight").map_err(se)?, None))
        };
        let ml = |vb: &VarBuilder, n: &str, i: usize, o: usize| -> Result<Linear, String> {
            candle_nn::linear(i, o, vb.pp(n)).map_err(se)
        };
        Ok(Self {
            eq: mk(&vb, "eq")?, ek: mk(&vb, "ek")?, ev: mk(&vb, "ev")?, eo: mk(&vb, "eo")?,
            ef1: ml(&vb, "ef1", ed, ed*2)?, ef2: ml(&vb, "ef2", ed*2, ed)?,
            fq: mk(&vb, "fq")?, fk: mk(&vb, "fk")?, fv: mk(&vb, "fv")?, fo: mk(&vb, "fo")?,
            ff1: ml(&vb, "ff1", ed, ed*2)?, ff2: ml(&vb, "ff2", ed*2, ed)?,
            nw: vb.get(ed, "nw").map_err(se)?, nb: vb.get(ed, "nb").map_err(se)?,
            ed, nh,
        })
    }

    fn mha(qp:&Linear, kp:&Linear, vp:&Linear, op:&Linear,
           q:&Tensor, k:&Tensor, v:&Tensor, nh:usize, ed:usize) -> Result<Tensor, String> {
        let hd = ed / nh;
        let (b, sq, _) = q.dims3().map_err(se)?;
        let (_, sk, _) = k.dims3().map_err(se)?;
        let q = qp.forward(q).map_err(se)?.reshape((b,sq,nh,hd)).map_err(se)?.transpose(1,2).map_err(se)?;
        let k = kp.forward(k).map_err(se)?.reshape((b,sk,nh,hd)).map_err(se)?.transpose(1,2).map_err(se)?;
        let v = vp.forward(v).map_err(se)?.reshape((b,sk,nh,hd)).map_err(se)?.transpose(1,2).map_err(se)?;
        let sc = (hd as f64).sqrt();
        let s = q.matmul(&k.transpose(2,3).map_err(se)?).map_err(se)?.affine(1.0/sc, 0.0).map_err(se)?;
        let aw = candle_nn::ops::softmax(&s, 3).map_err(se)?;
        let ao = aw.matmul(&v).map_err(se)?.transpose(1,2).map_err(se)?.reshape((b,sq,ed)).map_err(se)?;
        op.forward(&ao).map_err(se)
    }

    fn ffn(f1:&Linear, f2:&Linear, x:&Tensor) -> Result<Tensor, String> {
        f2.forward(&f1.forward(x).map_err(se)?.relu().map_err(se)?).map_err(se)
    }

    fn ln(&self, x:&Tensor) -> Result<Tensor, String> {
        let m = x.mean_keepdim(2).map_err(se)?;
        let d = x.sub(&m).map_err(se)?;
        let v = d.sqr().map_err(se)?.mean_keepdim(2).map_err(se)?;
        let eps = Tensor::new(1e-5f64, x.device()).map_err(se)?;
        let is_ = v.add(&eps).map_err(se)?.sqrt().map_err(se)?.recip().map_err(se)?;
        d.broadcast_mul(&is_).map_err(se)?.broadcast_mul(&self.nw).map_err(se)?.broadcast_add(&self.nb).map_err(se)
    }

    fn fwd(&self, lf:&Tensor, rf:&Tensor, dropout: f32, train: bool) -> Result<(Tensor,Tensor), String> {
        let (nh,ed) = (self.nh, self.ed);
        // 敌方交叉注意力
        let dl = Self::mha(&self.eq,&self.ek,&self.ev,&self.eo, lf,rf,rf, nh,ed)?;
        let dr = Self::mha(&self.eq,&self.ek,&self.ev,&self.eo, rf,lf,lf, nh,ed)?;
        let dl = apply_dropout(&dl, dropout, train)?;
        let dr = apply_dropout(&dr, dropout, train)?;
        let lf = lf.add(&dl).map_err(se)?;
        let rf = rf.add(&dr).map_err(se)?;
        // 敌方 FFN
        let lf_ffn = Self::ffn(&self.ef1,&self.ef2,&lf)?;
        let lf_ffn = apply_dropout(&lf_ffn, dropout, train)?;
        let rf_ffn = Self::ffn(&self.ef1,&self.ef2,&rf)?;
        let rf_ffn = apply_dropout(&rf_ffn, dropout, train)?;
        let lf = lf.add(&lf_ffn).map_err(se)?;
        let rf = rf.add(&rf_ffn).map_err(se)?;
        // 友方自注意力
        let dl = Self::mha(&self.fq,&self.fk,&self.fv,&self.fo, &lf,&lf,&lf, nh,ed)?;
        let dr = Self::mha(&self.fq,&self.fk,&self.fv,&self.fo, &rf,&rf,&rf, nh,ed)?;
        let dl = apply_dropout(&dl, dropout, train)?;
        let dr = apply_dropout(&dr, dropout, train)?;
        let lf = lf.add(&dl).map_err(se)?;
        let rf = rf.add(&dr).map_err(se)?;
        // 友方 FFN
        let lf_ffn = Self::ffn(&self.ff1,&self.ff2,&lf)?;
        let lf_ffn = apply_dropout(&lf_ffn, dropout, train)?;
        let rf_ffn = Self::ffn(&self.ff1,&self.ff2,&rf)?;
        let rf_ffn = apply_dropout(&rf_ffn, dropout, train)?;
        let lf = lf.add(&lf_ffn).map_err(se)?;
        let rf = rf.add(&rf_ffn).map_err(se)?;
        // LayerNorm
        Ok((self.ln(&lf)?, self.ln(&rf)?))
    }
}

pub struct UnitAwareTransformer {
    uew: Tensor, vf1: Linear, vf2: Linear,
    layers: Vec<TransformerLayer>, fc1: Linear, fc2: Linear,
    nu: usize, ed: usize, nh: usize,
}

impl UnitAwareTransformer {
    pub fn new(vb: VarBuilder, nu:usize, ed:usize, nh:usize, nl:usize) -> Result<Self, String> {
        let uew = vb.get((nu,ed), "uew").map_err(se)?;
        let vf1 = candle_nn::linear(ed, ed*2, vb.pp("vf1")).map_err(se)?;
        let vf2 = candle_nn::linear(ed*2, ed, vb.pp("vf2")).map_err(se)?;
        let mut layers = Vec::new();
        for i in 0..nl { layers.push(TransformerLayer::new(vb.pp(&format!("l{i}")), ed, nh)?); }
        let fc1 = candle_nn::linear(ed, ed*2, vb.pp("fc1")).map_err(se)?;
        let fc2 = candle_nn::linear(ed*2, 1, vb.pp("fc2")).map_err(se)?;
        Ok(Self { uew, vf1, vf2, layers, fc1, fc2, nu, ed, nh })
    }

    pub fn forward(&self, _ls:&Tensor, lc:&Tensor, _rs:&Tensor, rc:&Tensor, dropout: f32, train: bool) -> Result<Tensor, String> {
        let (b, fd) = lc.dims2().map_err(se)?;
        let k = TOPK.min(fd);
        let ed = self.ed;

        // TopK
        let (lv, li) = topk_dim1(lc, k)?;
        let (rv, ri) = topk_dim1(rc, k)?;

        // 嵌入
        let li64 = li.to_dtype(DType::I64).map_err(se)?.flatten_all().map_err(se)?;
        let ri64 = ri.to_dtype(DType::I64).map_err(se)?.flatten_all().map_err(se)?;
        let lf = self.uew.index_select(&li64, 0).map_err(se)?.reshape((b,k,ed)).map_err(se)?;
        let rf = self.uew.index_select(&ri64, 0).map_err(se)?.reshape((b,k,ed)).map_err(se)?;

        // 数值调制: 前半维不变，后半维 *= 数量
        let half = ed / 2;
        let lf1 = lf.narrow(2,0,half).map_err(se)?;
        let lf2 = lf.narrow(2,half,ed-half).map_err(se)?;
        let lve = lv.unsqueeze(2).map_err(se)?.broadcast_as(lf2.shape()).map_err(se)?;
        let lf2 = lf2.broadcast_mul(&lve).map_err(se)?;
        let lf = Tensor::cat(&[&lf1,&lf2], 2).map_err(se)?;

        let rf1 = rf.narrow(2,0,half).map_err(se)?;
        let rf2 = rf.narrow(2,half,ed-half).map_err(se)?;
        let rve = rv.unsqueeze(2).map_err(se)?.broadcast_as(rf2.shape()).map_err(se)?;
        let rf2 = rf2.broadcast_mul(&rve).map_err(se)?;
        let rf = Tensor::cat(&[&rf1,&rf2], 2).map_err(se)?;

        // 值 FFN (残差)
        let lffn = TransformerLayer::ffn(&self.vf1,&self.vf2,&lf)?;
        let rffn = TransformerLayer::ffn(&self.vf1,&self.vf2,&rf)?;
        let mut lf = lf.add(&lffn).map_err(se)?;
        let mut rf = rf.add(&rffn).map_err(se)?;

        // mask
        let lm = lv.ge(0.1f64).map_err(se)?;
        let rm = rv.ge(0.1f64).map_err(se)?;

        // Transformer 层
        for layer in &self.layers {
            let (nl, nr) = layer.fwd(&lf, &rf, dropout, train)?;
            lf = nl; rf = nr;
        }

        // 输出战斗力
        let lp = self.fc1.forward(&lf).map_err(se)?.relu().map_err(se)?;
        let lp = self.fc2.forward(&lp).map_err(se)?.squeeze(2).map_err(se)?;
        let lmf = lm.to_dtype(DType::F32).map_err(se)?;
        let lt = lp.broadcast_mul(&lmf).map_err(se)?.sum(1).map_err(se)?;

        let rp = self.fc1.forward(&rf).map_err(se)?.relu().map_err(se)?;
        let rp = self.fc2.forward(&rp).map_err(se)?.squeeze(2).map_err(se)?;
        let rmf = rm.to_dtype(DType::F32).map_err(se)?;
        let rt = rp.broadcast_mul(&rmf).map_err(se)?.sum(1).map_err(se)?;

        // sigmoid(右 - 左)
        let diff = rt.sub(&lt).map_err(se)?;
        sigmoid(&diff)
    }
}

fn sigmoid(t: &Tensor) -> Result<Tensor, String> {
    let en = t.neg().map_err(se)?.exp().map_err(se)?;
    let one = Tensor::ones(en.shape(), en.dtype(), en.device()).map_err(se)?;
    one.add(&en).map_err(se)?.recip().map_err(se) // 1/(1+exp(-x)) = sigmoid
}

// ── 训练流水线 ──

pub struct TrainingPipeline { config: TrainConfig }

impl TrainingPipeline {
    pub fn new(config: TrainConfig) -> Self { Self { config } }

    pub fn run(&self, tx: std::sync::mpsc::Sender<TrainProgress>) -> Result<TrainResult, String> {
        let cfg = &self.config;
        let device = select_device(true)?;
        let dev_info = if device.is_cuda() { "CUDA" } else { "CPU" }.to_string();

        let (train, val) = load_training_data(cfg, DEFAULT_MONSTER_COUNT, DEFAULT_FIELD_FEATURE_COUNT)?;
        let dl = train.len() + val.len();
        let tfc = DEFAULT_MONSTER_COUNT + DEFAULT_FIELD_FEATURE_COUNT;

        fs::create_dir_all(&cfg.save_dir).map_err(|e| e.to_string())?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = UnitAwareTransformer::new(vb, tfc, cfg.embed_dim, cfg.num_heads, cfg.n_layers)?;

        let mut opt = AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW {
            lr: cfg.learning_rate, weight_decay: cfg.weight_decay,
            beta1: 0.9, beta2: 0.999, eps: 1e-8,
        }).map_err(|e| e.to_string())?;

        let mut best_acc = 0.0f32;
        let mut best_loss = f32::INFINITY;
        let mut paths = Vec::new();
        let t0 = std::time::Instant::now();

        for epoch in 0..cfg.epochs {
            // 学习率调度：根据 lr_scheduler 更新学习率
            let new_lr = match &cfg.lr_scheduler {
                crate::core::LrScheduler::Fixed => cfg.learning_rate,
                crate::core::LrScheduler::CosineAnnealing { t_max, eta_min } => {
                    let t_max = *t_max as f64;
                    let eta_min = *eta_min;
                    let base_lr = cfg.learning_rate;
                    eta_min + (base_lr - eta_min) * (1.0 + (std::f64::consts::PI * epoch as f64 / t_max).cos()) / 2.0
                }
            };
            opt.set_learning_rate(new_lr);

            let (train_loss, train_acc) = train_epoch(&model, &train, &device, &mut opt, cfg.batch_size, tfc, cfg.dropout, &varmap.all_vars(), cfg.gradient_clip_norm)?;
            let (val_loss, val_acc) = eval(&model, &val, &device, cfg.batch_size, tfc)?;

            let sd = Path::new(&cfg.save_dir);
            if val_acc > best_acc {
                best_acc = val_acc;
                let p = sd.join("best_model_acc.safetensors");
                export_safetensors(&varmap, &p)?;
                paths.push(p);
            }
            if val_loss < best_loss {
                best_loss = val_loss;
                let p = sd.join("best_model_loss.safetensors");
                export_safetensors(&varmap, &p)?;
                paths.push(p);
            }

            let elapsed = t0.elapsed().as_secs_f64();
            let avg = elapsed / (epoch as f64 + 1.0);
            let rem = avg * (cfg.epochs as f64 - epoch as f64 - 1.0);
            let _ = tx.send(TrainProgress {
                epoch: epoch+1, total_epochs: cfg.epochs,
                train_loss, train_acc, val_loss, val_acc,
                best_acc, best_loss, elapsed_secs: elapsed,
                estimated_remaining_secs: rem.max(0.0), device_info: dev_info.clone(),
            });
        }

        rename_model_files(Path::new(&cfg.save_dir), dl, best_acc, best_loss)?;
        Ok(TrainResult { best_acc, best_loss, total_epochs: cfg.epochs, data_length: dl, model_paths: paths })
    }
}

fn train_epoch(
    model: &UnitAwareTransformer,
    samples: &[TrainSample],
    device: &Device,
    optimizer: &mut AdamW,
    batch_size: usize,
    total_feature_count: usize,
    dropout: f32,
    train_vars: &[candle_core::Var],
    max_grad_norm: f64,
) -> Result<(f32, f32), String> {
    let mut total_loss = 0.0f32;
    let mut correct_count = 0usize;
    let mut total_count = 0usize;
    let num_batches = (samples.len() + batch_size - 1) / batch_size;

    for batch_index in 0..num_batches {
        let start = batch_index * batch_size;
        let end = (start + batch_size).min(samples.len());
        let batch = &samples[start..end];
        let actual_batch_size = batch.len();

        let left_signs = batch_to_tensor(batch, |sample| &sample.left_signs, device, (actual_batch_size, total_feature_count))?;
        let left_counts = batch_to_tensor(batch, |sample| &sample.left_counts, device, (actual_batch_size, total_feature_count))?;
        let right_signs = batch_to_tensor(batch, |sample| &sample.right_signs, device, (actual_batch_size, total_feature_count))?;
        let right_counts = batch_to_tensor(batch, |sample| &sample.right_counts, device, (actual_batch_size, total_feature_count))?;

        let labels: Vec<f32> = batch.iter().map(|sample| sample.label).collect();
        let labels_tensor = Tensor::from_vec(labels, (actual_batch_size,), device).map_err(se)?;

        let output = model.forward(&left_signs, &left_counts, &right_signs, &right_counts, dropout, true)?;
        let diff = output.sub(&labels_tensor).map_err(se)?;
        let loss = diff.sqr().map_err(se)?.mean_all().map_err(se)?;

        // 反向传播
        let mut grads = loss.backward().map_err(|e| e.to_string())?;

        // 梯度裁剪
        clip_grad_norm_(&mut grads, train_vars, max_grad_norm);

        optimizer.step(&grads).map_err(|e| e.to_string())?;

        total_loss += loss.to_scalar::<f32>().map_err(se)?;

        let predictions = output.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let label_binary = labels_tensor.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let matches: u32 = predictions.eq(&label_binary).map_err(se)?
            .to_dtype(DType::U32).map_err(se)?
            .sum_all().map_err(se)?
            .to_scalar::<u32>().map_err(se)?;

        correct_count += matches as usize;
        total_count += actual_batch_size;
    }

    let avg_loss = total_loss / num_batches.max(1) as f32;
    let accuracy = 100.0 * correct_count as f32 / total_count.max(1) as f32;
    Ok((avg_loss, accuracy))
}

/// 梯度裁剪：计算所有参数梯度的L2范数，若超过max_norm则按比例缩放
fn clip_grad_norm_(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
    max_norm: f64,
) {
    // 计算所有参数梯度的L2范数
    let mut total_norm_sq = 0.0f64;
    for var in vars {
        let tensor = var.as_tensor();
        if let Some(grad) = grads.get(tensor) {
            if let Ok(norm_sq) = grad.sqr().and_then(|s| s.sum_all()).and_then(|s| s.to_scalar::<f32>()) {
                total_norm_sq += norm_sq as f64;
            }
        }
    }
    let total_norm = total_norm_sq.sqrt();

    // 若范数超过max_norm，按比例缩放所有梯度
    if total_norm > max_norm && total_norm > 0.0 {
        let scale = max_norm / total_norm;
        for var in vars {
            let tensor = var.as_tensor();
            if let Some(grad) = grads.get(tensor).cloned() {
                if let Ok(scaled) = grad.affine(scale as f64, 0.0) {
                    grads.insert(tensor, scaled);
                }
            }
        }
    }
}

fn eval(
    model: &UnitAwareTransformer,
    samples: &[TrainSample],
    device: &Device,
    batch_size: usize,
    total_feature_count: usize,
) -> Result<(f32, f32), String> {
    let mut total_loss = 0.0f32;
    let mut correct_count = 0usize;
    let mut total_count = 0usize;
    let num_batches = (samples.len() + batch_size - 1) / batch_size;

    for batch_index in 0..num_batches {
        let start = batch_index * batch_size;
        let end = (start + batch_size).min(samples.len());
        let batch = &samples[start..end];
        let actual_batch_size = batch.len();

        let left_signs = batch_to_tensor(batch, |sample| &sample.left_signs, device, (actual_batch_size, total_feature_count))?;
        let left_counts = batch_to_tensor(batch, |sample| &sample.left_counts, device, (actual_batch_size, total_feature_count))?;
        let right_signs = batch_to_tensor(batch, |sample| &sample.right_signs, device, (actual_batch_size, total_feature_count))?;
        let right_counts = batch_to_tensor(batch, |sample| &sample.right_counts, device, (actual_batch_size, total_feature_count))?;

        let labels: Vec<f32> = batch.iter().map(|sample| sample.label).collect();
        let labels_tensor = Tensor::from_vec(labels, (actual_batch_size,), device).map_err(se)?;

        let output = model.forward(&left_signs, &left_counts, &right_signs, &right_counts, 0.0, false)?;
        let diff = output.sub(&labels_tensor).map_err(se)?;
        let loss = diff.sqr().map_err(se)?.mean_all().map_err(se)?;

        total_loss += loss.to_scalar::<f32>().map_err(se)?;

        let predictions = output.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let label_binary = labels_tensor.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let matches: u32 = predictions.eq(&label_binary).map_err(se)?
            .to_dtype(DType::U32).map_err(se)?
            .sum_all().map_err(se)?
            .to_scalar::<u32>().map_err(se)?;

        correct_count += matches as usize;
        total_count += actual_batch_size;
    }

    let avg_loss = total_loss / num_batches.max(1) as f32;
    let accuracy = 100.0 * correct_count as f32 / total_count.max(1) as f32;
    Ok((avg_loss, accuracy))
}

fn batch_to_tensor<F>(
    batch: &[TrainSample],
    extract: F,
    device: &Device,
    shape: (usize, usize),
) -> Result<Tensor, String>
where
    F: Fn(&TrainSample) -> &Vec<f32>,
{
    let mut data = Vec::with_capacity(shape.0 * shape.1);
    for sample in batch {
        data.extend_from_slice(extract(sample));
    }
    Tensor::from_vec(data, shape, device).map_err(se)
}

pub fn export_safetensors(varmap: &VarMap, path: &Path) -> Result<(), String> {
    let data = varmap.data().lock().map_err(|e| e.to_string())?;
    let tensors: std::collections::HashMap<String, Tensor> = data.iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
    candle_core::safetensors::save(&tensors, path).map_err(|e| format!("导出safetensors失败: {e}"))
}

fn rename_model_files(sd: &Path, dl: usize, ba: f32, bl: f32) -> Result<(), String> {
    let now = chrono_ts();
    let base = format!("data{}_acc{:.4}_loss{:.4}_{}", dl, ba, bl, now);
    for p in &["best_model_acc", "best_model_loss", "best_model_full"] {
        let old = sd.join(format!("{p}.safetensors"));
        if old.exists() { fs::rename(&old, sd.join(format!("{p}_{base}.safetensors"))).map_err(|e| e.to_string())?; }
    }
    Ok(())
}

/// 生成时间戳字符串（考虑闰年的近似计算）
fn chrono_ts() -> String {
    let total_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut remaining_days = total_secs / 86400;
    let time_of_day = total_secs % 86400;

    // 从1970年开始逐年减去天数，考虑闰年
    let mut year = 1970u32;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    // 逐月减去天数
    let month_days = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    for &days in &month_days {
        if remaining_days < days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }

    let day = remaining_days + 1;
    let hour = time_of_day / 3600;
    let minute = (time_of_day % 3600) / 60;
    let second = time_of_day % 60;

    format!(
        "{:04}_{:02}_{:02}_{:02}_{:02}_{:02}",
        year, month, day, hour, minute, second
    )
}

fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}
