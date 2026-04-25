use crate::core::{TrainConfig, TrainProgress, TrainResult, TrainSample};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, VarBuilder, VarMap};
use std::fs;
use std::path::Path;

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
    let val_size = (all_samples.len() as f32 * config.test_size) as usize;
    let val_size = val_size.max(1).min(all_samples.len() - 1);
    let mut rng = rand::rng();
    use rand::seq::SliceRandom;
    all_samples.shuffle(&mut rng);
    let val_samples = all_samples.split_off(all_samples.len() - val_size);
    Ok((all_samples, val_samples))
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

    fn fwd(&self, lf:&Tensor, rf:&Tensor) -> Result<(Tensor,Tensor), String> {
        let (nh,ed) = (self.nh, self.ed);
        // 敌方交叉注意力
        let dl = Self::mha(&self.eq,&self.ek,&self.ev,&self.eo, lf,rf,rf, nh,ed)?;
        let dr = Self::mha(&self.eq,&self.ek,&self.ev,&self.eo, rf,lf,lf, nh,ed)?;
        let lf = lf.add(&dl).map_err(se)?;
        let rf = rf.add(&dr).map_err(se)?;
        // 敌方 FFN
        let lf = lf.add(&Self::ffn(&self.ef1,&self.ef2,&lf)?).map_err(se)?;
        let rf = rf.add(&Self::ffn(&self.ef1,&self.ef2,&rf)?).map_err(se)?;
        // 友方自注意力
        let dl = Self::mha(&self.fq,&self.fk,&self.fv,&self.fo, &lf,&lf,&lf, nh,ed)?;
        let dr = Self::mha(&self.fq,&self.fk,&self.fv,&self.fo, &rf,&rf,&rf, nh,ed)?;
        let lf = lf.add(&dl).map_err(se)?;
        let rf = rf.add(&dr).map_err(se)?;
        // 友方 FFN
        let lf = lf.add(&Self::ffn(&self.ff1,&self.ff2,&lf)?).map_err(se)?;
        let rf = rf.add(&Self::ffn(&self.ff1,&self.ff2,&rf)?).map_err(se)?;
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

    pub fn forward(&self, _ls:&Tensor, lc:&Tensor, _rs:&Tensor, rc:&Tensor) -> Result<Tensor, String> {
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
            let (nl, nr) = layer.fwd(&lf, &rf)?;
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
            let (tl, ta) = train_epoch(&model, &train, &device, &mut opt, cfg.batch_size, tfc)?;
            let (vl, va) = eval(&model, &val, &device, cfg.batch_size, tfc)?;

            let sd = Path::new(&cfg.save_dir);
            if va > best_acc {
                best_acc = va;
                let p = sd.join("best_model_acc.safetensors");
                export_safetensors(&varmap, &p)?;
                paths.push(p);
            }
            if vl < best_loss {
                best_loss = vl;
                let p = sd.join("best_model_loss.safetensors");
                export_safetensors(&varmap, &p)?;
                paths.push(p);
            }

            let elapsed = t0.elapsed().as_secs_f64();
            let avg = elapsed / (epoch as f64 + 1.0);
            let rem = avg * (cfg.epochs as f64 - epoch as f64 - 1.0);
            let _ = tx.send(TrainProgress {
                epoch: epoch+1, total_epochs: cfg.epochs,
                train_loss: tl, train_acc: ta, val_loss: vl, val_acc: va,
                best_acc, best_loss, elapsed_secs: elapsed,
                estimated_remaining_secs: rem.max(0.0), device_info: dev_info.clone(),
            });
        }

        rename_model_files(Path::new(&cfg.save_dir), dl, best_acc, best_loss)?;
        Ok(TrainResult { best_acc, best_loss, total_epochs: cfg.epochs, data_length: dl, model_paths: paths })
    }
}

fn train_epoch(model:&UnitAwareTransformer, samples:&[TrainSample], device:&Device,
               opt:&mut AdamW, bs:usize, tfc:usize) -> Result<(f32,f32), String> {
    let mut tl = 0.0f32; let mut cor = 0usize; let mut tot = 0usize;
    let nb = (samples.len()+bs-1)/bs;
    for bi in 0..nb {
        let s = bi*bs; let e = (s+bs).min(samples.len());
        let batch = &samples[s..e]; let bsz = batch.len();
        let ls = b2t(batch, |x| &x.left_signs, device, (bsz,tfc))?;
        let lc = b2t(batch, |x| &x.left_counts, device, (bsz,tfc))?;
        let rs = b2t(batch, |x| &x.right_signs, device, (bsz,tfc))?;
        let rc = b2t(batch, |x| &x.right_counts, device, (bsz,tfc))?;
        let lab: Vec<f32> = batch.iter().map(|x| x.label).collect();
        let lab = Tensor::from_vec(lab, (bsz,), device).map_err(se)?;
        let out = model.forward(&ls,&lc,&rs,&rc)?;
        let diff = out.sub(&lab).map_err(se)?;
        let loss = diff.sqr().map_err(se)?.mean_all().map_err(se)?;
        opt.backward_step(&loss).map_err(|e| e.to_string())?;
        tl += loss.to_scalar::<f32>().map_err(se)?;
        let pred = out.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let lp = lab.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let eq: u32 = pred.eq(&lp).map_err(se)?.to_dtype(DType::U32).map_err(se)?.sum_all().map_err(se)?.to_scalar::<u32>().map_err(se)?;
        cor += eq as usize; tot += bsz;
    }
    Ok((tl/nb.max(1) as f32, 100.0*cor as f32/tot.max(1) as f32))
}

fn eval(model:&UnitAwareTransformer, samples:&[TrainSample], device:&Device,
        bs:usize, tfc:usize) -> Result<(f32,f32), String> {
    let mut tl = 0.0f32; let mut cor = 0usize; let mut tot = 0usize;
    let nb = (samples.len()+bs-1)/bs;
    for bi in 0..nb {
        let s = bi*bs; let e = (s+bs).min(samples.len());
        let batch = &samples[s..e]; let bsz = batch.len();
        let ls = b2t(batch, |x| &x.left_signs, device, (bsz,tfc))?;
        let lc = b2t(batch, |x| &x.left_counts, device, (bsz,tfc))?;
        let rs = b2t(batch, |x| &x.right_signs, device, (bsz,tfc))?;
        let rc = b2t(batch, |x| &x.right_counts, device, (bsz,tfc))?;
        let lab: Vec<f32> = batch.iter().map(|x| x.label).collect();
        let lab = Tensor::from_vec(lab, (bsz,), device).map_err(se)?;
        let out = model.forward(&ls,&lc,&rs,&rc)?;
        let diff = out.sub(&lab).map_err(se)?;
        let loss = diff.sqr().map_err(se)?.mean_all().map_err(se)?;
        tl += loss.to_scalar::<f32>().map_err(se)?;
        let pred = out.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let lp = lab.ge(0.5f64).map_err(se)?.to_dtype(DType::F32).map_err(se)?;
        let eq: u32 = pred.eq(&lp).map_err(se)?.to_dtype(DType::U32).map_err(se)?.sum_all().map_err(se)?.to_scalar::<u32>().map_err(se)?;
        cor += eq as usize; tot += bsz;
    }
    Ok((tl/nb.max(1) as f32, 100.0*cor as f32/tot.max(1) as f32))
}

fn b2t<F>(batch:&[TrainSample], ext:F, device:&Device, shape:(usize,usize)) -> Result<Tensor,String>
where F: Fn(&TrainSample)->&Vec<f32> {
    let mut d = Vec::with_capacity(shape.0*shape.1);
    for s in batch { d.extend_from_slice(ext(s)); }
    Tensor::from_vec(d, shape, device).map_err(se)
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

fn chrono_ts() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let d = now/86400; let tod = now%86400;
    format!("{:04}_{:02}_{:02}_{:02}_{:02}_{:02}", 1970+d/365, d%365/30+1, d%365%30+1, tod/3600, tod%3600/60, tod%60)
}
