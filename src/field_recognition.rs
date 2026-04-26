use crate::core::RelativeRoi;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use image::RgbaImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// 场地识别 ROI 坐标（相对比例，适配 720p~1080p+）
const ROI_COORDINATES_REL: &[(&str, &[RelativeRoi])] = &[
    (
        "altar_vertical",
        &[
            RelativeRoi { x: 0.4740, y: 0.1611, width: 0.0495, height: 0.0963 },
            RelativeRoi { x: 0.4740, y: 0.3972, width: 0.0531, height: 0.1000 },
            RelativeRoi { x: 0.4688, y: 0.6991, width: 0.0625, height: 0.1000 },
        ],
    ),
    (
        "block_parallel",
        &[
            RelativeRoi { x: 0.3615, y: 0.2222, width: 0.2760, height: 0.1130 },
            RelativeRoi { x: 0.3391, y: 0.5694, width: 0.3229, height: 0.1324 },
        ],
    ),
    (
        "block_vertical",
        &[
            RelativeRoi { x: 0.3370, y: 0.2157, width: 0.0797, height: 0.4843 },
            RelativeRoi { x: 0.5792, y: 0.2213, width: 0.0828, height: 0.4759 },
        ],
    ),
    (
        "coil_narrow",
        &[
            RelativeRoi { x: 0.4766, y: 0.1019, width: 0.0443, height: 0.0824 },
            RelativeRoi { x: 0.4245, y: 0.2380, width: 0.0448, height: 0.0907 },
            RelativeRoi { x: 0.5333, y: 0.2389, width: 0.0411, height: 0.0907 },
            RelativeRoi { x: 0.4115, y: 0.5954, width: 0.0505, height: 0.0944 },
            RelativeRoi { x: 0.5369, y: 0.5917, width: 0.0531, height: 0.1000 },
        ],
    ),
    (
        "coil_wide",
        &[
            RelativeRoi { x: 0.3745, y: 0.1676, width: 0.0422, height: 0.0824 },
            RelativeRoi { x: 0.3135, y: 0.3204, width: 0.0422, height: 0.0870 },
            RelativeRoi { x: 0.3010, y: 0.4954, width: 0.0422, height: 0.0880 },
            RelativeRoi { x: 0.3484, y: 0.7028, width: 0.0474, height: 0.0880 },
            RelativeRoi { x: 0.6036, y: 0.7009, width: 0.0484, height: 0.0852 },
            RelativeRoi { x: 0.6547, y: 0.4935, width: 0.0490, height: 0.0944 },
            RelativeRoi { x: 0.6438, y: 0.3185, width: 0.0443, height: 0.0898 },
            RelativeRoi { x: 0.5833, y: 0.1667, width: 0.0391, height: 0.0843 },
        ],
    ),
    (
        "crossbow_top",
        &[RelativeRoi { x: 0.3740, y: 0.0120, width: 0.2521, height: 0.0981 }],
    ),
    (
        "fire_side_left",
        &[RelativeRoi { x: 0.0510, y: 0.2278, width: 0.0958, height: 0.2602 }],
    ),
    (
        "fire_side_right",
        &[RelativeRoi { x: 0.8625, y: 0.3981, width: 0.1224, height: 0.2917 }],
    ),
    (
        "fire_top",
        &[
            RelativeRoi { x: 0.2771, y: 0.0157, width: 0.0979, height: 0.0898 },
            RelativeRoi { x: 0.6891, y: 0.0130, width: 0.0313, height: 0.0926 },
        ],
    ),
];

// ── MobileNetV3 Small 场地分类模型 ──

/// 全局缓存的场地分类模型（OnceLock 保证只加载一次）
static FIELD_MODEL: OnceLock<Result<FieldModel, String>> = OnceLock::new();

/// 场地分类模型封装
struct FieldModel {
    varmap: candle_nn::VarMap,
    num_classes: usize,
}

impl FieldModel {
    /// 加载 safetensors 模型文件
    fn load(model_path: &Path) -> Result<Self, String> {
        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(model_path, &device)
            .map_err(|e| format!("加载场地模型失败: {e}"))?;

        // 推断类别数：从最后一个全连接层权重获取
        let num_classes = tensors.keys()
            .filter_map(|k| {
                if k.contains("classifier") && k.contains("weight") {
                    tensors.get(k).map(|t| t.dims()[0])
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(12);

        let mut varmap = candle_nn::VarMap::new();
        for (key, tensor) in &tensors {
            varmap.set_one(key, tensor)
                .map_err(|e| format!("设置场地模型权重失败: {e}"))?;
        }

        Ok(Self { varmap, num_classes })
    }

    /// 对单张 224x224 RGBA 图像进行分类，返回 (类别索引, 置信度)
    fn classify(&self, image: &RgbaImage) -> Result<(usize, f32), String> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&self.varmap, DType::F32, &device);

        // 将 RGBA 图像转为 CHW 张量 [1, 3, 224, 224]，使用 ImageNet 标准归一化
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut pixel_data = Vec::with_capacity(3 * 224 * 224);
        for c in 0..3usize {
            for y in 0..224usize {
                for x in 0..224usize {
                    let pixel = image.get_pixel(x as u32, y as u32);
                    let normalized = ((pixel[c] as f32 / 255.0) - mean[c]) / std[c];
                    pixel_data.push(normalized);
                }
            }
        }

        let input = Tensor::from_vec(pixel_data, (1, 3, 224, 224), &device)
            .map_err(|e| e.to_string())?;

        // 使用 MobileNetV3 Small 架构进行前向推理
        let logits = mobilenetv3_small_forward(&input, vb, self.num_classes)?;

        // 计算 softmax 得到概率分布
        let probs = candle_nn::ops::softmax(&logits, 1).map_err(|e| e.to_string())?;

        // 取 argmax 作为预测类别
        let class_index = probs
            .argmax(1)
            .map_err(|e| e.to_string())?
            .to_scalar::<u32>()
            .map_err(|e| e.to_string())? as usize;

        // 获取最高概率（置信度）
        let max_prob = probs
            .max(1)
            .map_err(|e| e.to_string())?
            .to_scalar::<f32>()
            .map_err(|e| e.to_string())?;

        // 若最高概率 < 0.5，返回 (0, 0.0) 表示"未知"地形
        if max_prob < 0.5 {
            Ok((0, 0.0))
        } else {
            Ok((class_index, max_prob))
        }
    }
}

/// MobileNetV3 Small 前向传播（简化版，使用卷积块堆叠）
///
/// 架构参考 PyTorch torchvision MobileNetV3_Small:
///   Conv2d(3,16) -> InvertedResidual x 11 -> Conv2d -> Linear
///   每个 InvertedResidual: expand -> depthwise -> project (with SE)
fn mobilenetv3_small_forward(
    input: &Tensor,
    vb: VarBuilder,
    num_classes: usize,
) -> Result<Tensor, String> {
    let se = |e: candle_core::Error| e.to_string();

    // 初始卷积: Conv2d(3, 16, 3, stride=2, padding=1) + BatchNorm + HardSwish
    let conv0_w = vb.get((16, 3, 3, 3), "features.0.0.weight").map_err(se)?;
    let conv0_b = vb.get((16,), "features.0.0.bias").map_err(se)?;
    let conv0 = candle_nn::Conv2d::new(conv0_w, Some(conv0_b), candle_nn::Conv2dConfig {
        stride: 2, padding: 1, ..Default::default()
    });
    let bn0_w = vb.get((16,), "features.0.1.weight").map_err(se)?;
    let bn0_b = vb.get((16,), "features.0.1.bias").map_err(se)?;
    let bn0_rm = vb.get((16,), "features.0.1.running_mean").map_err(se)?;
    let bn0_rv = vb.get((16,), "features.0.1.running_var").map_err(se)?;

    let x = conv0.forward(input).map_err(se)?;
    let x = batch_norm_eval(&x, &bn0_w, &bn0_b, &bn0_rm, &bn0_rv)?;
    let x = hard_swish(&x)?;

    // 依次通过 InvertedResidual 块 (features.1 ~ features.11)
    // MobileNetV3 Small 的 11 个 InvertedResidual 块配置:
    // (expand, out_ch, kernel, stride, use_se, activation)
    let block_configs: &[(usize, usize, usize, usize, bool, &str)] = &[
        (16,  16,  3, 2, true,  "relu"),   // features.1
        (72,  24,  3, 2, false, "relu"),   // features.2
        (88,  24,  3, 1, false, "relu"),   // features.3
        (96,  40,  5, 2, true,  "hardswish"), // features.4
        (240, 40,  5, 1, true,  "hardswish"), // features.5
        (240, 40,  5, 1, true,  "hardswish"), // features.6
        (120, 48,  5, 1, true,  "hardswish"), // features.7
        (144, 48,  5, 1, true,  "hardswish"), // features.8
        (288, 96,  5, 2, true,  "hardswish"), // features.9
        (576, 96,  5, 1, true,  "hardswish"), // features.10
        (576, 96,  5, 1, true,  "hardswish"), // features.11
    ];

    let mut x = x;
    let mut in_channels = 16usize;

    for (i, &(expand_ch, out_ch, kernel, stride, use_se, activation)) in block_configs.iter().enumerate() {
        let block_vb = vb.pp(&format!("features.{}", i + 1));

        // 判断是否使用扩展层（expand_ch != in_channels）
        let use_expand = expand_ch != in_channels;

        let mut h = x.clone();

        // 扩展卷积 (1x1)
        if use_expand {
            let exp_w = block_vb.get((expand_ch, in_channels, 1, 1), "block.0.0.weight").map_err(se)?;
            let exp_b = block_vb.get((expand_ch,), "block.0.0.bias").map_err(se)?;
            let exp_conv = candle_nn::Conv2d::new(exp_w, Some(exp_b), candle_nn::Conv2dConfig::default());
            let exp_bn_w = block_vb.get((expand_ch,), "block.0.1.weight").map_err(se)?;
            let exp_bn_b = block_vb.get((expand_ch,), "block.0.1.bias").map_err(se)?;
            let exp_bn_rm = block_vb.get((expand_ch,), "block.0.1.running_mean").map_err(se)?;
            let exp_bn_rv = block_vb.get((expand_ch,), "block.0.1.running_var").map_err(se)?;

            h = exp_conv.forward(&h).map_err(se)?;
            h = batch_norm_eval(&h, &exp_bn_w, &exp_bn_b, &exp_bn_rm, &exp_bn_rv)?;
            h = if activation == "hardswish" { hard_swish(&h)? } else { relu6(&h)? };
        }

        // 深度卷积 (depthwise)
        let dw_idx = if use_expand { 3 } else { 0 };
        let dw_w = block_vb.get((expand_ch, 1, kernel, kernel), &format!("block.{}.0.weight", dw_idx)).map_err(se)?;
        let dw_b = block_vb.get((expand_ch,), &format!("block.{}.0.bias", dw_idx)).map_err(se)?;
        let padding = kernel / 2;
        let dw_conv = candle_nn::Conv2d::new(dw_w, Some(dw_b), candle_nn::Conv2dConfig {
            stride, padding, groups: expand_ch, ..Default::default()
        });
        let dw_bn_w = block_vb.get((expand_ch,), &format!("block.{}.1.weight", dw_idx)).map_err(se)?;
        let dw_bn_b = block_vb.get((expand_ch,), &format!("block.{}.1.bias", dw_idx)).map_err(se)?;
        let dw_bn_rm = block_vb.get((expand_ch,), &format!("block.{}.1.running_mean", dw_idx)).map_err(se)?;
        let dw_bn_rv = block_vb.get((expand_ch,), &format!("block.{}.1.running_var", dw_idx)).map_err(se)?;

        h = dw_conv.forward(&h).map_err(se)?;
        h = batch_norm_eval(&h, &dw_bn_w, &dw_bn_b, &dw_bn_rm, &dw_bn_rv)?;

        // SE (Squeeze-and-Excitation) 模块
        if use_se {
            let se_idx = dw_idx + 1;
            let se_reduce_ch = expand_ch / 4;
            // 全局平均池化
            let pooled = h.mean(2).map_err(se)?.mean(2).map_err(se)?; // [B, C]
            // FC reduce
            let se_fc0_w = block_vb.get((se_reduce_ch, expand_ch), &format!("block.{}.fc1.weight", se_idx)).map_err(se)?;
            let se_fc0_b = block_vb.get((se_reduce_ch,), &format!("block.{}.fc1.bias", se_idx)).map_err(se)?;
            let se_fc1_w = block_vb.get((expand_ch, se_reduce_ch), &format!("block.{}.fc2.weight", se_idx)).map_err(se)?;
            let se_fc1_b = block_vb.get((expand_ch,), &format!("block.{}.fc2.bias", se_idx)).map_err(se)?;

            let se_out = pooled.matmul(&se_fc0_w).map_err(se)?.broadcast_add(&se_fc0_b).map_err(se)?;
            let se_out = relu(&se_out)?;
            let se_out = se_out.matmul(&se_fc1_w).map_err(se)?.broadcast_add(&se_fc1_b).map_err(se)?;
            let se_out = candle_nn::ops::sigmoid(&se_out).map_err(se)?;
            // 重新塑形并乘到 h 上
            let (b, c, _, _) = h.dims4().map_err(se)?;
            let se_scale = se_out.reshape((b, c, 1, 1)).map_err(se)?;
            h = h.broadcast_mul(&se_scale).map_err(se)?;
        }

        // 激活
        h = if activation == "hardswish" { hard_swish(&h)? } else { relu6(&h)? };

        // 投影卷积 (1x1, 线性)
        let proj_idx = dw_idx + if use_se { 2 } else { 1 };
        let proj_w = block_vb.get((out_ch, expand_ch, 1, 1), &format!("block.{}.0.weight", proj_idx)).map_err(se)?;
        let proj_b = block_vb.get((out_ch,), &format!("block.{}.0.bias", proj_idx)).map_err(se)?;
        let proj_conv = candle_nn::Conv2d::new(proj_w, Some(proj_b), candle_nn::Conv2dConfig::default());
        let proj_bn_w = block_vb.get((out_ch,), &format!("block.{}.1.weight", proj_idx)).map_err(se)?;
        let proj_bn_b = block_vb.get((out_ch,), &format!("block.{}.1.bias", proj_idx)).map_err(se)?;
        let proj_bn_rm = block_vb.get((out_ch,), &format!("block.{}.1.running_mean", proj_idx)).map_err(se)?;
        let proj_bn_rv = block_vb.get((out_ch,), &format!("block.{}.1.running_var", proj_idx)).map_err(se)?;

        h = proj_conv.forward(&h).map_err(se)?;
        h = batch_norm_eval(&h, &proj_bn_w, &proj_bn_b, &proj_bn_rm, &proj_bn_rv)?;

        // 残差连接（stride=1 且 in_channels == out_ch 时）
        if stride == 1 && in_channels == out_ch {
            x = x.add(&h).map_err(se)?;
        } else {
            x = h;
        }
        in_channels = out_ch;
    }

    // 最终卷积: Conv2d(96, 576, 1) + BatchNorm + HardSwish
    let final_w = vb.get((576, 96, 1, 1), "features.12.0.weight").map_err(se)?;
    let final_b = vb.get((576,), "features.12.0.bias").map_err(se)?;
    let final_conv = candle_nn::Conv2d::new(final_w, Some(final_b), candle_nn::Conv2dConfig::default());
    let final_bn_w = vb.get((576,), "features.12.1.weight").map_err(se)?;
    let final_bn_b = vb.get((576,), "features.12.1.bias").map_err(se)?;
    let final_bn_rm = vb.get((576,), "features.12.1.running_mean").map_err(se)?;
    let final_bn_rv = vb.get((576,), "features.12.1.running_var").map_err(se)?;

    let x = final_conv.forward(&x).map_err(se)?;
    let x = batch_norm_eval(&x, &final_bn_w, &final_bn_b, &final_bn_rm, &final_bn_rv)?;
    let x = hard_swish(&x)?;

    // 全局平均池化
    let x = x.mean(2).map_err(se)?.mean(2).map_err(se)?; // [B, 576]

    // 分类器: Linear(576, num_classes)
    let cls_w = vb.get((num_classes, 576), "classifier.3.weight").map_err(se)?;
    let cls_b = vb.get((num_classes,), "classifier.3.bias").map_err(se)?;
    let output = x.matmul(&cls_w).map_err(se)?.broadcast_add(&cls_b).map_err(se)?;

    Ok(output)
}

/// 推理模式 BatchNorm（使用 running_mean/running_var）
fn batch_norm_eval(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
) -> Result<Tensor, String> {
    let eps = 1e-5f64;
    let var_eps = running_var.add(&Tensor::new(eps, x.device()).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let inv_std = var_eps.sqrt().map_err(|e| e.to_string())?.recip().map_err(|e| e.to_string())?;
    let x_norm = x.sub(running_mean).map_err(|e| e.to_string())?.broadcast_mul(&inv_std).map_err(|e| e.to_string())?;
    x_norm.broadcast_mul(weight).map_err(|e| e.to_string())?.broadcast_add(bias).map_err(|e| e.to_string())
}

/// HardSwish 激活函数: x * relu6(x + 3) / 6
fn hard_swish(x: &Tensor) -> Result<Tensor, String> {
    let six = Tensor::new(6.0f64, x.device()).map_err(|e| e.to_string())?;
    let three = Tensor::new(3.0f64, x.device()).map_err(|e| e.to_string())?;
    let x_plus_3 = x.add(&three).map_err(|e| e.to_string())?;
    let relu6_x = x_plus_3.clamp(&Tensor::zeros(x.shape(), x.dtype(), x.device()).map_err(|e| e.to_string())?, &six)
        .map_err(|e| e.to_string())?;
    x.mul(&relu6_x).map_err(|e| e.to_string())?.broadcast_div(&six).map_err(|e| e.to_string())
}

/// ReLU6 激活函数: min(max(x, 0), 6)
fn relu6(x: &Tensor) -> Result<Tensor, String> {
    let six = Tensor::new(6.0f64, x.device()).map_err(|e| e.to_string())?;
    let zero = Tensor::zeros(x.shape(), x.dtype(), x.device()).map_err(|e| e.to_string())?;
    x.clamp(&zero, &six).map_err(|e| e.to_string())
}

/// ReLU 激活函数
fn relu(x: &Tensor) -> Result<Tensor, String> {
    let zero = Tensor::zeros(x.shape(), x.dtype(), x.device()).map_err(|e| e.to_string())?;
    x.maximum(&zero).map_err(|e| e.to_string())
}

// ── FieldRecognizer ──

#[derive(Debug, Clone)]
pub struct FieldRecognizer {
    idx_to_class: HashMap<usize, String>,
    grouped_elements: HashMap<String, Vec<String>>,
    feature_columns: Vec<String>,
    is_initialized: bool,
    model_path: Option<PathBuf>,
}

impl FieldRecognizer {
    pub fn new(model_dir: &Path) -> Self {
        let mut recognizer = Self {
            idx_to_class: HashMap::new(),
            grouped_elements: HashMap::new(),
            feature_columns: Vec::new(),
            is_initialized: false,
            model_path: None,
        };

        let class_map_path = model_dir.join("class_to_idx.json");
        if !class_map_path.exists() {
            return recognizer;
        }

        let Ok(text) = std::fs::read_to_string(&class_map_path) else {
            return recognizer;
        };
        let Ok(class_to_idx): Result<HashMap<String, usize>, _> =
            serde_json::from_str(&text)
        else {
            return recognizer;
        };

        recognizer.idx_to_class = class_to_idx
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        for class_name in class_to_idx.keys() {
            if class_name.ends_with("_none") {
                continue;
            }
            let condensed = class_name
                .replace("_left_", "_")
                .replace("_right_", "_");
            recognizer
                .grouped_elements
                .entry(condensed)
                .or_default()
                .push(class_name.clone());
        }

        let mut columns: Vec<String> = recognizer.grouped_elements.keys().cloned().collect();
        columns.sort();
        recognizer.feature_columns = columns;
        recognizer.is_initialized = true;

        // 查找模型文件
        let model_candidates = [
            model_dir.join("mobilenetv3_small.safetensors"),
            model_dir.join("field_model.safetensors"),
        ];
        for candidate in &model_candidates {
            if candidate.exists() {
                recognizer.model_path = Some(candidate.clone());
                break;
            }
        }

        recognizer
    }

    pub fn recognize_field_elements(
        &self,
        screenshot: &RgbaImage,
    ) -> HashMap<String, f32> {
        if !self.is_initialized {
            return HashMap::new();
        }

        let width = screenshot.width();
        let height = screenshot.height();
        // 检查是否为合理的 16:9 分辨率（最低 720p）
        if width < 1280 || height < 720 {
            return HashMap::new();
        }

        let mut detected_classes: Vec<String> = Vec::new();

        for (_location, rois) in ROI_COORDINATES_REL {
            for rel_roi in *rois {
                let roi = rel_roi.to_absolute(width, height).clamp(width, height);
                let cropped = image::imageops::crop_imm(
                    screenshot,
                    roi.x,
                    roi.y,
                    roi.width.max(1),
                    roi.height.max(1),
                )
                .to_image();

                let resized =
                    image::imageops::resize(&cropped, 224, 224, image::imageops::FilterType::Triangle);

                if let Some(predicted_class) = self.classify_roi(&resized) {
                    if !predicted_class.ends_with("_none") {
                        detected_classes.push(predicted_class);
                    }
                }
            }
        }

        let detected_set: std::collections::HashSet<&str> =
            detected_classes.iter().map(|s| s.as_str()).collect();

        let mut field_data = HashMap::new();
        for (condensed_name, full_names) in &self.grouped_elements {
            let num_positions = full_names.len();
            if num_positions == 1 {
                field_data.insert(
                    condensed_name.clone(),
                    if detected_set.contains(full_names[0].as_str()) {
                        1.0
                    } else {
                        0.0
                    },
                );
            } else {
                let num_detected = full_names
                    .iter()
                    .filter(|fn_| detected_set.contains(fn_.as_str()))
                    .count();
                let value = if num_detected == num_positions {
                    1.0
                } else if num_detected == 0 {
                    0.0
                } else {
                    -1.0
                };
                field_data.insert(condensed_name.clone(), value);
            }
        }

        field_data
    }

    fn classify_roi(&self, roi: &RgbaImage) -> Option<String> {
        let model_path = self.model_path.as_ref()?;

        // 全局加载模型（OnceLock 保证只加载一次）
        let model_result = FIELD_MODEL.get_or_init(|| {
            FieldModel::load(model_path)
        });

        let model = model_result.as_ref().ok()?;

        // 执行推理
        match model.classify(roi) {
            Ok((class_index, confidence)) => {
                // 置信度为0表示"未知"地形
                if confidence == 0.0 {
                    None
                } else {
                    self.idx_to_class.get(&class_index).cloned()
                }
            }
            Err(e) => {
                eprintln!("场地分类推理失败: {e}");
                None
            }
        }
    }

    pub fn get_feature_columns(&self) -> Vec<String> {
        self.feature_columns.clone()
    }

    pub fn is_ready(&self) -> bool {
        self.is_initialized
    }

    /// 返回模型加载状态描述
    pub fn model_status(&self) -> String {
        if let Some(path) = &self.model_path {
            if path.exists() {
                format!("场地模型: {}", path.display())
            } else {
                format!("场地模型文件不存在: {}", path.display())
            }
        } else {
            "场地模型: 未找到模型文件（场地识别已禁用）".to_string()
        }
    }
}
