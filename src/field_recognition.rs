use crate::core::RelativeRoi;
use image::RgbaImage;
use std::collections::HashMap;
use std::path::Path;

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

#[derive(Debug, Clone)]
pub struct FieldRecognizer {
    idx_to_class: HashMap<usize, String>,
    grouped_elements: HashMap<String, Vec<String>>,
    feature_columns: Vec<String>,
    is_initialized: bool,
}

impl FieldRecognizer {
    pub fn new(model_dir: &Path) -> Self {
        let mut recognizer = Self {
            idx_to_class: HashMap::new(),
            grouped_elements: HashMap::new(),
            feature_columns: Vec::new(),
            is_initialized: false,
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

    fn classify_roi(&self, _roi: &RgbaImage) -> Option<String> {
        // Placeholder: actual candle MobileNetV3 inference would go here.
        // For now, returns None (no detection) until a model is loaded.
        None
    }

    pub fn get_feature_columns(&self) -> Vec<String> {
        self.feature_columns.clone()
    }

    pub fn is_ready(&self) -> bool {
        self.is_initialized
    }
}
