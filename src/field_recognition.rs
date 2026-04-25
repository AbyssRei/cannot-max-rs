use crate::core::Roi;
use image::RgbaImage;
use std::collections::HashMap;
use std::path::Path;

const ROI_COORDINATES: &[(&str, &[Roi])] = &[
    (
        "altar_vertical",
        &[
            Roi { x: 910, y: 174, width: 95, height: 104 },
            Roi { x: 910, y: 429, width: 102, height: 108 },
            Roi { x: 900, y: 755, width: 120, height: 108 },
        ],
    ),
    (
        "block_parallel",
        &[
            Roi { x: 694, y: 240, width: 530, height: 122 },
            Roi { x: 651, y: 614, width: 620, height: 143 },
        ],
    ),
    (
        "block_vertical",
        &[
            Roi { x: 647, y: 233, width: 153, height: 523 },
            Roi { x: 1112, y: 239, width: 159, height: 514 },
        ],
    ),
    (
        "coil_narrow",
        &[
            Roi { x: 915, y: 110, width: 85, height: 89 },
            Roi { x: 815, y: 257, width: 86, height: 98 },
            Roi { x: 1024, y: 258, width: 79, height: 98 },
            Roi { x: 790, y: 643, width: 97, height: 102 },
            Roi { x: 1031, y: 639, width: 102, height: 108 },
        ],
    ),
    (
        "coil_wide",
        &[
            Roi { x: 719, y: 181, width: 81, height: 89 },
            Roi { x: 602, y: 346, width: 81, height: 94 },
            Roi { x: 578, y: 535, width: 81, height: 95 },
            Roi { x: 669, y: 759, width: 91, height: 95 },
            Roi { x: 1159, y: 757, width: 93, height: 92 },
            Roi { x: 1257, y: 533, width: 94, height: 102 },
            Roi { x: 1236, y: 344, width: 85, height: 97 },
            Roi { x: 1120, y: 180, width: 75, height: 91 },
        ],
    ),
    (
        "crossbow_top",
        &[Roi { x: 718, y: 13, width: 484, height: 106 }],
    ),
    (
        "fire_side_left",
        &[Roi { x: 98, y: 246, width: 184, height: 281 }],
    ),
    (
        "fire_side_right",
        &[Roi { x: 1656, y: 430, width: 235, height: 315 }],
    ),
    (
        "fire_top",
        &[
            Roi { x: 532, y: 17, width: 188, height: 97 },
            Roi { x: 1325, y: 14, width: 60, height: 100 },
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
        if width != 1920 || height != 1080 {
            return HashMap::new();
        }

        let mut detected_classes: Vec<String> = Vec::new();

        for (_location, rois) in ROI_COORDINATES {
            for roi in *rois {
                let roi = roi.clamp(width, height);
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
