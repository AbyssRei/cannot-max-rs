use crate::config::AppConfig;
use image::{DynamicImage, GrayImage};
use maa_framework::buffer::MaaImageBuffer;
use maa_framework::resource::Resource;
use maa_framework::tasker::Tasker;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OcrBackend {
    #[default]
    Maa,
    PaddleOcrVl,
}

impl fmt::Display for OcrBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Maa => f.write_str("MAA OCR"),
            Self::PaddleOcrVl => f.write_str("PaddleOCR-VL"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcrValue {
    pub text: String,
    pub confidence: f32,
    pub backend: OcrBackend,
}

static MAA_LOAD_RESULT: OnceLock<Result<Option<PathBuf>, String>> = OnceLock::new();

pub fn recognize_count(image: &GrayImage, config: &AppConfig) -> Result<Option<OcrValue>, String> {
    match config.ocr_backend {
        OcrBackend::Maa => recognize_with_maa(image, config),
        OcrBackend::PaddleOcrVl => recognize_with_paddle_ocr_vl(image),
    }
}

pub fn prepare_maa_runtime(config: &AppConfig) -> Result<Option<PathBuf>, String> {
    MAA_LOAD_RESULT
        .get_or_init(|| {
            let candidates = candidate_library_paths(config);
            for candidate in candidates {
                if candidate.exists() {
                    maa_framework::load_library(candidate.as_path())?;
                    return Ok(Some(candidate));
                }
            }
            Ok(None)
        })
        .clone()
}

pub fn library_hint(config: &AppConfig) -> String {
    match prepare_maa_runtime(config) {
        Ok(Some(path)) => format!("MAA 动态库: {}", path.display()),
        Ok(None) => "MAA 动态库未找到，运行时将无法使用 MAA OCR/Tasker".to_string(),
        Err(error) => format!("MAA 动态库加载失败: {error}"),
    }
}

pub fn ocr_hint(config: &AppConfig) -> String {
    let model = resolve_ocr_model_path(config)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "未找到".to_string());
    format!("OCR 后端: {} | OCR 模型目录: {model}", config.ocr_backend)
}

fn recognize_with_maa(image: &GrayImage, config: &AppConfig) -> Result<Option<OcrValue>, String> {
    let model_path =
        resolve_ocr_model_path(config).ok_or_else(|| "MAA OCR model path not found".to_string())?;

    let _ = prepare_maa_runtime(config)?;

    let rgb = DynamicImage::ImageLuma8(image.clone()).to_rgb8();
    let maa_image = MaaImageBuffer::from_rgb_image(&rgb)
        .map_err(|error| format!("MAA image conversion failed: {error}"))?;

    let resource = Resource::new().map_err(|error| error.to_string())?;
    let model_job = resource
        .post_ocr_model(model_path.to_string_lossy().as_ref())
        .map_err(|error| error.to_string())?;
    let status = model_job.wait();
    if !status.succeeded() {
        return Err(format!("MAA OCR model load failed: {status}"));
    }

    let tasker = Tasker::new().map_err(|error| error.to_string())?;
    tasker
        .bind_resource(&resource)
        .map_err(|error| error.to_string())?;

    let params = json!({
        "expected": ["^\\d+$"],
        "threshold": 0.2,
        "only_rec": true,
        "replace": [["O", "0"], ["o", "0"], ["I", "1"], ["l", "1"]],
        "order_by": "Horizontal",
        "index": 0,
        "model": "",
        "color_filter": "",
    });

    let job = tasker
        .post_recognition("OCR", &params.to_string(), &maa_image)
        .map_err(|error| error.to_string())?;
    let status = job.wait();
    if !status.succeeded() {
        return Err(format!("MAA OCR recognition failed: {status}"));
    }

    let detail = job
        .get(false)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "MAA OCR returned no detail".to_string())?;

    parse_maa_ocr_detail(detail.detail)
}

fn parse_maa_ocr_detail(detail: serde_json::Value) -> Result<Option<OcrValue>, String> {
    let candidates = match detail {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(map) => vec![serde_json::Value::Object(map)],
        _ => Vec::new(),
    };

    for item in candidates {
        if let Some(text) = item
            .get("text")
            .or_else(|| item.get("label"))
            .and_then(|value| value.as_str())
        {
            let digits: String = text.chars().filter(|char| char.is_ascii_digit()).collect();
            if digits.is_empty() {
                continue;
            }

            let confidence = item
                .get("score")
                .or_else(|| item.get("confidence"))
                .and_then(|value| value.as_f64())
                .unwrap_or(0.75) as f32;

            return Ok(Some(OcrValue {
                text: digits,
                confidence,
                backend: OcrBackend::Maa,
            }));
        }
    }

    Ok(None)
}

fn recognize_with_paddle_ocr_vl(_image: &GrayImage) -> Result<Option<OcrValue>, String> {
    Err(
        "PaddleOCR-VL backend is reserved for deepseek-ocr.rs integration and is not wired yet"
            .to_string(),
    )
}

fn resolve_ocr_model_path(config: &AppConfig) -> Option<PathBuf> {
    let candidates = [
        config.ocr_model_path.clone(),
        config.resource_root.join("model").join("ocr"),
        config.resource_root.join("maa").join("model").join("ocr"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

fn candidate_library_paths(config: &AppConfig) -> Vec<PathBuf> {
    let mut paths = vec![config.maa_library_path.clone()];

    if let Ok(current) = std::env::current_dir() {
        paths.push(current.join("MaaFramework.dll"));
        paths.push(current.join("maa").join("MaaFramework.dll"));
    }

    paths
}
