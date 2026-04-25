use crate::config::AppConfig;
use image::{DynamicImage, GrayImage};
use maa_framework::buffer::MaaImageBuffer;
use maa_framework::resource::Resource;
use maa_framework::tasker::Tasker;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OcrBackend {
    #[default]
    Maa,
    #[serde(alias = "PaddleOcrVl")]
    DeepseekCli,
}

impl fmt::Display for OcrBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Maa => f.write_str("MAA OCR"),
            Self::DeepseekCli => f.write_str("deepseek-ocr.rs CLI"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DeepseekCliModel {
    DeepseekOcr,
    #[default]
    PaddleOcrVl,
    DotsOcr,
}

impl DeepseekCliModel {
    pub const ALL: [Self; 3] = [Self::DeepseekOcr, Self::PaddleOcrVl, Self::DotsOcr];

    pub fn as_id(self) -> &'static str {
        match self {
            Self::DeepseekOcr => "deepseek-ocr",
            Self::PaddleOcrVl => "paddleocr-vl",
            Self::DotsOcr => "dots-ocr",
        }
    }
}

impl fmt::Display for DeepseekCliModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeepseekOcr => f.write_str("DeepSeek-OCR"),
            Self::PaddleOcrVl => f.write_str("PaddleOCR-VL"),
            Self::DotsOcr => f.write_str("DotsOCR"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcrValue {
    pub text: String,
    pub confidence: f32,
    pub backend: OcrBackend,
    pub source_label: String,
    pub cached: bool,
}

static MAA_LOAD_RESULT: OnceLock<Result<Option<PathBuf>, String>> = OnceLock::new();
static OCR_CACHE: OnceLock<Mutex<HashMap<OcrCacheKey, Option<OcrValue>>>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OcrCacheKey {
    backend: OcrBackend,
    model: Option<DeepseekCliModel>,
    image_hash: u64,
}

pub fn recognize_count(image: &GrayImage, config: &AppConfig) -> Result<Option<OcrValue>, String> {
    let key = OcrCacheKey {
        backend: config.ocr_backend,
        model: (config.ocr_backend == OcrBackend::DeepseekCli).then_some(config.deepseek_model),
        image_hash: hash_image(image),
    };

    if let Some(cached) = load_cached_ocr_value(key)? {
        return Ok(Some(cached));
    }

    let value = match config.ocr_backend {
        OcrBackend::Maa => recognize_with_maa(image, config),
        OcrBackend::DeepseekCli => recognize_with_deepseek_cli(image, config),
    }?;

    store_cached_ocr_value(key, value.clone())?;
    Ok(value)
}

fn load_cached_ocr_value(key: OcrCacheKey) -> Result<Option<OcrValue>, String> {
    let cache = OCR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let guard = cache
        .lock()
        .map_err(|_| "OCR cache lock poisoned".to_string())?;
    Ok(guard.get(&key).cloned().flatten().map(|mut value| {
        value.cached = true;
        value
    }))
}

fn store_cached_ocr_value(key: OcrCacheKey, value: Option<OcrValue>) -> Result<(), String> {
    let cache = OCR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| "OCR cache lock poisoned".to_string())?;
    guard.insert(key, value);
    if guard.len() > 256 {
        guard.clear();
    }
    Ok(())
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
    let enhancement = match resolve_deepseek_cli_path(config) {
        Some(path) => format!(
            "增强 OCR CLI: {} | 模型: {} | 设备: {}",
            path.display(),
            config.deepseek_model,
            config.deepseek_device
        ),
        None => format!(
            "增强 OCR CLI: 未找到 | 模型: {} | 设备: {}",
            config.deepseek_model, config.deepseek_device
        ),
    };
    format!(
        "OCR 后端: {} | OCR 模型目录: {model} | {enhancement}",
        config.ocr_backend
    )
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
                source_label: "MAA OCR".to_string(),
                cached: false,
            }));
        }
    }

    Ok(None)
}

fn recognize_with_deepseek_cli(
    image: &GrayImage,
    config: &AppConfig,
) -> Result<Option<OcrValue>, String> {
    let cli = resolve_deepseek_cli_path(config).ok_or_else(|| {
        "deepseek-ocr-cli not found; set the CLI path or add it to PATH".to_string()
    })?;

    let temp_path = temp_image_path("cannot-max-count");
    DynamicImage::ImageLuma8(image.clone())
        .save(&temp_path)
        .map_err(|error| format!("failed to write temporary OCR image: {error}"))?;

    let prompt = "<image>\nRead only the Arabic numeral in this cropped game counter image. Reply with digits only. If uncertain, still return only the most likely digits.";
    let model_id = config.deepseek_model.as_id();

    let mut child = Command::new(&cli)
        .arg("--model")
        .arg(model_id)
        .arg("--prompt")
        .arg(prompt)
        .arg("--image")
        .arg(&temp_path)
        .arg("--device")
        .arg(config.deepseek_device.trim())
        .arg("--max-new-tokens")
        .arg("16")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to launch deepseek-ocr-cli: {error}"))?;

    let output = wait_with_timeout(&mut child, Duration::from_secs(20))
        .map_err(|error| format!("deepseek-ocr-cli invocation failed: {error}"))?;

    let _ = fs::remove_file(&temp_path);

    if !output.status.success() {
        return Err(format!(
            "deepseek-ocr-cli failed for model {}: {}",
            config.deepseek_model,
            summarize_cli_failure(&output)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut digits = extract_digits(&stdout);
    if digits.is_empty() {
        digits = extract_digits(&stderr);
    }
    if digits.is_empty() {
        return Err(format!(
            "deepseek-ocr-cli returned no usable digits for model {}. stdout: {} | stderr: {}",
            config.deepseek_model,
            clean_diagnostic(&stdout),
            clean_diagnostic(&stderr)
        ));
    }

    Ok(Some(OcrValue {
        text: digits,
        confidence: 0.85,
        backend: OcrBackend::DeepseekCli,
        source_label: format!("deepseek-ocr.rs/{}", config.deepseek_model),
        cached: false,
    }))
}

fn resolve_ocr_model_path(config: &AppConfig) -> Option<PathBuf> {
    let candidates = [
        config.ocr_model_path.clone(),
        config.resource_root.join("model").join("ocr"),
        config.resource_root.join("maafw").join("model").join("ocr"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

pub fn resolve_deepseek_cli_path(config: &AppConfig) -> Option<PathBuf> {
    let candidates = [
        config.deepseek_cli_path.clone(),
        PathBuf::from("deepseek-ocr-cli.exe"),
        PathBuf::from("deepseek-ocr-cli"),
    ];

    candidates.into_iter().find(|path| {
        if path.components().count() == 1 {
            return true;
        }
        path.exists()
    })
}

fn candidate_library_paths(config: &AppConfig) -> Vec<PathBuf> {
    let mut paths = vec![config.maa_library_path.clone()];

    if let Ok(current) = std::env::current_dir() {
        paths.push(current.join("MaaFramework.dll"));
        paths.push(current.join("maafw").join("MaaFramework.dll"));
    }

    paths
}

fn extract_digits(text: &str) -> String {
    text.lines()
        .filter_map(candidate_digits)
        .max_by_key(|digits| digits.len())
        .or_else(|| candidate_digits(text))
        .unwrap_or_default()
}

fn temp_image_path(prefix: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{now}.png"))
}

fn candidate_digits(text: &str) -> Option<String> {
    let stripped = text
        .trim()
        .trim_matches('`')
        .replace("\\n", " ")
        .replace('\r', " ");
    let digits: String = stripped
        .chars()
        .filter(|char| char.is_ascii_digit())
        .collect();
    (!digits.is_empty()).then_some(digits)
}

fn wait_with_timeout(child: &mut Child, timeout: Duration) -> Result<std::process::Output, String> {
    let start = Instant::now();
    loop {
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timed out after {}s", timeout.as_secs()));
        }

        match child.try_wait() {
            Ok(Some(status)) => return collect_output(child, status),
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("failed while waiting for process: {error}"));
            }
        }
    }
}

fn collect_output(child: &mut Child, status: ExitStatus) -> Result<std::process::Output, String> {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();

    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_end(&mut stdout)
            .map_err(|error| format!("failed to read stdout: {error}"))?;
    }
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_end(&mut stderr)
            .map_err(|error| format!("failed to read stderr: {error}"))?;
    }

    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

fn summarize_cli_failure(output: &std::process::Output) -> String {
    let code = exit_code_hint(output.status);
    let stdout = clean_diagnostic(&String::from_utf8_lossy(&output.stdout));
    let stderr = clean_diagnostic(&String::from_utf8_lossy(&output.stderr));
    format!("status {code}; stdout: {stdout}; stderr: {stderr}")
}

fn clean_diagnostic(text: &str) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let compact = compact.trim();
    if compact.is_empty() {
        "(empty)".to_string()
    } else if compact.len() > 240 {
        format!("{}...", &compact[..240])
    } else {
        compact.to_string()
    }
}

fn exit_code_hint(status: ExitStatus) -> String {
    status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string())
}

fn hash_image(image: &GrayImage) -> u64 {
    let mut hasher = DefaultHasher::new();
    image.width().hash(&mut hasher);
    image.height().hash(&mut hasher);
    image.as_raw().hash(&mut hasher);
    hasher.finish()
}
