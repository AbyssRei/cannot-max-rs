use crate::core::{CaptureSource, GameMode, Roi};
use crate::ocr::{DeepseekCliModel, OcrBackend};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub last_capture_source: Option<CaptureSource>,
    pub game_mode: GameMode,
    pub invest_mode: bool,
    pub roi: Option<Roi>,
    pub model_path: PathBuf,
    pub resource_root: PathBuf,
    pub maa_library_path: PathBuf,
    pub ocr_model_path: PathBuf,
    pub ocr_backend: OcrBackend,
    pub deepseek_cli_path: PathBuf,
    pub deepseek_model: DeepseekCliModel,
    pub deepseek_device: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let python_root = workspace_root.join("..").join("cannot-max-py");

        Self {
            last_capture_source: None,
            game_mode: GameMode::Pc,
            invest_mode: false,
            roi: None,
            model_path: workspace_root
                .join("models")
                .join("cannot-max-v1.safetensors"),
            resource_root: python_root,
            maa_library_path: workspace_root.join("maa").join("MaaFramework.dll"),
            ocr_model_path: workspace_root.join("maa").join("model").join("ocr"),
            ocr_backend: OcrBackend::Maa,
            deepseek_cli_path: workspace_root
                .join("tools")
                .join("deepseek-ocr")
                .join("deepseek-ocr-cli.exe"),
            deepseek_model: DeepseekCliModel::PaddleOcrVl,
            deepseek_device: "cpu".to_string(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        let path = Self::config_path();
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };

        ron::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }

        let pretty = PrettyConfig::new()
            .depth_limit(3)
            .separate_tuple_members(true)
            .enumerate_arrays(true);
        let content =
            ron::ser::to_string_pretty(self, pretty).map_err(|error| error.to_string())?;
        fs::write(path, content).map_err(|error| error.to_string())
    }

    pub fn config_path() -> PathBuf {
        let base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        base.join("cannot-max-rs").join("app.ron")
    }

    pub fn resource_exists(&self) -> bool {
        Path::new(&self.resource_root).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::AppConfig;

    #[test]
    fn default_paths_are_not_empty() {
        let config = AppConfig::default();
        assert!(!config.model_path.as_os_str().is_empty());
        assert!(!config.resource_root.as_os_str().is_empty());
    }
}
