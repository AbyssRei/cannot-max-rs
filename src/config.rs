use crate::core::{CaptureSource, GameMode, Roi, TrainConfig};
use crate::ocr::{DeepseekCliModel, OcrBackend};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Win32InputMethodConfig {
    Seize,
    SendMessageWithCursorPos,
    SendMessageWithWindowPos,
}

impl Win32InputMethodConfig {
    pub const ALL: [Self; 3] = [
        Self::SendMessageWithCursorPos,
        Self::SendMessageWithWindowPos,
        Self::Seize,
    ];
}

impl Default for Win32InputMethodConfig {
    fn default() -> Self {
        Self::SendMessageWithCursorPos
    }
}

impl fmt::Display for Win32InputMethodConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::SendMessageWithCursorPos => {
                "SEND_MESSAGE_WITH_CURSOR_POS（默认，推荐）"
            }
            Self::SendMessageWithWindowPos => "SEND_MESSAGE_WITH_WINDOW_POS",
            Self::Seize => "SEIZE（前台占用）",
        };
        write!(f, "{label}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "AppConfig::schema_version")]
    pub schema_version: u32,
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
    #[serde(default = "Win32InputMethodConfig::default")]
    pub win32_input_method: Win32InputMethodConfig,
    #[serde(default)]
    pub train_config: TrainConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            schema_version: Self::schema_version(),
            last_capture_source: None,
            game_mode: GameMode::Pc,
            invest_mode: false,
            roi: None,
            model_path: PathBuf::from("models/cannot-max-v1.safetensors"),
            resource_root: PathBuf::from("resources"),
            maa_library_path: PathBuf::from("maafw/MaaFramework.dll"),
            ocr_model_path: PathBuf::from("maafw/model/ocr"),
            ocr_backend: OcrBackend::Maa,
            deepseek_cli_path: PathBuf::from("tools/deepseek-ocr/deepseek-ocr-cli.exe"),
            deepseek_model: DeepseekCliModel::PaddleOcrVl,
            deepseek_device: "cpu".to_string(),
            win32_input_method: Win32InputMethodConfig::default(),
            train_config: TrainConfig::default(),
        }
    }
}

impl AppConfig {
    pub const fn schema_version() -> u32 {
        2
    }

    pub fn workspace_root() -> PathBuf {
        let mut candidates = Vec::new();

        if let Ok(current) = std::env::current_dir() {
            candidates.push(current);
        }

        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                candidates.push(parent.to_path_buf());
            }
        }

        for start in candidates {
            for ancestor in start.ancestors() {
                let cargo_toml = ancestor.join("Cargo.toml");
                let resource_dir = ancestor.join("resources");

                if cargo_toml.exists() || resource_dir.exists() {
                    return ancestor.to_path_buf();
                }
            }
        }

        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };

        let loaded: Self = ron::from_str(&text).unwrap_or_default();
        loaded.migrate_if_needed()
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

    pub fn migrate_if_needed(mut self) -> Self {
        if self.schema_version != Self::schema_version() {
            self.schema_version = Self::schema_version();
        }

        let defaults = Self::default();

        if self.resource_root.as_os_str().is_empty() {
            self.resource_root = defaults.resource_root;
        }

        if self.maa_library_path.as_os_str().is_empty() {
            self.maa_library_path = defaults.maa_library_path;
        }

        if self.ocr_model_path.as_os_str().is_empty() {
            self.ocr_model_path = defaults.ocr_model_path;
        }

        if self.deepseek_cli_path.as_os_str().is_empty() {
            self.deepseek_cli_path = defaults.deepseek_cli_path;
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::{AppConfig, Win32InputMethodConfig};
    use std::path::PathBuf;

    #[test]
    fn default_paths_are_not_empty() {
        let config = AppConfig::default();
        assert!(!config.model_path.as_os_str().is_empty());
        assert!(!config.resource_root.as_os_str().is_empty());
        assert!(config.resource_root.ends_with("resources"));
        assert_eq!(config.maa_library_path, PathBuf::from("maafw/MaaFramework.dll"));
        assert_eq!(
            config.win32_input_method,
            Win32InputMethodConfig::SendMessageWithCursorPos
        );
    }

    #[test]
    fn schema_version_is_set() {
        let config = AppConfig::default();
        assert_eq!(config.schema_version, AppConfig::schema_version());
    }
}
