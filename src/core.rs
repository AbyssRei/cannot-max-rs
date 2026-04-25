use image::RgbaImage;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

// ── 相对比例 ROI（用于多分辨率适配）──

/// 相对比例 ROI，所有值在 0.0~1.0 范围内
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelativeRoi {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl RelativeRoi {
    /// 将相对比例 ROI 换算为绝对像素 ROI
    pub fn to_absolute(self, frame_width: u32, frame_height: u32) -> Roi {
        Roi {
            x: (frame_width as f32 * self.x) as u32,
            y: (frame_height as f32 * self.y) as u32,
            width: (frame_width as f32 * self.width) as u32,
            height: (frame_height as f32 * self.height) as u32,
        }
    }
}

// ── 游戏状态（状态机）──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameState {
    MainMenu,
    ModeSelectionUnselected,
    ModeSelectionSelected,
    PreBattle,
    InBattle,
    Settlement,
    Finished,
    Unknown,
}

impl Default for GameState {
    fn default() -> Self {
        Self::Unknown
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MainMenu => f.write_str("主菜单"),
            Self::ModeSelectionUnselected => f.write_str("模式选择(未选)"),
            Self::ModeSelectionSelected => f.write_str("模式选择(已选)"),
            Self::PreBattle => f.write_str("战前"),
            Self::InBattle => f.write_str("战斗中"),
            Self::Settlement => f.write_str("结算"),
            Self::Finished => f.write_str("已完成"),
            Self::Unknown => f.write_str("未知"),
        }
    }
}

pub type DeviceId = String;
pub type WindowId = isize;
pub type MonitorId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CaptureSource {
    Adb(DeviceId),
    DesktopWindow(WindowId),
    Monitor(MonitorId),
}

impl fmt::Display for CaptureSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adb(address) => write!(f, "ADB: {address}"),
            Self::DesktopWindow(hwnd) => write!(f, "Window HWND: {hwnd}"),
            Self::Monitor(index) => write!(f, "Monitor #{index}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameMode {
    Emulator,
    Pc,
}

impl fmt::Display for GameMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Emulator => f.write_str("模拟器模式"),
            Self::Pc => f.write_str("PC 模式"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Roi {
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn clamp(self, max_width: u32, max_height: u32) -> Self {
        let x = self.x.min(max_width.saturating_sub(1));
        let y = self.y.min(max_height.saturating_sub(1));
        let width = self.width.min(max_width.saturating_sub(x));
        let height = self.height.min(max_height.saturating_sub(y));
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => f.write_str("左"),
            Self::Right => f.write_str("右"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizedUnit {
    pub side: Side,
    pub slot: usize,
    pub unit_id: String,
    pub count: u32,
    pub confidence: f32,
    pub count_source: String,
    pub count_cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleSnapshot {
    pub source: CaptureSource,
    pub frame_size: (u32, u32),
    pub roi: Option<Roi>,
    pub units: Vec<RecognizedUnit>,
    pub terrain_features: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Winner {
    Left,
    Right,
    TossUp,
}

impl fmt::Display for Winner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => f.write_str("左方"),
            Self::Right => f.write_str("右方"),
            Self::TossUp => f.write_str("难说"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub left_win_rate: f32,
    pub right_win_rate: f32,
    pub winner: Winner,
    pub confidence_band: f32,
}

#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub image: RgbaImage,
    pub note: String,
}

#[derive(Debug, Clone, Default)]
pub struct CaptureCatalog {
    pub adb_devices: Vec<AdbDeviceInfo>,
    pub windows: Vec<DesktopWindowInfo>,
    pub monitors: Vec<MonitorInfo>,
}

#[derive(Debug, Clone)]
pub struct AdbDeviceInfo {
    pub name: String,
    pub adb_path: PathBuf,
    pub address: String,
}

#[derive(Debug, Clone)]
pub struct DesktopWindowInfo {
    pub hwnd: WindowId,
    pub title: String,
    pub class_name: String,
    pub process_name: String,
}

#[derive(Debug, Clone)]
pub struct MonitorInfo {
    pub index: MonitorId,
    pub name: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChoice {
    pub label: String,
    pub source: CaptureSource,
}

impl fmt::Display for SourceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label)
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisOutput {
    pub frame: CapturedFrame,
    pub snapshot: BattleSnapshot,
    pub prediction: PredictionResult,
}

// ── 训练相关类型 ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub data_file: PathBuf,
    pub batch_size: usize,
    pub test_size: f32,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub num_heads: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub seed: u64,
    pub save_dir: PathBuf,
    pub max_feature_value: f32,
    pub weight_decay: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_file: PathBuf::from("arknights.csv"),
            batch_size: 1024,
            test_size: 0.1,
            embed_dim: 128,
            n_layers: 3,
            num_heads: 16,
            learning_rate: 3e-4,
            epochs: 200,
            seed: 42,
            save_dir: PathBuf::from("models"),
            max_feature_value: 100.0,
            weight_decay: 1e-1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub train_loss: f32,
    pub train_acc: f32,
    pub val_loss: f32,
    pub val_acc: f32,
    pub best_acc: f32,
    pub best_loss: f32,
    pub elapsed_secs: f64,
    pub estimated_remaining_secs: f64,
    pub device_info: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainSample {
    pub left_signs: Vec<f32>,
    pub left_counts: Vec<f32>,
    pub right_signs: Vec<f32>,
    pub right_counts: Vec<f32>,
    pub label: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainResult {
    pub best_acc: f32,
    pub best_loss: f32,
    pub total_epochs: usize,
    pub data_length: usize,
    pub model_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutoFetchStats {
    pub total_fill_count: u32,
    pub incorrect_fill_count: u32,
    pub elapsed_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialMonsterInfo {
    pub name: String,
    pub win_message: String,
    pub lose_message: String,
}

impl CaptureCatalog {
    pub fn source_choices(&self, game_mode: GameMode) -> Vec<SourceChoice> {
        let mut sources = Vec::new();

        for adb in &self.adb_devices {
            sources.push(SourceChoice {
                label: format!("ADB | {} | {}", adb.name, adb.address),
                source: CaptureSource::Adb(adb.address.clone()),
            });
        }

        for window in &self.windows {
            sources.push(SourceChoice {
                label: format!("窗口 | {} | {}", window.process_name, window.title),
                source: CaptureSource::DesktopWindow(window.hwnd),
            });
        }

        for monitor in &self.monitors {
            sources.push(SourceChoice {
                label: format!(
                    "显示器 | #{} | {} ({}x{})",
                    monitor.index, monitor.name, monitor.width, monitor.height
                ),
                source: CaptureSource::Monitor(monitor.index),
            });
        }

        sources.sort_by_key(|choice| choice.priority(game_mode));
        sources
    }

    pub fn preferred_source(&self, game_mode: GameMode) -> Option<SourceChoice> {
        self.source_choices(game_mode).into_iter().next()
    }

    pub fn find_adb(&self, address: &str) -> Option<&AdbDeviceInfo> {
        self.adb_devices
            .iter()
            .find(|device| device.address == address)
    }

    pub fn find_window(&self, hwnd: WindowId) -> Option<&DesktopWindowInfo> {
        self.windows.iter().find(|window| window.hwnd == hwnd)
    }
}

impl SourceChoice {
    fn priority(&self, game_mode: GameMode) -> (u8, String) {
        let source_rank = match (&self.source, game_mode) {
            (CaptureSource::DesktopWindow(_), GameMode::Pc) => 0,
            (CaptureSource::Monitor(_), GameMode::Pc) => 1,
            (CaptureSource::Adb(_), GameMode::Emulator) => 0,
            (CaptureSource::DesktopWindow(_), GameMode::Emulator) => 1,
            (CaptureSource::Monitor(_), GameMode::Emulator) => 2,
            (CaptureSource::Adb(_), GameMode::Pc) => 2,
        };

        (source_rank, self.label.to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::{AdbDeviceInfo, CaptureCatalog, CaptureSource, DesktopWindowInfo, GameMode, MonitorInfo, Roi};

    #[test]
    fn clamp_roi_to_bounds() {
        let roi = Roi {
            x: 110,
            y: 30,
            width: 50,
            height: 40,
        };

        let clamped = roi.clamp(128, 64);

        assert_eq!(clamped.x, 110);
        assert_eq!(clamped.width, 18);
        assert_eq!(clamped.height, 34);
    }

    #[test]
    fn pc_mode_prefers_windows() {
        let catalog = CaptureCatalog {
            adb_devices: vec![],
            windows: vec![DesktopWindowInfo {
                hwnd: 42,
                title: "Arknights".to_string(),
                class_name: "UnityWndClass".to_string(),
                process_name: "Arknights.exe".to_string(),
            }],
            monitors: vec![MonitorInfo {
                index: 1,
                name: "Display 1".to_string(),
                width: 1920,
                height: 1080,
            }],
        };

        let first = catalog.preferred_source(GameMode::Pc).unwrap();
        assert!(matches!(first.source, CaptureSource::DesktopWindow(42)));
    }

    #[test]
    fn emulator_mode_prefers_adb() {
        let catalog = CaptureCatalog {
            adb_devices: vec![AdbDeviceInfo {
                name: "emu".to_string(),
                adb_path: std::path::PathBuf::from("adb.exe"),
                address: "127.0.0.1:5555".to_string(),
            }],
            windows: vec![DesktopWindowInfo {
                hwnd: 42,
                title: "Arknights".to_string(),
                class_name: "UnityWndClass".to_string(),
                process_name: "Arknights.exe".to_string(),
            }],
            monitors: vec![MonitorInfo {
                index: 1,
                name: "Display 1".to_string(),
                width: 1920,
                height: 1080,
            }],
        };

        let first = catalog.preferred_source(GameMode::Emulator).unwrap();
        assert!(matches!(first.source, CaptureSource::Adb(_)));
    }
}
