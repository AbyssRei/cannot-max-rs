use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Side {
    Left,
    Right,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum CaptureMode {
    Adb,
    Pc,
    WinCapture,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum TerrainType {
    Unknown,
    Altar,
    Block,
    Coil,
    Crossbow,
    Firearm,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FetchPhase {
    MainMenu,
    ModeSelect,
    PreBattle,
    InBattle,
    Settlement,
    Completed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Winner {
    Left,
    Right,
    Draw,
    Unknown,
}

impl Default for Winner {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppState {
    Idle,
    Recognizing,
    RecognizeSuccess,
    RecognizeFailed,
    Predicting,
    PredictSuccess,
    PredictFailed,
    RecognizeAndPredictSuccess,
    Fetching,
    FetchFailed,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct RoiRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl RoiRegion {
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    pub fn clip_to_bounds(&self, max_width: u32, max_height: u32) -> RoiRegion {
        let x = self.x.min(max_width.saturating_sub(1));
        let y = self.y.min(max_height.saturating_sub(1));
        let width = self.width.min(max_width.saturating_sub(x));
        let height = self.height.min(max_height.saturating_sub(y));
        RoiRegion { x, y, width, height }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct GridPosition {
    pub col: usize,
    pub row: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecognizedMonster {
    pub name: String,
    pub icon: Vec<u8>,
    pub count: i32,
    pub side: Side,
    pub confidence: f64,
    pub low_confidence: bool,
    pub count_uncertain: bool,
    pub ocr_unavailable: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RecognitionResult {
    pub left: Vec<RecognizedMonster>,
    pub right: Vec<RecognizedMonster>,
    pub left_terrain: Vec<TerrainType>,
    pub right_terrain: Vec<TerrainType>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictionInput {
    pub left_monsters: Vec<(String, i32)>,
    pub right_monsters: Vec<(String, i32)>,
    pub left_terrain: Vec<TerrainType>,
    pub right_terrain: Vec<TerrainType>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictionResult {
    pub model_serial: u32,
    pub confidence: f64,
    pub terrain_included: bool,
    pub load_failed: bool,
    pub inference_timeout: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HistoryMatchResult {
    pub left_monsters: Vec<(String, i32)>,
    pub right_monsters: Vec<(String, i32)>,
    pub terrain: Vec<TerrainType>,
    pub winner: Winner,
    pub similarity: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FetchRecord {
    pub timestamp: String,
    pub left_monsters: Vec<(String, i32)>,
    pub right_monsters: Vec<(String, i32)>,
    pub left_terrain: Vec<TerrainType>,
    pub right_terrain: Vec<TerrainType>,
    pub winner: Winner,
    pub screenshot_path: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceDescriptor {
    pub name: String,
    pub address: String,
    pub adb_path: Option<String>,
    pub hwnd: Option<u64>,
    pub mode: CaptureMode,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceConnection {
    pub device: DeviceDescriptor,
    pub connected: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MaaOption {
    pub draw_quality: u32,
    pub logging: bool,
    pub save_draw: bool,
    pub save_on_error: bool,
    pub stdout_level: u32,
}

impl Default for MaaOption {
    fn default() -> Self {
        Self {
            draw_quality: 85,
            logging: true,
            save_draw: false,
            save_on_error: true,
            stdout_level: 2,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AppConfig {
    pub capture_mode: CaptureMode,
    pub maa_option: MaaOption,
    pub model_paths: Vec<String>,
    pub terrain_model_path: String,
    pub maa_ocr_resource_dir: String,
    pub history_data_path: String,
    pub monster_data_path: String,
    pub template_dir: String,
    pub simulator_config_path: String,
    pub fetch_data_dir: String,
    pub preset_roi: RoiRegion,
    pub custom_roi: Option<RoiRegion>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            capture_mode: CaptureMode::Adb,
            maa_option: MaaOption::default(),
            model_paths: vec!["resource/models/predict/model.safetensors".into()],
            terrain_model_path: "resource/models/terrain/model.safetensors".into(),
            maa_ocr_resource_dir: "resource/maa".into(),
            history_data_path: "resource/data/arknights.csv".into(),
            monster_data_path: "resource/data/monster_greenvine.csv".into(),
            template_dir: "resource/templates".into(),
            simulator_config_path: "resource/simulator".into(),
            fetch_data_dir: "data".into(),
            preset_roi: RoiRegion { x: 0, y: 0, width: 1280, height: 720 },
            custom_roi: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct MonsterInfo {
    pub id: u32,
    pub name: String,
    pub original_name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MonsterDef {
    pub name: String,
    pub attack: f64,
    pub attack_type: String,
    pub hp: f64,
    pub defense: f64,
    pub magic_resist: f64,
    pub attack_interval: f64,
    pub attack_range: f64,
    pub move_speed: f64,
    pub special: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SimulatorConfig {
    pub left: Vec<(String, i32)>,
    pub right: Vec<(String, i32)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct MonteCarloResult {
    pub left_win_rate: f64,
    pub right_win_rate: f64,
    pub total_samples: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SimResult {
    pub winner: Winner,
    pub rounds: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FetchTransition {
    Continue,
    Retry,
    BackToMenu,
    Timeout,
    Stop,
}
