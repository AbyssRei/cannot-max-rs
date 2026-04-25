use image::RgbaImage;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

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

impl CaptureCatalog {
    pub fn source_choices(&self) -> Vec<SourceChoice> {
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

        sources
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

#[cfg(test)]
mod tests {
    use super::Roi;

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
}
