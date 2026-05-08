use std::sync::Arc;

use crate::error::{AppError, AppResult};
use crate::types::{CaptureMode, DeviceDescriptor};
use image::DynamicImage;

pub trait ICaptureProvider: Send + Sync {
    fn capture(&self, mode: CaptureMode, identifier: &str) -> AppResult<DynamicImage>;
    fn enumerate_windows(&self) -> AppResult<Vec<DeviceDescriptor>>;
}

pub struct CaptureProvider {
    maa: Arc<dyn crate::infra::maa_bridge::IMaaBridge>,
}

impl CaptureProvider {
    pub fn new(maa: Arc<dyn crate::infra::maa_bridge::IMaaBridge>) -> Self {
        Self { maa }
    }
}

impl ICaptureProvider for CaptureProvider {
    fn capture(&self, mode: CaptureMode, _identifier: &str) -> AppResult<DynamicImage> {
        match mode {
            CaptureMode::Adb | CaptureMode::Pc | CaptureMode::WinCapture => self.maa.screenshot(),
        }
    }

    fn enumerate_windows(&self) -> AppResult<Vec<DeviceDescriptor>> {
        match maa_framework::toolkit::Toolkit::find_desktop_windows() {
            Ok(windows) => {
                let result: Vec<DeviceDescriptor> = windows
                    .into_iter()
                    .filter(|w| !w.window_name.is_empty())
                    .map(|w| DeviceDescriptor {
                        name: w.window_name.clone(),
                        address: format!("{}", w.hwnd),
                        adb_path: None,
                        hwnd: Some(w.hwnd as u64),
                        mode: CaptureMode::WinCapture,
                    })
                    .collect();
                Ok(result)
            }
            Err(e) => {
                tracing::warn!("Failed to enumerate windows: {:?}", e);
                Ok(Vec::new())
            }
        }
    }
}
