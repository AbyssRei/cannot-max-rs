use crate::config::AppConfig;
use crate::core::{AdbDeviceInfo, CaptureCatalog, CaptureSource, CapturedFrame, DesktopWindowInfo};
use image::{ImageBuffer, Rgba, RgbaImage};
use maa_framework::buffer::MaaImageBuffer;
use maa_framework::common::{Win32InputMethod, Win32ScreencapMethod};
use maa_framework::controller::{AdbControllerBuilder, Controller};
use std::ffi::c_void;

#[derive(Debug)]
pub struct MaaControllerSession {
    controller: Controller,
}

impl MaaControllerSession {
    pub fn for_source(
        source: &CaptureSource,
        catalog: &CaptureCatalog,
        _config: &AppConfig,
    ) -> Result<Self, String> {
        match source {
            CaptureSource::Adb(address) => {
                let device = catalog
                    .find_adb(address)
                    .ok_or_else(|| format!("ADB device not found: {address}"))?;
                Self::from_adb(device, address)
            }
            CaptureSource::DesktopWindow(hwnd) => {
                let window = catalog
                    .find_window(*hwnd)
                    .ok_or_else(|| format!("window not found: {hwnd}"))?;
                Self::from_window(*hwnd, window)
            }
            CaptureSource::Monitor(index) => Err(format!(
                "monitor capture is handled separately and cannot build a MAA controller session: #{index}"
            )),
        }
    }

    pub fn from_adb(device: &AdbDeviceInfo, address: &str) -> Result<Self, String> {
        let controller = AdbControllerBuilder::new(&device.adb_path.to_string_lossy(), address)
            .build()
            .map_err(|error| error.to_string())?;

        let session = Self { controller };
        session.connect()?;
        Ok(session)
    }

    pub fn from_window(hwnd: isize, _window: &DesktopWindowInfo) -> Result<Self, String> {
        let controller = Controller::new_win32(
            hwnd as *mut c_void,
            Win32ScreencapMethod::FRAME_POOL.bits(),
            Win32InputMethod::SEIZE.bits(),
            Win32InputMethod::SEIZE.bits(),
        )
        .map_err(|error| error.to_string())?;

        let session = Self { controller };
        session.connect()?;
        Ok(session)
    }

    pub fn capture_frame(&self, note: String) -> Result<CapturedFrame, String> {
        let screencap_job = self
            .controller
            .post_screencap()
            .map_err(|error| error.to_string())?;
        let screencap_status = self.controller.wait(screencap_job);
        if !screencap_status.succeeded() {
            return Err(format!("MAA screencap failed: {screencap_status}"));
        }

        let image = self
            .controller
            .cached_image()
            .map_err(|error| error.to_string())
            .and_then(maa_image_to_rgba)?;

        Ok(CapturedFrame { image, note })
    }

    pub fn click(&self, x: i32, y: i32) -> Result<(), String> {
        let id = self
            .controller
            .post_click(x, y)
            .map_err(|error| error.to_string())?;
        self.wait_job(id, "click")
    }

    pub fn input_text(&self, text: &str) -> Result<(), String> {
        let id = self
            .controller
            .post_input_text(text)
            .map_err(|error| error.to_string())?;
        self.wait_job(id, "text input")
    }

    pub fn inactive(&self) -> Result<(), String> {
        let id = self
            .controller
            .post_inactive()
            .map_err(|error| error.to_string())?;
        self.wait_job(id, "inactive")
    }

    fn connect(&self) -> Result<(), String> {
        let id = self
            .controller
            .post_connection()
            .map_err(|error| error.to_string())?;
        self.wait_job(id, "connection")
    }

    fn wait_job(&self, id: maa_framework::common::MaaId, action: &str) -> Result<(), String> {
        let status = self.controller.wait(id);
        if status.succeeded() {
            Ok(())
        } else {
            Err(format!("MAA {action} failed: {status}"))
        }
    }
}

fn maa_image_to_rgba(buffer: MaaImageBuffer) -> Result<RgbaImage, String> {
    let dynamic = buffer
        .to_dynamic_image()
        .map_err(|error| format!("failed to decode MAA image: {error}"))?;
    Ok(dynamic.to_rgba8())
}

pub fn blank_rgba(width: u32, height: u32) -> RgbaImage {
    ImageBuffer::from_pixel(width, height, Rgba([0, 0, 0, 255]))
}