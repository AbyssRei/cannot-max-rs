use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::error::{AppError, AppResult};
use crate::types::Rect;
use image::DynamicImage;
use maa_framework::controller::Controller;

pub trait IMaaBridge: Send + Sync {
    fn init(&self, resource_dir: &Path) -> AppResult<()>;
    fn connect_adb(&self, adb_path: &str, address: &str) -> AppResult<()>;
    fn connect_win32(&self, hwnd: u64, screencap_method: u64, mouse_method: u64, keyboard_method: u64) -> AppResult<()>;
    fn disconnect(&self);
    fn screenshot(&self) -> AppResult<DynamicImage>;
    fn click(&self, x: i32, y: i32) -> AppResult<()>;
    fn swipe(&self, x1: i32, y1: i32, x2: i32, y2: i32, duration_ms: u32) -> AppResult<()>;
    fn ocr(&self, image: &DynamicImage, roi: &Rect) -> AppResult<String>;
    fn is_connected(&self) -> bool;
    fn is_ocr_available(&self) -> bool;
}

pub struct MaaBridge {
    connected: AtomicBool,
    ocr_available: AtomicBool,
    resource_dir: Mutex<Option<PathBuf>>,
    controller: Mutex<Option<Controller>>,
}

impl MaaBridge {
    pub fn new() -> Self {
        Self {
            connected: AtomicBool::new(false),
            ocr_available: AtomicBool::new(false),
            resource_dir: Mutex::new(None),
            controller: Mutex::new(None),
        }
    }
}

impl IMaaBridge for MaaBridge {
    fn init(&self, resource_dir: &Path) -> AppResult<()> {
        *self.resource_dir.lock().unwrap() = Some(resource_dir.to_path_buf());
        let pipeline_dir = resource_dir.join("pipeline");
        if pipeline_dir.exists() {
            self.ocr_available.store(true, Ordering::SeqCst);
            tracing::info!("MAA Framework initialized with OCR resource");
        } else {
            self.ocr_available.store(false, Ordering::SeqCst);
            tracing::warn!("MAA OCR resource not found, OCR will be unavailable");
        }
        Ok(())
    }

    fn connect_adb(&self, adb_path: &str, address: &str) -> AppResult<()> {
        self.disconnect();

        let controller = Controller::new_adb(
            adb_path,
            address,
            "{}",
            "",
        ).map_err(|e| AppError::Device(format!("Failed to create ADB controller: {:?}", e)))?;

        let conn_id = controller.post_connection()
            .map_err(|e| AppError::Device(format!("Failed to post ADB connection: {:?}", e)))?;
        controller.wait(conn_id);

        if !controller.connected() {
            return Err(AppError::Device("ADB connection failed".into()));
        }

        *self.controller.lock().unwrap() = Some(controller);
        self.connected.store(true, Ordering::SeqCst);
        tracing::info!("ADB connected to {}", address);
        Ok(())
    }

    fn connect_win32(&self, hwnd: u64, screencap_method: u64, mouse_method: u64, keyboard_method: u64) -> AppResult<()> {
        self.disconnect();

        let controller = Controller::new_win32(
            hwnd as *mut std::ffi::c_void,
            screencap_method,
            mouse_method,
            keyboard_method,
        ).map_err(|e| AppError::Device(format!("Failed to create Win32 controller: {:?}", e)))?;

        let conn_id = controller.post_connection()
            .map_err(|e| AppError::Device(format!("Failed to post Win32 connection: {:?}", e)))?;
        controller.wait(conn_id);

        if !controller.connected() {
            return Err(AppError::Device("Win32 connection failed".into()));
        }

        *self.controller.lock().unwrap() = Some(controller);
        self.connected.store(true, Ordering::SeqCst);
        tracing::info!("Win32 connected to hwnd {}", hwnd);
        Ok(())
    }

    fn disconnect(&self) {
        let mut ctrl = self.controller.lock().unwrap();
        if ctrl.is_some() {
            *ctrl = None;
            self.connected.store(false, Ordering::SeqCst);
            tracing::info!("Disconnected");
        }
    }

    fn screenshot(&self) -> AppResult<DynamicImage> {
        if !self.is_connected() {
            return Err(AppError::Device("Not connected".into()));
        }

        let ctrl = self.controller.lock().unwrap();
        let controller = ctrl.as_ref().ok_or_else(|| AppError::Device("No controller".into()))?;

        let screencap_id = controller.post_screencap()
            .map_err(|e| AppError::Device(format!("Failed to post screencap: {:?}", e)))?;
        controller.wait(screencap_id);

        let image_buffer = controller.cached_image()
            .map_err(|e| AppError::Device(format!("Failed to get cached image: {:?}", e)))?;

        if image_buffer.is_empty() {
            return Err(AppError::Device("Screenshot returned empty image".into()));
        }

        image_buffer.to_dynamic_image()
            .map_err(|e| AppError::Device(format!("Failed to convert screenshot to DynamicImage: {:?}", e)))
    }

    fn click(&self, x: i32, y: i32) -> AppResult<()> {
        if !self.is_connected() {
            return Err(AppError::Device("Not connected".into()));
        }

        let ctrl = self.controller.lock().unwrap();
        let controller = ctrl.as_ref().ok_or_else(|| AppError::Device("No controller".into()))?;

        let click_id = controller.post_click(x, y)
            .map_err(|e| AppError::Device(format!("Failed to post click: {:?}", e)))?;
        controller.wait(click_id);
        Ok(())
    }

    fn swipe(&self, x1: i32, y1: i32, x2: i32, y2: i32, duration_ms: u32) -> AppResult<()> {
        if !self.is_connected() {
            return Err(AppError::Device("Not connected".into()));
        }

        let ctrl = self.controller.lock().unwrap();
        let controller = ctrl.as_ref().ok_or_else(|| AppError::Device("No controller".into()))?;

        let swipe_id = controller.post_swipe(x1, y1, x2, y2, duration_ms as i32)
            .map_err(|e| AppError::Device(format!("Failed to post swipe: {:?}", e)))?;
        controller.wait(swipe_id);
        Ok(())
    }

    fn ocr(&self, _image: &DynamicImage, _roi: &Rect) -> AppResult<String> {
        if !self.is_ocr_available() {
            return Err(AppError::OcrUnavailable("MAA OCR service not available".into()));
        }
        if !self.is_connected() {
            return Err(AppError::Device("Not connected".into()));
        }
        Err(AppError::OcrUnavailable("OCR not implemented - requires MaaFramework runtime".into()))
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    fn is_ocr_available(&self) -> bool {
        self.ocr_available.load(Ordering::SeqCst)
    }
}
