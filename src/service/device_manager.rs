use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::{AppError, AppResult};
use crate::infra::maa_bridge::IMaaBridge;
use crate::infra::capture_provider::ICaptureProvider;
use crate::types::{CaptureMode, DeviceConnection, DeviceDescriptor};
use image::DynamicImage;

const DEVICE_DETECT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct DeviceManager {
    maa: Arc<dyn IMaaBridge>,
    capture: Arc<dyn ICaptureProvider>,
    current_connection: std::sync::Mutex<Option<DeviceConnection>>,
    monitoring: AtomicBool,
}

impl DeviceManager {
    pub fn new(maa: Arc<dyn IMaaBridge>, capture: Arc<dyn ICaptureProvider>) -> Self {
        Self {
            maa,
            capture,
            current_connection: std::sync::Mutex::new(None),
            monitoring: AtomicBool::new(false),
        }
    }

    pub fn enumerate_adb_devices(&self) -> AppResult<Vec<DeviceDescriptor>> {
        let result = std::thread::scope(|s| {
            let handle = s.spawn(|| match maa_framework::toolkit::Toolkit::find_adb_devices() {
                Ok(devices) => {
                    let result: Vec<DeviceDescriptor> = devices
                        .into_iter()
                        .map(|d| DeviceDescriptor {
                            name: d.name,
                            address: d.address,
                            adb_path: d.adb_path.to_str().map(|s| s.to_string()),
                            hwnd: None,
                            mode: CaptureMode::Adb,
                        })
                        .collect();
                    tracing::info!("Found {} ADB devices", result.len());
                    Ok(result)
                }
                Err(e) => {
                    tracing::warn!("Failed to enumerate ADB devices: {:?}", e);
                    Ok(Vec::new())
                }
            });
            match handle.join() {
                Ok(r) => r,
                Err(_) => Err(AppError::Device("ADB device enumeration failed".into())),
            }
        });

        match result {
            Ok(devices) => Ok(devices),
            Err(e) => Err(e),
        }
    }

    pub fn enumerate_adb_devices_with_timeout(&self) -> AppResult<Vec<DeviceDescriptor>> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(maa_framework::toolkit::Toolkit::find_adb_devices());
        });
        match rx.recv_timeout(DEVICE_DETECT_TIMEOUT) {
            Ok(Ok(devices)) => {
                let result: Vec<DeviceDescriptor> = devices
                    .into_iter()
                    .map(|d| DeviceDescriptor {
                        name: d.name,
                        address: d.address,
                        adb_path: d.adb_path.to_str().map(|s| s.to_string()),
                        hwnd: None,
                        mode: CaptureMode::Adb,
                    })
                    .collect();
                tracing::info!("Found {} ADB devices", result.len());
                Ok(result)
            }
            Ok(Err(e)) => {
                tracing::warn!("Failed to enumerate ADB devices: {:?}", e);
                Ok(Vec::new())
            }
            Err(_) => {
                tracing::warn!("ADB device enumeration timed out after {:?}", DEVICE_DETECT_TIMEOUT);
                Err(AppError::Device("设备检测超时".into()))
            }
        }
    }

    pub fn enumerate_pc_windows_with_timeout(&self) -> AppResult<Vec<DeviceDescriptor>> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(maa_framework::toolkit::Toolkit::find_desktop_windows());
        });
        match rx.recv_timeout(DEVICE_DETECT_TIMEOUT) {
            Ok(Ok(windows)) => {
                let keywords = ["明日方舟", "Arknights", "MuMu", "雷电模拟器", "BlueStacks"];
                let result: Vec<DeviceDescriptor> = windows
                    .into_iter()
                    .filter(|w| keywords.iter().any(|kw| w.window_name.contains(kw)))
                    .map(|w| DeviceDescriptor {
                        name: w.window_name.clone(),
                        address: format!("{}", w.hwnd),
                        adb_path: None,
                        hwnd: Some(w.hwnd as u64),
                        mode: CaptureMode::Pc,
                    })
                    .collect();
                tracing::info!("Found {} PC game windows", result.len());
                Ok(result)
            }
            Ok(Err(e)) => {
                tracing::warn!("Failed to enumerate desktop windows: {:?}", e);
                Ok(Vec::new())
            }
            Err(_) => {
                tracing::warn!("PC window enumeration timed out after {:?}", DEVICE_DETECT_TIMEOUT);
                Err(AppError::Device("设备检测超时".into()))
            }
        }
    }

    pub fn enumerate_capture_windows_with_timeout(&self) -> AppResult<Vec<DeviceDescriptor>> {
        let capture = self.capture.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(capture.enumerate_windows());
        });
        match rx.recv_timeout(DEVICE_DETECT_TIMEOUT) {
            Ok(result) => result,
            Err(_) => {
                tracing::warn!("Capture window enumeration timed out after {:?}", DEVICE_DETECT_TIMEOUT);
                Err(AppError::Device("设备检测超时".into()))
            }
        }
    }

    pub fn enumerate_adb_devices_legacy(&self) -> AppResult<Vec<DeviceDescriptor>> {
        match maa_framework::toolkit::Toolkit::find_adb_devices() {
            Ok(devices) => {
                let result: Vec<DeviceDescriptor> = devices
                    .into_iter()
                    .map(|d| DeviceDescriptor {
                        name: d.name,
                        address: d.address,
                        adb_path: d.adb_path.to_str().map(|s| s.to_string()),
                        hwnd: None,
                        mode: CaptureMode::Adb,
                    })
                    .collect();
                tracing::info!("Found {} ADB devices", result.len());
                Ok(result)
            }
            Err(e) => {
                tracing::warn!("Failed to enumerate ADB devices: {:?}", e);
                Ok(Vec::new())
            }
        }
    }

    pub fn enumerate_pc_windows(&self) -> AppResult<Vec<DeviceDescriptor>> {
        match maa_framework::toolkit::Toolkit::find_desktop_windows() {
            Ok(windows) => {
                let keywords = ["明日方舟", "Arknights", "MuMu", "雷电模拟器", "BlueStacks"];
                let result: Vec<DeviceDescriptor> = windows
                    .into_iter()
                    .filter(|w| keywords.iter().any(|kw| w.window_name.contains(kw)))
                    .map(|w| DeviceDescriptor {
                        name: w.window_name.clone(),
                        address: format!("{}", w.hwnd),
                        adb_path: None,
                        hwnd: Some(w.hwnd as u64),
                        mode: CaptureMode::Pc,
                    })
                    .collect();
                tracing::info!("Found {} PC game windows", result.len());
                Ok(result)
            }
            Err(e) => {
                tracing::warn!("Failed to enumerate desktop windows: {:?}", e);
                Ok(Vec::new())
            }
        }
    }

    pub fn enumerate_capture_windows(&self) -> AppResult<Vec<DeviceDescriptor>> {
        self.capture.enumerate_windows()
    }

    pub fn connect(&self, device: DeviceDescriptor, screencap_method: u64, mouse_method: u64, keyboard_method: u64) -> AppResult<()> {
        self.disconnect()?;
        match device.mode {
            CaptureMode::Adb => {
                let adb_path = device.adb_path.as_deref().unwrap_or("adb");
                self.maa.connect_adb(adb_path, &device.address)?;
            }
            CaptureMode::Pc => {
                let hwnd = device.hwnd.unwrap_or(0);
                self.maa.connect_win32(hwnd, screencap_method, mouse_method, keyboard_method)?;
            }
            CaptureMode::WinCapture => {
                use maa_framework::common::{Win32ScreencapMethod, Win32InputMethod};
                let hwnd = device.hwnd.unwrap_or(0);
                self.maa.connect_win32(
                    hwnd,
                    Win32ScreencapMethod::FRAME_POOL.bits(),
                    Win32InputMethod::SEND_MESSAGE_WITH_CURSOR_POS.bits(),
                    Win32InputMethod::POST_MESSAGE.bits(),
                )?;
            }
        }
        let mut conn = self.current_connection.lock().unwrap();
        *conn = Some(DeviceConnection {
            device,
            connected: true,
        });
        Ok(())
    }

    pub fn disconnect(&self) -> AppResult<()> {
        let mut conn = self.current_connection.lock().unwrap();
        if conn.is_some() && conn.as_ref().unwrap().connected {
            self.maa.disconnect();
            *conn = None;
        }
        Ok(())
    }

    pub fn check_connection_alive(&self) -> bool {
        self.maa.is_connected()
    }

    pub fn capture_screenshot(&self) -> AppResult<DynamicImage> {
        let conn = self.current_connection.lock().unwrap();
        if conn.is_none() || !conn.as_ref().unwrap().connected {
            return Err(AppError::Device("Not connected".into()));
        }
        let mode = conn.as_ref().unwrap().device.mode.clone();
        let address = conn.as_ref().unwrap().device.address.clone();
        drop(conn);
        self.capture.capture(mode, &address)
    }

    pub fn is_connected(&self) -> bool {
        self.maa.is_connected()
    }

    pub fn start_monitoring(&self) {
        self.monitoring.store(true, Ordering::SeqCst);
    }

    pub fn stop_monitoring(&self) {
        self.monitoring.store(false, Ordering::SeqCst);
    }
}
