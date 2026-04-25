use crate::config::AppConfig;
use crate::core::{
    AdbDeviceInfo, CaptureCatalog, CaptureSource, CapturedFrame, DesktopWindowInfo, GameMode,
    MonitorInfo,
};
use image::{ImageBuffer, Rgba, RgbaImage};
use maa_framework::buffer::MaaImageBuffer;
use maa_framework::common::{Win32InputMethod, Win32ScreencapMethod};
use maa_framework::controller::{AdbControllerBuilder, Controller};
use maa_framework::toolkit::Toolkit;
use std::ffi::c_void;
use window_enumerator::WindowEnumerator;
use windows_capture::dxgi_duplication_api::DxgiDuplicationApi;
use windows_capture::monitor::Monitor;

pub fn discover_sources(game_mode: GameMode) -> CaptureCatalog {
    let adb_devices = Toolkit::find_adb_devices()
        .unwrap_or_default()
        .into_iter()
        .map(|device| AdbDeviceInfo {
            name: device.name,
            adb_path: device.adb_path,
            address: device.address,
        })
        .collect();

    let windows = discover_windows(game_mode);
    let monitors = discover_monitors();

    CaptureCatalog {
        adb_devices,
        windows,
        monitors,
    }
}

pub fn capture_frame(
    source: &CaptureSource,
    config: &AppConfig,
    catalog: &CaptureCatalog,
) -> Result<CapturedFrame, String> {
    match source {
        CaptureSource::Adb(address) => capture_adb(address, catalog),
        CaptureSource::DesktopWindow(hwnd) => capture_window(*hwnd, catalog),
        CaptureSource::Monitor(index) => capture_monitor(*index, config),
    }
}

fn discover_windows(game_mode: GameMode) -> Vec<DesktopWindowInfo> {
    let mut enumerator = WindowEnumerator::new();
    if enumerator.enumerate_all_windows().is_err() {
        return Vec::new();
    }

    let mut windows: Vec<_> = enumerator
        .get_windows()
        .iter()
        .filter(|window| !window.title.trim().is_empty())
        .map(|window| DesktopWindowInfo {
            hwnd: window.hwnd,
            title: window.title.clone(),
            class_name: window.class_name.clone(),
            process_name: window.process_name.clone(),
        })
        .collect();

    windows.sort_by_key(|window| window_priority(window, game_mode));
    windows
}

fn window_priority(window: &DesktopWindowInfo, game_mode: GameMode) -> (u8, String) {
    let title = window.title.to_lowercase();
    let process_name = window.process_name.to_lowercase();
    let class_name = window.class_name.to_lowercase();
    let combined = format!("{title} {process_name} {class_name}");
    let is_game_window = combined.contains("arknights")
        || combined.contains("明日方舟")
        || combined.contains("hypergryph")
        || combined.contains("prts");
    let is_emulator_window = combined.contains("emu")
        || combined.contains("bluestacks")
        || combined.contains("mumu")
        || combined.contains("leidian")
        || combined.contains("nox");

    let group = match game_mode {
        GameMode::Pc if is_game_window => 0,
        GameMode::Emulator if is_emulator_window => 0,
        GameMode::Pc if is_emulator_window => 2,
        GameMode::Emulator if is_game_window => 2,
        _ => 1,
    };

    (group, window.title.to_lowercase())
}

fn discover_monitors() -> Vec<MonitorInfo> {
    Monitor::enumerate()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(offset, monitor)| MonitorInfo {
            index: offset + 1,
            name: monitor
                .name()
                .unwrap_or_else(|_| format!("Display {}", offset + 1)),
            width: monitor.width().unwrap_or(0),
            height: monitor.height().unwrap_or(0),
        })
        .collect()
}

fn capture_adb(address: &str, catalog: &CaptureCatalog) -> Result<CapturedFrame, String> {
    let device = catalog
        .find_adb(address)
        .ok_or_else(|| format!("ADB device not found: {address}"))?;
    let controller = AdbControllerBuilder::new(&device.adb_path.to_string_lossy(), address)
        .build()
        .map_err(|error| error.to_string())?;

    capture_with_maa_controller(controller, format!("ADB 截图: {address}"))
}

fn capture_window(hwnd: isize, catalog: &CaptureCatalog) -> Result<CapturedFrame, String> {
    let window = catalog
        .find_window(hwnd)
        .ok_or_else(|| format!("window not found: {hwnd}"))?;
    let controller = Controller::new_win32(
        hwnd as *mut c_void,
        Win32ScreencapMethod::FRAME_POOL.bits(),
        Win32InputMethod::SEIZE.bits(),
        Win32InputMethod::SEIZE.bits(),
    )
    .map_err(|error| error.to_string())?;

    capture_with_maa_controller(
        controller,
        format!("Windows 窗口截图: {} ({})", window.title, window.class_name),
    )
}

fn capture_with_maa_controller(
    controller: Controller,
    note: String,
) -> Result<CapturedFrame, String> {
    let connect_job = controller
        .post_connection()
        .map_err(|error| error.to_string())?;
    let connect_status = controller.wait(connect_job);
    if !connect_status.succeeded() {
        return Err(format!("MAA connection failed: {connect_status}"));
    }

    let screencap_job = controller
        .post_screencap()
        .map_err(|error| error.to_string())?;
    let screencap_status = controller.wait(screencap_job);
    if !screencap_status.succeeded() {
        return Err(format!("MAA screencap failed: {screencap_status}"));
    }

    let image = controller
        .cached_image()
        .map_err(|error| error.to_string())
        .and_then(maa_image_to_rgba)?;

    Ok(CapturedFrame { image, note })
}

fn maa_image_to_rgba(buffer: MaaImageBuffer) -> Result<RgbaImage, String> {
    let dynamic = buffer
        .to_dynamic_image()
        .map_err(|error| format!("failed to decode MAA image: {error}"))?;
    Ok(dynamic.to_rgba8())
}

fn capture_monitor(index: usize, _config: &AppConfig) -> Result<CapturedFrame, String> {
    let monitor = Monitor::from_index(index).map_err(|error| error.to_string())?;
    let note = format!(
        "显示器截图: #{} {}",
        index,
        monitor
            .name()
            .unwrap_or_else(|_| "Unknown Monitor".to_string())
    );

    let mut duplication = DxgiDuplicationApi::new(monitor).map_err(|error| error.to_string())?;
    let mut frame = duplication
        .acquire_next_frame(250)
        .map_err(|error| error.to_string())?;
    let buffer = frame.buffer().map_err(|error| error.to_string())?;
    let mut scratch = Vec::new();

    let width = buffer.width();
    let height = buffer.height();
    let packed = buffer.as_nopadding_buffer(&mut scratch);
    let rgba = buffer_to_rgba(width, height, packed);

    Ok(CapturedFrame { image: rgba, note })
}

fn buffer_to_rgba(width: u32, height: u32, bytes: &[u8]) -> RgbaImage {
    let mut image = ImageBuffer::from_pixel(width, height, Rgba([0, 0, 0, 255]));
    let stride = 4usize;

    for (pixel_index, pixel) in image.pixels_mut().enumerate() {
        let base = pixel_index * stride;
        if base + 3 >= bytes.len() {
            break;
        }

        let b = bytes[base];
        let g = bytes[base + 1];
        let r = bytes[base + 2];
        let a = bytes[base + 3];
        *pixel = Rgba([r, g, b, a]);
    }

    image
}
