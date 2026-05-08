use crate::types::{AppState, CaptureMode, FetchPhase, TerrainType};

pub fn rust_app_state_to_slint(state: &AppState) -> i32 {
    match state {
        AppState::Idle => 0,
        AppState::Recognizing => 1,
        AppState::RecognizeSuccess => 2,
        AppState::RecognizeAndPredictSuccess => 3,
        AppState::RecognizeFailed => 4,
        AppState::Predicting => 5,
        AppState::PredictSuccess => 6,
        AppState::PredictFailed => 7,
        AppState::Fetching => 8,
        AppState::FetchFailed => 9,
    }
}

pub fn slint_data_source_to_rust(source: i32) -> CaptureMode {
    match source {
        1 => CaptureMode::Adb,
        2 => CaptureMode::Pc,
        3 => CaptureMode::WinCapture,
        _ => CaptureMode::Adb,
    }
}

pub fn rust_terrain_to_slint(terrain: &TerrainType) -> i32 {
    match terrain {
        TerrainType::Unknown => 0,
        TerrainType::Altar => 1,
        TerrainType::Block => 2,
        TerrainType::Coil => 3,
        TerrainType::Crossbow => 4,
        TerrainType::Firearm => 5,
    }
}

pub fn rust_fetch_phase_to_slint(phase: &FetchPhase) -> i32 {
    match phase {
        FetchPhase::MainMenu => 0,
        FetchPhase::ModeSelect => 1,
        FetchPhase::PreBattle => 2,
        FetchPhase::InBattle => 3,
        FetchPhase::Settlement => 4,
        FetchPhase::Completed => 5,
    }
}

pub fn image_from_rgba8(width: u32, height: u32, data: &[u8]) -> slint::Image {
    let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
    let slice = buffer.make_mut_slice();
    for (i, pixel) in slice.iter_mut().enumerate() {
        let offset = i * 4;
        if offset + 3 < data.len() {
            *pixel = slint::Rgba8Pixel {
                r: data[offset],
                g: data[offset + 1],
                b: data[offset + 2],
                a: data[offset + 3],
            };
        }
    }
    slint::Image::from_rgba8(buffer)
}

pub fn image_from_path(path: &std::path::Path) -> Option<slint::Image> {
    let img = image::open(path).ok()?;
    let rgba = img.to_rgba8();
    let width = rgba.width();
    let height = rgba.height();
    Some(image_from_rgba8(width, height, rgba.as_raw()))
}

fn create_placeholder_image(r: u8, g: u8, b: u8) -> slint::Image {
    let width = 80u32;
    let height = 80u32;
    let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
    for pixel in buffer.make_mut_slice().iter_mut() {
        *pixel = slint::Rgba8Pixel { r, g, b, a: 255 };
    }
    slint::Image::from_rgba8(buffer)
}
