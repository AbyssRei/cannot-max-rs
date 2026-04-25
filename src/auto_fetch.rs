use crate::capture::capture_frame;
use crate::config::AppConfig;
use crate::core::{AutoFetchStats, CaptureCatalog, CaptureSource, CapturedFrame, Side};
use crate::maa_controller::MaaControllerSession;
use crate::prediction::{CandlePredictor, Predictor};
use crate::recognition::analyze_frame;
use crate::resources::ResourceStore;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[allow(dead_code)]
const MONSTER_COUNT: usize = 61;

/// Relative click points (x_ratio, y_ratio) matching Python auto_fetch
const RELATIVE_POINTS: [(f32, f32); 5] = [
    (0.9297, 0.8833), // right ALL / return home / join / start
    (0.0713, 0.8833), // left ALL
    (0.8281, 0.8833), // right gift / self-play
    (0.1640, 0.8833), // left gift
    (0.4979, 0.6324), // watch this round
];

pub struct AutoFetch {
    running: Arc<AtomicBool>,
    stats: AutoFetchStats,
}

impl AutoFetch {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            stats: AutoFetchStats::default(),
        }
    }

    pub fn start(
        &mut self,
        source: CaptureSource,
        catalog: CaptureCatalog,
        config: AppConfig,
        _game_mode: String,
        is_invest: bool,
        training_duration_secs: f64,
    ) -> Result<(), String> {
        if self.running.load(Ordering::Relaxed) {
            return Err("自动获取已在运行中".to_string());
        }

        self.running.store(true, Ordering::Relaxed);
        self.stats = AutoFetchStats::default();

        let running = self.running.clone();
        std::thread::spawn(move || {
            auto_fetch_loop(
                running,
                source,
                catalog,
                config,
                _game_mode,
                is_invest,
                training_duration_secs,
            );
        });

        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), String> {
        self.running.store(false, Ordering::Relaxed);
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> &AutoFetchStats {
        &self.stats
    }
}

fn auto_fetch_loop(
    running: Arc<AtomicBool>,
    source: CaptureSource,
    catalog: CaptureCatalog,
    config: AppConfig,
    _game_mode: String,
    is_invest: bool,
    training_duration_secs: f64,
) {
    let start_time = std::time::Instant::now();
    let resources = ResourceStore::load(&config).ok();
    let predictor = CandlePredictor::new(config.model_path.clone());

    let _total_fill_count = 0u32;
    let _incorrect_fill_count = 0u32;
    let mut current_prediction = 0.5f32;

    while running.load(Ordering::Relaxed) {
        // Check duration limit
        if training_duration_secs > 0.0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed >= training_duration_secs {
                break;
            }
        }

        // Capture screenshot
        let frame = match capture_frame(&source, &config, &catalog) {
            Ok(frame) => frame,
            Err(_) => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        };

        // Match game state templates (simplified: always try to recognize)
        let snapshot = if let Some(ref res) = resources {
            analyze_frame(&source, &frame, config.roi, res, &config)
        } else {
            continue;
        };

        // If units detected, predict
        if !snapshot.units.is_empty() {
            if let Ok(prediction) = predictor.predict(&snapshot) {
                current_prediction = prediction.right_win_rate;
            }
        }

        // Determine game state and act
        // Simplified: check if we have units (battle state) or not
        if !snapshot.units.is_empty() {
            // Battle state: click based on prediction
            if is_invest {
                if current_prediction > 0.5 {
                    // Invest right
                    click_relative(&source, &catalog, &config, RELATIVE_POINTS[0]);
                } else {
                    // Invest left
                    click_relative(&source, &catalog, &config, RELATIVE_POINTS[1]);
                }
                std::thread::sleep(std::time::Duration::from_secs(3));
            } else {
                // Watch this round
                click_relative(&source, &catalog, &config, RELATIVE_POINTS[4]);
                std::thread::sleep(std::time::Duration::from_secs(3));
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn click_relative(
    source: &CaptureSource,
    catalog: &CaptureCatalog,
    config: &AppConfig,
    point: (f32, f32),
) {
    let x = (1920.0 * point.0) as i32;
    let y = (1080.0 * point.1) as i32;

    match MaaControllerSession::for_source(source, catalog, config) {
        Ok(session) => {
            let _ = session.click(x, y);
        }
        Err(_) => {}
    }
}

/// Determine battle result from screenshot saturation analysis
#[allow(dead_code)]
fn calculate_battle_result(frame: &CapturedFrame) -> Option<Side> {
    let img = &frame.image;
    let width = img.width() as usize;
    let height = img.height() as usize;
    if width == 0 || height == 0 {
        return None;
    }

    // Get saturation of top-left and top-right corners
    let left_top = img.get_pixel(0, 0);
    let right_top = img.get_pixel((width - 1) as u32, 0);

    let sat_left = get_saturation(*left_top);
    let sat_right = get_saturation(*right_top);

    let diff = sat_left - sat_right;
    if diff.abs() <= 20.0 {
        return None;
    }

    if diff > 20.0 {
        Some(Side::Left)
    } else {
        Some(Side::Right)
    }
}

#[allow(dead_code)]
fn get_saturation(pixel: image::Rgba<u8>) -> f32 {
    let r = pixel[0] as f32 / 255.0;
    let g = pixel[1] as f32 / 255.0;
    let b = pixel[2] as f32 / 255.0;
    let cmax = r.max(g).max(b);
    let cmin = r.min(g).min(b);
    let delta = cmax - cmin;
    if cmax > 0.0 {
        (delta / cmax) * 255.0
    } else {
        0.0
    }
}
