use crate::capture::capture_frame;
use crate::config::AppConfig;
use crate::core::{AutoFetchStats, CaptureCatalog, CaptureSource, CapturedFrame, GameState, Side};
use crate::maa_controller::MaaControllerSession;
use crate::prediction::{CandlePredictor, Predictor};
use crate::recognition::analyze_frame;
use crate::resources::ResourceStore;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Relative click points (x_ratio, y_ratio) matching Python auto_fetch
const RELATIVE_POINTS: [(f32, f32); 5] = [
    (0.9297, 0.8833), // right ALL / return home / join / start
    (0.0713, 0.8833), // left ALL
    (0.8281, 0.8833), // right gift / self-play
    (0.1640, 0.8833), // left gift
    (0.4979, 0.6324), // watch this round
];

/// 连续空帧阈值：连续多少帧无单位才判定为战斗中
const EMPTY_FRAMES_FOR_BATTLE: u32 = 3;

/// 结算检测重试次数
const SETTLEMENT_CHECK_RETRIES: u32 = 5;

pub struct AutoFetch {
    running: Arc<AtomicBool>,
    stats: Arc<Mutex<AutoFetchStats>>,
}

impl AutoFetch {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(Mutex::new(AutoFetchStats::default())),
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
        *self.stats.lock().map_err(|e| e.to_string())? = AutoFetchStats::default();

        let running = self.running.clone();
        let stats = self.stats.clone();
        std::thread::spawn(move || {
            auto_fetch_loop(
                running,
                stats,
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

    /// 获取当前统计信息（线程安全快照）
    pub fn stats_snapshot(&self) -> AutoFetchStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

fn update_stats(stats: &Arc<Mutex<AutoFetchStats>>, f: impl FnOnce(&mut AutoFetchStats)) {
    if let Ok(mut s) = stats.lock() {
        f(&mut s);
    }
}

fn auto_fetch_loop(
    running: Arc<AtomicBool>,
    stats: Arc<Mutex<AutoFetchStats>>,
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

    let mut total_fill_count = 0u32;
    let mut incorrect_fill_count = 0u32;
    let mut current_prediction = 0.5f32;
    let mut current_state = GameState::Unknown;
    let mut last_frame_size = (1920u32, 1080u32);
    let mut consecutive_empty_frames: u32 = 0;
    let mut settlement_check_count: u32 = 0;

    while running.load(Ordering::Relaxed) {
        // Check duration limit
        if training_duration_secs > 0.0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed >= training_duration_secs {
                break;
            }
        }

        // 更新运行时长统计
        let elapsed = start_time.elapsed().as_secs_f64();
        update_stats(&stats, |s| {
            s.total_fill_count = total_fill_count;
            s.incorrect_fill_count = incorrect_fill_count;
            s.elapsed_secs = elapsed;
        });

        // Capture screenshot
        let frame = match capture_frame(&source, &config, &catalog) {
            Ok(frame) => frame,
            Err(_) => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        };

        last_frame_size = (frame.image.width(), frame.image.height());

        // 判断游戏状态（基于识别结果）
        let snapshot = if let Some(ref res) = resources {
            analyze_frame(&source, &frame, config.roi, res, &config)
        } else {
            continue;
        };

        // 改进的状态判断逻辑
        if !snapshot.units.is_empty() {
            // 有单位识别结果
            consecutive_empty_frames = 0;
            settlement_check_count = 0;
            match current_state {
                GameState::InBattle | GameState::Settlement => {
                    // 战斗中/结算后重新出现单位 → 新一轮战前
                    current_state = GameState::PreBattle;
                }
                GameState::Unknown | GameState::MainMenu | GameState::Finished => {
                    // 未知/主菜单/完成后出现单位 → 战前
                    current_state = GameState::PreBattle;
                }
                _ => {
                    current_state = GameState::PreBattle;
                }
            }
        } else {
            // 无单位识别结果
            consecutive_empty_frames += 1;
            match current_state {
                GameState::PreBattle => {
                    // 从战前进入战斗中（连续空帧超过阈值才判定）
                    if consecutive_empty_frames >= EMPTY_FRAMES_FOR_BATTLE {
                        current_state = GameState::InBattle;
                    }
                }
                GameState::InBattle => {
                    // 战斗中单位持续为空，检查是否进入结算
                    settlement_check_count += 1;
                    if settlement_check_count >= SETTLEMENT_CHECK_RETRIES {
                        if let Some(_result) = calculate_battle_result(&frame) {
                            current_state = GameState::Settlement;
                            settlement_check_count = 0;
                        }
                    }
                }
                GameState::Unknown => {
                    // 未知状态且无单位，尝试推进到主菜单
                    current_state = GameState::MainMenu;
                }
                _ => {}
            }
        }

        // 根据状态执行对应操作
        match current_state {
            GameState::MainMenu => {
                // 主菜单：点击加入赛事
                click_relative(&source, &catalog, &config, RELATIVE_POINTS[0], last_frame_size);
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            GameState::ModeSelectionUnselected | GameState::ModeSelectionSelected => {
                // 模式选择：点击自娱自乐
                click_relative(&source, &catalog, &config, RELATIVE_POINTS[2], last_frame_size);
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            GameState::PreBattle => {
                // 战前：执行识别并预测
                if let Ok(prediction) = predictor.predict(&snapshot) {
                    current_prediction = prediction.right_win_rate;
                }

                // 根据预测结果投资或观战
                if is_invest {
                    if current_prediction > 0.5 {
                        click_relative(&source, &catalog, &config, RELATIVE_POINTS[0], last_frame_size);
                    } else {
                        click_relative(&source, &catalog, &config, RELATIVE_POINTS[1], last_frame_size);
                    }
                    total_fill_count += 1;
                    std::thread::sleep(std::time::Duration::from_secs(3));
                } else {
                    click_relative(&source, &catalog, &config, RELATIVE_POINTS[4], last_frame_size);
                    std::thread::sleep(std::time::Duration::from_secs(3));
                }
                current_state = GameState::InBattle;
            }
            GameState::InBattle => {
                // 战斗中：等待战斗结束
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            GameState::Settlement => {
                // 结算：判断胜负并记录
                if let Some(result) = calculate_battle_result(&frame) {
                    if result == Side::Left && current_prediction > 0.5 {
                        incorrect_fill_count += 1;
                    } else if result == Side::Right && current_prediction <= 0.5 {
                        incorrect_fill_count += 1;
                    }
                }
                // 点击返回
                click_relative(&source, &catalog, &config, RELATIVE_POINTS[0], last_frame_size);
                std::thread::sleep(std::time::Duration::from_secs(3));
                current_state = GameState::Finished;
            }
            GameState::Finished => {
                // 完成：回到主菜单
                current_state = GameState::MainMenu;
            }
            GameState::Unknown => {
                // 未知状态：尝试点击右下角继续
                click_relative(&source, &catalog, &config, RELATIVE_POINTS[0], last_frame_size);
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // 最终更新统计
    let elapsed = start_time.elapsed().as_secs_f64();
    update_stats(&stats, |s| {
        s.total_fill_count = total_fill_count;
        s.incorrect_fill_count = incorrect_fill_count;
        s.elapsed_secs = elapsed;
    });
}

fn click_relative(
    source: &CaptureSource,
    catalog: &CaptureCatalog,
    config: &AppConfig,
    point: (f32, f32),
    frame_size: (u32, u32),
) {
    let x = (frame_size.0 as f32 * point.0) as i32;
    let y = (frame_size.1 as f32 * point.1) as i32;

    match MaaControllerSession::for_source(source, catalog, config) {
        Ok(session) => {
            let _ = session.click(x, y);
        }
        Err(_) => {}
    }
}

/// Determine battle result from screenshot saturation analysis
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
