use cannot_max_rs::config::AppConfig;
use cannot_max_rs::core::{CaptureSource, Roi};
use cannot_max_rs::prediction::{BaselinePredictor, CandlePredictor, Predictor};
use cannot_max_rs::resources::ResourceStore;
use cannot_max_rs::recognition::default_battle_roi;
use cannot_max_rs::special_monster::SpecialMonsterHandler;
use cannot_max_rs::core::{BattleSnapshot, Winner};

#[test]
fn analysis_pipeline_graceful_on_missing_resources() {
    let config = AppConfig::default();
    // ResourceStore::load should not panic even with missing resources
    let result = ResourceStore::load(&config);
    // It may succeed or fail, but should not panic
    let _ = result;
}

#[test]
fn default_roi_is_valid() {
    let roi = default_battle_roi(1920, 1080);
    assert!(roi.width > 0);
    assert!(roi.height > 0);
    assert!(roi.x < 1920);
    assert!(roi.y < 1080);
}

#[test]
fn baseline_predictor_works() {
    let predictor = BaselinePredictor::new();
    let snapshot = BattleSnapshot {
        source: CaptureSource::Monitor(1),
        frame_size: (1920, 1080),
        roi: None,
        units: Vec::new(),
        terrain_features: Vec::new(),
        terrain_name: None,
    };
    let result = predictor.predict(&snapshot).unwrap();
    assert!((result.left_win_rate + result.right_win_rate - 1.0).abs() < 0.001);
    assert!(result.left_win_rate >= 0.0 && result.left_win_rate <= 1.0);
    assert!(result.right_win_rate >= 0.0 && result.right_win_rate <= 1.0);
}

#[test]
fn candle_predictor_falls_back_to_baseline() {
    let predictor = CandlePredictor::new(std::path::PathBuf::from("nonexistent_model.safetensors"));
    let snapshot = BattleSnapshot {
        source: CaptureSource::Monitor(1),
        frame_size: (1920, 1080),
        roi: None,
        units: Vec::new(),
        terrain_features: Vec::new(),
        terrain_name: None,
    };
    let result = predictor.predict(&snapshot).unwrap();
    assert!((result.left_win_rate + result.right_win_rate - 1.0).abs() < 0.001);
}

#[test]
fn special_monster_handler_works() {
    let handler = SpecialMonsterHandler::new();
    let left_ids = vec!["1".to_string()];
    let right_ids: Vec<String> = Vec::new();
    let msg = handler.check_special_monsters(&left_ids, &right_ids, &Winner::Left);
    assert!(!msg.is_empty());
}

#[test]
fn config_loads_without_panic() {
    let config = AppConfig::load();
    assert_eq!(config.schema_version, AppConfig::schema_version());
}

#[test]
fn roi_clamp_works() {
    let roi = Roi {
        x: 110,
        y: 30,
        width: 50,
        height: 40,
    };
    let clamped = roi.clamp(128, 64);
    assert_eq!(clamped.x, 110);
    assert!(clamped.width <= 18);
    assert!(clamped.height <= 34);
}
