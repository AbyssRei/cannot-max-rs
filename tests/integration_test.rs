#[test]
fn test_app_config_default() {
    let config = cannot_max_rs::types::AppConfig::default();
    assert!(!config.model_paths.is_empty());
    assert!(config.terrain_model_path.ends_with(".safetensors"));
}

#[test]
fn test_maa_bridge_init() {
    use cannot_max_rs::infra::maa_bridge::IMaaBridge;
    let bridge = cannot_max_rs::infra::maa_bridge::MaaBridge::new();
    assert!(!bridge.is_connected());
}

#[test]
fn test_battle_simulator() {
    use cannot_max_rs::engine::battle_simulator::{BattleSimulator, IBattleSimulator};
    use cannot_max_rs::types::MonsterDef;
    let left = vec![MonsterDef {
        name: "test_left".into(),
        attack: 100.0,
        attack_type: "物理".into(),
        hp: 1000.0,
        defense: 50.0,
        magic_resist: 0.0,
        attack_interval: 1.0,
        attack_range: 5.0,
        move_speed: 1.0,
        special: None,
    }];
    let right = vec![MonsterDef {
        name: "test_right".into(),
        attack: 80.0,
        attack_type: "物理".into(),
        hp: 800.0,
        defense: 30.0,
        magic_resist: 0.0,
        attack_interval: 1.0,
        attack_range: 5.0,
        move_speed: 1.0,
        special: None,
    }];
    let mut sim = BattleSimulator::new();
    sim.initialize(&left, &right).unwrap();
    let result = sim.run_until_end(5000).unwrap();
    assert_ne!(result.rounds, 0);
}

#[test]
fn test_config_manager() {
    use cannot_max_rs::infra::config_manager::ConfigManager;
    use cannot_max_rs::infra::config_manager::IConfigManager;
    use std::path::Path;
    let dir = std::env::temp_dir().join("cannot-max-rs-test-config");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_config.ron");
    let _ = std::fs::remove_file(&path);
    let mgr = ConfigManager::new(&path);
    let config = mgr.load().unwrap();
    assert!(!config.model_paths.is_empty());
}

#[test]
fn test_history_match_cosine() {
    use cannot_max_rs::engine::history::HistoryMatcher;
    use cannot_max_rs::infra::storage_manager::StorageManager;
    use cannot_max_rs::engine::history::IHistoryMatcher;
    use std::path::Path;
    use std::sync::Arc;
    let path = std::env::temp_dir().join("cannot-max-rs-test-history").join("history.ron");
    let storage = Arc::new(StorageManager::new(&path));
    let matcher = HistoryMatcher::new(storage, &["怪物A".into(), "怪物B".into()]);
    let result = matcher.match_history(&[(String::from("怪物A"), 5)], 10).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_roi_region_area_and_validity() {
    use cannot_max_rs::types::RoiRegion;
    let roi = RoiRegion { x: 10, y: 20, width: 100, height: 200 };
    assert_eq!(roi.area(), 20000);
    assert!(roi.is_valid());
    let zero_roi = RoiRegion { x: 0, y: 0, width: 0, height: 0 };
    assert!(!zero_roi.is_valid());
}

#[test]
fn test_roi_region_clip_to_bounds() {
    use cannot_max_rs::types::RoiRegion;
    let roi = RoiRegion { x: 1200, y: 600, width: 200, height: 200 };
    let clipped = roi.clip_to_bounds(1280, 720);
    assert_eq!(clipped.x, 1200);
    assert_eq!(clipped.y, 600);
    assert!(clipped.width <= 80);
    assert!(clipped.height <= 120);
}

#[test]
fn test_roi_strategy_preset_for_adb() {
    use cannot_max_rs::engine::roi_strategy::RoiStrategy;
    use cannot_max_rs::types::{CaptureMode, RoiRegion};
    let preset = RoiRegion { x: 0, y: 0, width: 1280, height: 720 };
    let custom = RoiRegion { x: 100, y: 100, width: 800, height: 600 };
    let result = RoiStrategy::resolve_roi(&CaptureMode::Adb, &preset, Some(&custom)).unwrap();
    assert_eq!(result, preset);
}

#[test]
fn test_roi_strategy_custom_for_wincapture() {
    use cannot_max_rs::engine::roi_strategy::RoiStrategy;
    use cannot_max_rs::types::{CaptureMode, RoiRegion};
    let preset = RoiRegion { x: 0, y: 0, width: 1280, height: 720 };
    let custom = RoiRegion { x: 100, y: 100, width: 800, height: 600 };
    let result = RoiStrategy::resolve_roi(&CaptureMode::WinCapture, &preset, Some(&custom)).unwrap();
    assert_eq!(result, custom);
}

#[test]
fn test_roi_strategy_fallback_when_custom_missing() {
    use cannot_max_rs::engine::roi_strategy::RoiStrategy;
    use cannot_max_rs::types::{CaptureMode, RoiRegion};
    let preset = RoiRegion { x: 0, y: 0, width: 1280, height: 720 };
    let result = RoiStrategy::resolve_roi(&CaptureMode::WinCapture, &preset, None).unwrap();
    assert_eq!(result, preset);
}

#[test]
fn test_roi_validate() {
    use cannot_max_rs::engine::roi_strategy::RoiStrategy;
    use cannot_max_rs::types::RoiRegion;
    let roi = RoiRegion { x: 100, y: 100, width: 800, height: 600 };
    let validated = RoiStrategy::validate_roi(&roi, 1280, 720).unwrap();
    assert_eq!(validated, roi);
}
