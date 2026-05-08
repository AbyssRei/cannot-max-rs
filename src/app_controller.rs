use std::sync::Arc;

use crate::error::{AppError, AppResult};
use crate::types::*;
use crate::infra::config_manager::{ConfigManager, IConfigManager};
use crate::infra::storage_manager::StorageManager;
use crate::infra::resource_manager::{ResourceManager, IResourceManager};
use crate::infra::maa_bridge::{MaaBridge, IMaaBridge};
use crate::infra::capture_provider::CaptureProvider;
use crate::infra::inference_backend::InferenceBackend;
use crate::engine::recognition::{RecognitionEngine, IRecognitionEngine};
use crate::engine::prediction::{PredictionEngine, IPredictionEngine};
use crate::engine::terrain::{TerrainClassifier, ITerrainClassifier};
use crate::engine::history::{HistoryMatcher, IHistoryMatcher};
use crate::engine::data_exporter::{DataExporter, IDataExporter};
use crate::service::device_manager::DeviceManager;

pub struct AppController {
    config_manager: Arc<ConfigManager>,
    storage_manager: Arc<StorageManager>,
    resource_manager: Arc<ResourceManager>,
    maa: Arc<MaaBridge>,
    capture: Arc<CaptureProvider>,
    device_manager: Arc<DeviceManager>,
    recognition: Arc<dyn IRecognitionEngine>,
    prediction: Arc<dyn IPredictionEngine>,
    terrain: Arc<dyn ITerrainClassifier>,
    history: Arc<dyn IHistoryMatcher>,
    data_exporter: Arc<dyn IDataExporter>,
    config: std::sync::Mutex<AppConfig>,
    current_recognition: std::sync::Mutex<Option<RecognitionResult>>,
    current_roi: std::sync::Mutex<RoiRegion>,
}

impl AppController {
    pub fn new(resource_root: &std::path::Path) -> AppResult<Self> {
        let config_path = resource_root.join("config.ron");
        let history_path = resource_root.join("data").join("history.ron");

        let config_manager = Arc::new(ConfigManager::new(&config_path));
        let storage_manager = Arc::new(StorageManager::new(&history_path));
        let resource_manager = Arc::new(ResourceManager::new(resource_root)?);
        let maa = Arc::new(MaaBridge::new());
        let capture = Arc::new(CaptureProvider::new(maa.clone()));
        let device_manager = Arc::new(DeviceManager::new(maa.clone(), capture.clone()));

        let config = config_manager.load().unwrap_or_default();

        maa.init(&std::path::PathBuf::from(&config.maa_ocr_resource_dir)).ok();

        let recognition = Arc::new(RecognitionEngine::new(
            maa.clone(),
            resource_manager.clone(),
            0.7,
        ));

        let prediction = Arc::new(PredictionEngine::new(
            config.model_paths.iter().map(|p| p.into()).collect(),
            resource_manager.clone(),
        ));

        let terrain_backend = Arc::new(std::sync::Mutex::new(InferenceBackend::new()));
        let terrain = Arc::new(TerrainClassifier::new(terrain_backend));

        let monster_names: Vec<String> = resource_manager.get_monster_list().iter().map(|m| m.name.clone()).collect();
        let history = Arc::new(HistoryMatcher::new(storage_manager.clone(), &monster_names));

        let data_exporter = Arc::new(DataExporter::new(storage_manager.clone()));

        Ok(Self {
            config_manager,
            storage_manager,
            resource_manager,
            maa,
            capture,
            device_manager,
            recognition,
            prediction,
            terrain,
            history,
            data_exporter,
            config: std::sync::Mutex::new(config.clone()),
            current_recognition: std::sync::Mutex::new(None),
            current_roi: std::sync::Mutex::new(config.preset_roi.clone()),
        })
    }

    pub fn handle_recognize(&self) -> AppResult<RecognitionResult> {
        let image = self.device_manager.capture_screenshot()?;
        let roi = self.current_roi.lock().unwrap().clone();
        let result = self.recognition.recognize(&image, &roi)?;
        *self.current_recognition.lock().unwrap() = Some(result.clone());
        Ok(result)
    }

    pub fn handle_predict(&self) -> AppResult<Vec<PredictionResult>> {
        let recognition = self.current_recognition.lock().unwrap();
        let rec = recognition.as_ref()
            .ok_or_else(|| AppError::Prediction("No recognition result available".into()))?;
        let input = PredictionInput {
            left_monsters: rec.left.iter().map(|m| (m.name.clone(), m.count)).collect(),
            right_monsters: rec.right.iter().map(|m| (m.name.clone(), m.count)).collect(),
            left_terrain: rec.left_terrain.clone(),
            right_terrain: rec.right_terrain.clone(),
        };
        drop(recognition);
        self.prediction.predict(&input)
    }

    pub fn handle_reset(&self) {
        *self.current_recognition.lock().unwrap() = None;
    }

    pub fn handle_match_history(&self, top_k: usize) -> AppResult<Vec<HistoryMatchResult>> {
        let recognition = self.current_recognition.lock().unwrap();
        let rec = recognition.as_ref()
            .ok_or_else(|| AppError::General("No recognition result".into()))?;
        let query: Vec<(String, i32)> = rec.left.iter()
            .chain(rec.right.iter())
            .map(|m| (m.name.clone(), m.count))
            .collect();
        drop(recognition);
        self.history.match_history(&query, top_k)
    }

    pub fn handle_data_package(&self) -> AppResult<std::path::PathBuf> {
        let config = self.config.lock().unwrap();
        self.data_exporter.export_data(&std::path::PathBuf::from(&config.fetch_data_dir))
    }

    pub fn config(&self) -> std::sync::MutexGuard<'_, AppConfig> {
        self.config.lock().unwrap()
    }

    pub fn device_manager(&self) -> &Arc<DeviceManager> {
        &self.device_manager
    }

    pub fn handle_manual_input(&self, left: Vec<(String, i32)>, right: Vec<(String, i32)>, left_terrain: Vec<TerrainType>, right_terrain: Vec<TerrainType>) {
        let left_monsters: Vec<RecognizedMonster> = left.iter().map(|(name, count)| RecognizedMonster {
            name: name.clone(),
            icon: Vec::new(),
            count: *count,
            side: Side::Left,
            confidence: 1.0,
            low_confidence: false,
            count_uncertain: false,
            ocr_unavailable: false,
        }).collect();
        let right_monsters: Vec<RecognizedMonster> = right.iter().map(|(name, count)| RecognizedMonster {
            name: name.clone(),
            icon: Vec::new(),
            count: *count,
            side: Side::Right,
            confidence: 1.0,
            low_confidence: false,
            count_uncertain: false,
            ocr_unavailable: false,
        }).collect();
        *self.current_recognition.lock().unwrap() = Some(RecognitionResult {
            left: left_monsters,
            right: right_monsters,
            left_terrain,
            right_terrain,
        });
    }

    pub fn handle_refresh_devices(&self, mode: CaptureMode) -> AppResult<Vec<DeviceDescriptor>> {
        match mode {
            CaptureMode::Adb => self.device_manager.enumerate_adb_devices_with_timeout(),
            CaptureMode::Pc => self.device_manager.enumerate_pc_windows_with_timeout(),
            CaptureMode::WinCapture => self.device_manager.enumerate_capture_windows_with_timeout(),
        }
    }

    pub fn update_custom_roi(&self, roi: RoiRegion) {
        {
            let mut current = self.current_roi.lock().unwrap();
            *current = roi.clone();
        }
        if let Err(e) = self.config_manager.update(|config| {
            config.custom_roi = Some(roi.clone());
        }) {
            tracing::warn!("Failed to persist custom ROI: {}", e);
        }
    }

    pub fn resolve_roi_for_mode(&self, mode: &CaptureMode) -> AppResult<RoiRegion> {
        let config = self.config.lock().unwrap();
        crate::engine::roi_strategy::RoiStrategy::resolve_roi(mode, &config.preset_roi, config.custom_roi.as_ref())
    }

    pub fn handle_connect_device(&self, device: DeviceDescriptor, screencap_method: u64, mouse_method: u64, keyboard_method: u64) -> AppResult<()> {
        self.device_manager.connect(device, screencap_method, mouse_method, keyboard_method)
    }
}
