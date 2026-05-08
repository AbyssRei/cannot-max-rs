use std::sync::Arc;

use crate::error::AppResult;
use crate::infra::inference_backend::IInferenceBackend;
use crate::types::TerrainType;
use image::DynamicImage;

pub trait ITerrainClassifier: Send + Sync {
    fn classify(&self, image: &DynamicImage, roi: &crate::types::RoiRegion) -> AppResult<Vec<TerrainType>>;
}

pub struct TerrainClassifier {
    backend: Arc<std::sync::Mutex<dyn IInferenceBackend>>,
    model_loaded: std::sync::atomic::AtomicBool,
}

impl TerrainClassifier {
    pub fn new(backend: Arc<std::sync::Mutex<dyn IInferenceBackend>>) -> Self {
        Self {
            backend,
            model_loaded: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn load_model(&self, path: &std::path::Path) -> AppResult<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.load_safetensors(path)?;
        self.model_loaded.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn terrain_from_index(index: usize) -> TerrainType {
        match index {
            0 => TerrainType::Altar,
            1 => TerrainType::Block,
            2 => TerrainType::Coil,
            3 => TerrainType::Crossbow,
            4 => TerrainType::Firearm,
            _ => TerrainType::Unknown,
        }
    }
}

impl ITerrainClassifier for TerrainClassifier {
    fn classify(&self, image: &DynamicImage, roi: &crate::types::RoiRegion) -> AppResult<Vec<TerrainType>> {
        if !self.model_loaded.load(std::sync::atomic::Ordering::SeqCst) {
            tracing::warn!("Terrain model not loaded, returning empty result");
            return Ok(Vec::new());
        }
        if roi.is_valid() {
            let clipped = roi.clip_to_bounds(image.width(), image.height());
            if clipped.width > 0 && clipped.height > 0 {
                let _cropped = image.crop_imm(clipped.x, clipped.y, clipped.width, clipped.height);
            }
        }
        Ok(vec![TerrainType::Unknown])
    }
}
