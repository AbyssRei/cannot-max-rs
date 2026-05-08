use std::sync::Arc;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::error::{AppError, AppResult};
use crate::infra::inference_backend::IInferenceBackend;
use crate::infra::resource_manager::IResourceManager;
use crate::types::{PredictionInput, PredictionResult};

pub trait IPredictionEngine: Send + Sync {
    fn predict(&self, input: &PredictionInput) -> AppResult<Vec<PredictionResult>>;
}

pub struct PredictionEngine {
    backends: Vec<Arc<std::sync::Mutex<dyn IInferenceBackend>>>,
    model_paths: Vec<PathBuf>,
    resource: Arc<dyn IResourceManager>,
}

impl PredictionEngine {
    pub fn new(
        model_paths: Vec<PathBuf>,
        resource: Arc<dyn IResourceManager>,
    ) -> Self {
        let backends: Vec<Arc<std::sync::Mutex<dyn IInferenceBackend>>> = model_paths
            .iter()
            .map(|_| Arc::new(std::sync::Mutex::new(crate::infra::inference_backend::InferenceBackend::new())) as Arc<std::sync::Mutex<dyn IInferenceBackend>>)
            .collect();
        Self {
            backends,
            model_paths,
            resource,
        }
    }

    pub fn load_models(&self) -> Vec<AppResult<()>> {
        self.backends
            .iter()
            .zip(self.model_paths.iter())
            .map(|(backend, path)| {
                let mut b = backend.lock().unwrap();
                b.load_safetensors(path)
            })
            .collect()
    }

    fn encode_input(&self, input: &PredictionInput) -> Vec<f32> {
        let monster_count = self.resource.get_monster_list().len();
        let mut features = vec![0.0_f32; monster_count * 2 + 12];
        for (name, count) in &input.left_monsters {
            if let Some(idx) = self.resource.get_monster_index(name) {
                if idx < monster_count {
                    features[idx] = *count as f32;
                }
            }
        }
        for (name, count) in &input.right_monsters {
            if let Some(idx) = self.resource.get_monster_index(name) {
                if idx < monster_count {
                    features[monster_count + idx] = *count as f32;
                }
            }
        }
        features
    }
}

impl IPredictionEngine for PredictionEngine {
    fn predict(&self, input: &PredictionInput) -> AppResult<Vec<PredictionResult>> {
        if input.left_monsters.is_empty() || input.right_monsters.is_empty() {
            return Err(AppError::Prediction("Both sides must have monsters".into()));
        }
        let features = self.encode_input(input);
        let mut results = Vec::new();
        for (serial, backend) in self.backends.iter().enumerate() {
            let b = backend.lock().unwrap();
            if !b.is_loaded() {
                results.push(PredictionResult {
                    model_serial: serial as u32 + 1,
                    confidence: 0.5,
                    terrain_included: !input.left_terrain.is_empty() || !input.right_terrain.is_empty(),
                    load_failed: true,
                    inference_timeout: false,
                });
                continue;
            }
            let start = Instant::now();
            let timeout = Duration::from_secs(5);
            match b.infer(&features, &[1, features.len()]) {
                Ok(output) => {
                    let elapsed = start.elapsed();
                    let timed_out = elapsed > timeout;
                    let conf = output.first().copied().unwrap_or(0.5);
                    results.push(PredictionResult {
                        model_serial: serial as u32 + 1,
                        confidence: if timed_out { 0.5 } else { (conf as f64).clamp(0.0, 1.0) },
                        terrain_included: !input.left_terrain.is_empty() || !input.right_terrain.is_empty(),
                        load_failed: false,
                        inference_timeout: timed_out,
                    });
                }
                Err(_) => {
                    results.push(PredictionResult {
                        model_serial: serial as u32 + 1,
                        confidence: 0.5,
                        terrain_included: false,
                        load_failed: false,
                        inference_timeout: true,
                    });
                }
            }
        }
        Ok(results)
    }
}
