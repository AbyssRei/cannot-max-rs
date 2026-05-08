use std::path::Path;

use crate::error::{AppError, AppResult};

pub trait IInferenceBackend: Send + Sync {
    fn load_safetensors(&mut self, path: &Path) -> AppResult<()>;
    fn infer(&self, input: &[f32], input_shape: &[usize]) -> AppResult<Vec<f32>>;
    fn is_loaded(&self) -> bool;
}

pub struct InferenceBackend {
    loaded: bool,
}

impl InferenceBackend {
    pub fn new() -> Self {
        Self { loaded: false }
    }
}

impl IInferenceBackend for InferenceBackend {
    fn load_safetensors(&mut self, path: &Path) -> AppResult<()> {
        if !path.exists() {
            return Err(AppError::ModelLoad(format!("Safetensors file not found: {}", path.display())));
        }
        tracing::info!("Loading safetensors model from: {}", path.display());
        self.loaded = true;
        Ok(())
    }

    fn infer(&self, _input: &[f32], _input_shape: &[usize]) -> AppResult<Vec<f32>> {
        if !self.loaded {
            return Err(AppError::Inference("Model not loaded".into()));
        }
        Err(AppError::Inference("Inference not fully implemented yet".into()))
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }
}
