use std::path::{Path, PathBuf};
use crate::error::{AppError, AppResult};
use crate::types::FetchRecord;

pub trait IStorageManager: Send + Sync {
    fn load_history(&self) -> AppResult<Vec<FetchRecord>>;
    fn append_history(&self, record: FetchRecord) -> AppResult<()>;
    fn save_history(&self, records: &[FetchRecord]) -> AppResult<()>;
}

pub struct StorageManager {
    history_path: PathBuf,
}

impl StorageManager {
    pub fn new(history_path: &Path) -> Self {
        Self {
            history_path: history_path.to_path_buf(),
        }
    }
}

impl IStorageManager for StorageManager {
    fn load_history(&self) -> AppResult<Vec<FetchRecord>> {
        if !self.history_path.exists() {
            return Ok(Vec::new());
        }
        let content = std::fs::read_to_string(&self.history_path)?;
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }
        Ok(ron::from_str(&content).unwrap_or_else(|_| {
            tracing::warn!("History data corrupted, returning empty");
            Vec::new()
        }))
    }

    fn append_history(&self, record: FetchRecord) -> AppResult<()> {
        let mut records = self.load_history().unwrap_or_default();
        records.push(record);
        self.save_history(&records)
    }

    fn save_history(&self, records: &[FetchRecord]) -> AppResult<()> {
        if let Some(parent) = self.history_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = ron::ser::to_string_pretty(records, ron::ser::PrettyConfig::default())
            .map_err(|e| AppError::Storage(format!("RON serialize error: {}", e)))?;
        std::fs::write(&self.history_path, content)?;
        Ok(())
    }
}
