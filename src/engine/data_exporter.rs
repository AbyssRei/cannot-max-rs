use std::path::{Path, PathBuf};

use crate::error::{AppError, AppResult};
use crate::infra::storage_manager::IStorageManager;
use crate::types::FetchRecord;

pub trait IDataExporter: Send + Sync {
    fn export_data(&self, output_dir: &Path) -> AppResult<PathBuf>;
    fn clean_data(&self, records: &mut Vec<FetchRecord>);
}

pub struct DataExporter {
    storage: std::sync::Arc<dyn IStorageManager>,
}

impl DataExporter {
    pub fn new(storage: std::sync::Arc<dyn IStorageManager>) -> Self {
        Self { storage }
    }
}

impl IDataExporter for DataExporter {
    fn export_data(&self, output_dir: &Path) -> AppResult<PathBuf> {
        let mut records = self.storage.load_history()?;
        if records.is_empty() {
            return Err(AppError::Fetch("No data to export".into()));
        }
        self.clean_data(&mut records);
        std::fs::create_dir_all(output_dir)?;
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("data_{}.ron", timestamp);
        let output_path = output_dir.join(&filename);
        let content = ron::ser::to_string_pretty(&records, ron::ser::PrettyConfig::default())
            .map_err(|e| AppError::Storage(format!("Serialize error: {}", e)))?;
        std::fs::write(&output_path, content)?;
        Ok(output_path)
    }

    fn clean_data(&self, records: &mut Vec<FetchRecord>) {
        let mut seen = std::collections::HashSet::new();
        records.retain(|record| {
            let key = format!("{}_{:?}", record.timestamp, record.left_monsters);
            seen.insert(key)
        });
    }
}
