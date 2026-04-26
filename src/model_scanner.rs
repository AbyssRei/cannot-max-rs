use crate::core::{ModelEntry, ModelSelection};
use crate::config::AppConfig;
use std::fs;
use std::path::{Path, PathBuf};

/// 模型文件扫描器
pub struct ModelScanner;

impl ModelScanner {
    /// 扫描指定目录下的所有 .safetensors 模型文件
    pub fn scan(models_dir: &Path) -> Vec<ModelEntry> {
        let Ok(entries) = fs::read_dir(models_dir) else {
            return Vec::new();
        };

        let mut models: Vec<ModelEntry> = entries
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
                    let metadata = entry.metadata().ok()?;
                    Some(ModelEntry {
                        file_name: path.file_name()?.to_string_lossy().to_string(),
                        path,
                        file_size: metadata.len(),
                    })
                } else {
                    None
                }
            })
            .collect();

        models.sort_by(|a, b| a.file_name.cmp(&b.file_name));
        models
    }

    /// 获取默认 models 目录路径
    pub fn default_models_dir() -> PathBuf {
        AppConfig::workspace_root().join("models")
    }
}

/// 模型选择管理器
pub struct ModelSelector {
    pub scanned_models: Vec<ModelEntry>,
    pub selection: ModelSelection,
}

impl ModelSelector {
    pub fn new() -> Self {
        let models_dir = ModelScanner::default_models_dir();
        let scanned_models = ModelScanner::scan(&models_dir);

        let selection = if let Some(first) = scanned_models.first() {
            ModelSelection::Scanned(first.clone())
        } else {
            ModelSelection::OtherOption
        };

        Self {
            scanned_models,
            selection,
        }
    }

    /// 刷新扫描结果
    pub fn refresh(&mut self, models_dir: &Path) {
        self.scanned_models = ModelScanner::scan(models_dir);
    }

    /// 选择扫描到的模型
    pub fn select_scanned(&mut self, entry: ModelEntry) {
        self.selection = ModelSelection::Scanned(entry);
    }

    /// 选择自定义路径模型
    pub fn select_custom(&mut self, path: PathBuf) {
        self.selection = ModelSelection::Custom(path);
    }

    /// 获取当前选中的模型路径
    pub fn current_path(&self) -> Option<&PathBuf> {
        match &self.selection {
            ModelSelection::Scanned(entry) => Some(&entry.path),
            ModelSelection::Custom(path) => Some(path),
            ModelSelection::OtherOption => None,
        }
    }

    /// 构建下拉列表选项（含"选择其他模型"）
    pub fn dropdown_options(&self) -> Vec<ModelSelection> {
        let mut options: Vec<ModelSelection> = self
            .scanned_models
            .iter()
            .map(|entry| ModelSelection::Scanned(entry.clone()))
            .collect();
        options.push(ModelSelection::OtherOption);
        options
    }
}

impl Default for ModelSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn scan_empty_directory() {
        let dir = std::env::temp_dir().join("cannotmax_test_scan_empty");
        let _ = fs::create_dir_all(&dir);
        let result = ModelScanner::scan(&dir);
        assert!(result.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_directory_with_models() {
        let dir = std::env::temp_dir().join("cannotmax_test_scan_models");
        let _ = fs::create_dir_all(&dir);
        // Create fake model files
        fs::write(dir.join("model-a.safetensors"), b"fake").unwrap();
        fs::write(dir.join("model-b.safetensors"), b"fake").unwrap();
        fs::write(dir.join("not-a-model.txt"), b"fake").unwrap();

        let result = ModelScanner::scan(&dir);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].file_name, "model-a.safetensors");
        assert_eq!(result[1].file_name, "model-b.safetensors");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dropdown_options_ends_with_other() {
        let dir = std::env::temp_dir().join("cannotmax_test_dropdown");
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("test.safetensors"), b"fake").unwrap();

        let mut selector = ModelSelector::new();
        selector.refresh(&dir);
        let options = selector.dropdown_options();

        assert!(matches!(options.last(), Some(ModelSelection::OtherOption)));
        assert_eq!(options.len(), 2); // 1 scanned + 1 OtherOption

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn current_path_returns_correct_path() {
        let dir = std::env::temp_dir().join("cannotmax_test_current_path");
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("test.safetensors"), b"fake").unwrap();

        let mut selector = ModelSelector::new();
        selector.refresh(&dir);

        if let Some(entry) = selector.scanned_models.first().cloned() {
            selector.select_scanned(entry);
            assert!(selector.current_path().is_some());
        }

        selector.select_custom(PathBuf::from("/custom/path/model.safetensors"));
        assert_eq!(
            selector.current_path().unwrap(),
            Path::new("/custom/path/model.safetensors")
        );

        selector.selection = ModelSelection::OtherOption;
        assert!(selector.current_path().is_none());

        let _ = fs::remove_dir_all(&dir);
    }
}
