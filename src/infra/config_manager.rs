use std::path::{Path, PathBuf};
use crate::error::{AppError, AppResult};
use crate::types::AppConfig;

pub trait IConfigManager: Send + Sync {
    fn load(&self) -> AppResult<AppConfig>;
    fn save(&self, config: &AppConfig) -> AppResult<()>;
    fn update<F>(&self, f: F) -> AppResult<AppConfig>
    where
        F: FnOnce(&mut AppConfig);
    fn config_path(&self) -> &Path;
}

pub struct ConfigManager {
    config_path: PathBuf,
}

impl ConfigManager {
    pub fn new(config_path: &Path) -> Self {
        Self {
            config_path: config_path.to_path_buf(),
        }
    }
}

impl IConfigManager for ConfigManager {
    fn load(&self) -> AppResult<AppConfig> {
        if !self.config_path.exists() {
            let default = AppConfig::default();
            self.save(&default)?;
            return Ok(default);
        }
        let content = std::fs::read_to_string(&self.config_path)?;
        ron::from_str(&content).map_err(|e| AppError::Config(format!("RON parse error: {}", e)))
    }

    fn save(&self, config: &AppConfig) -> AppResult<()> {
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = ron::ser::to_string_pretty(config, ron::ser::PrettyConfig::default())
            .map_err(|e| AppError::Config(format!("RON serialize error: {}", e)))?;
        std::fs::write(&self.config_path, content)?;
        Ok(())
    }

    fn update<F>(&self, f: F) -> AppResult<AppConfig>
    where
        F: FnOnce(&mut AppConfig),
    {
        let mut config = self.load()?;
        f(&mut config);
        self.save(&config)?;
        Ok(config)
    }

    fn config_path(&self) -> &Path {
        &self.config_path
    }
}
