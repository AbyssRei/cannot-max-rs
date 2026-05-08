use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{AppError, AppResult};
use crate::types::MonsterInfo;
use image::DynamicImage;

pub trait IResourceManager: Send + Sync {
    fn get_template(&self, name: &str) -> Option<&DynamicImage>;
    fn get_monster_list(&self) -> &[MonsterInfo];
    fn get_monster_by_name(&self, name: &str) -> Option<&MonsterInfo>;
    fn get_monster_index(&self, name: &str) -> Option<usize>;
    fn load_template_image(&self, path: &Path) -> AppResult<DynamicImage>;
}

pub struct ResourceManager {
    templates: HashMap<String, DynamicImage>,
    monster_database: Vec<MonsterInfo>,
    monster_name_index: HashMap<String, usize>,
    resource_root: PathBuf,
}

impl ResourceManager {
    pub fn new(resource_root: &Path) -> AppResult<Self> {
        let resource_root = resource_root.to_path_buf();
        let mut templates = HashMap::new();
        let mut monster_database = Vec::new();
        let mut monster_name_index = HashMap::new();

        let template_dir = resource_root.join("templates");
        if template_dir.exists() {
            Self::load_templates_recursive(&template_dir, &mut templates)?;
        }

        let csv_path = resource_root.join("data").join("monster_greenvine.csv");
        if csv_path.exists() {
            Self::load_monster_csv(&csv_path, &mut monster_database, &mut monster_name_index)?;
        }

        Ok(Self {
            templates,
            monster_database,
            monster_name_index,
            resource_root,
        })
    }

    fn load_templates_recursive(
        dir: &Path,
        templates: &mut HashMap<String, DynamicImage>,
    ) -> AppResult<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::load_templates_recursive(&path, templates)?;
            } else if path.extension().is_some_and(|ext| ext == "png") {
                if let Ok(img) = image::open(&path) {
                    let key = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    templates.insert(key, img);
                }
            }
        }
        Ok(())
    }

    fn load_monster_csv(
        path: &Path,
        database: &mut Vec<MonsterInfo>,
        name_index: &mut HashMap<String, usize>,
    ) -> AppResult<()> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;
        for (idx, result) in rdr.records().enumerate() {
            let record = result?;
            let id = idx as u32;
            let name = record.get(0).unwrap_or("").to_string();
            let original_name = record.get(1).unwrap_or("").to_string();
            let info = MonsterInfo {
                id,
                name: name.clone(),
                original_name,
            };
            name_index.insert(name.clone(), database.len());
            database.push(info);
        }
        Ok(())
    }

    pub fn resource_root(&self) -> &Path {
        &self.resource_root
    }
}

impl IResourceManager for ResourceManager {
    fn get_template(&self, name: &str) -> Option<&DynamicImage> {
        self.templates.get(name)
    }

    fn get_monster_list(&self) -> &[MonsterInfo] {
        &self.monster_database
    }

    fn get_monster_by_name(&self, name: &str) -> Option<&MonsterInfo> {
        self.monster_name_index
            .get(name)
            .map(|&idx| &self.monster_database[idx])
    }

    fn get_monster_index(&self, name: &str) -> Option<usize> {
        self.monster_name_index.get(name).copied()
    }

    fn load_template_image(&self, path: &Path) -> AppResult<DynamicImage> {
        image::open(path).map_err(AppError::from)
    }
}
