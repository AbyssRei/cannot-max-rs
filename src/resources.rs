use crate::config::AppConfig;
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct MonsterTemplate {
    pub id: u32,
    pub name: String,
    pub thumbnail: GrayImage,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceStore {
    pub root: PathBuf,
    pub templates: Vec<MonsterTemplate>,
    pub empty_thumbnail: Option<GrayImage>,
    pub skipped_templates: usize,
    pub diagnostics: Vec<String>,
    monster_count: usize,
}

impl ResourceStore {
    pub fn load(config: &AppConfig) -> Result<Self, String> {
        let root = config.resource_root.clone();
        // 优先使用 monster_greenvine.csv，回退到 monster.csv
        let csv_path = root.join("monster_greenvine.csv");
        let csv_path = if csv_path.exists() {
            csv_path
        } else {
            root.join("monster.csv")
        };
        if !csv_path.exists() {
            return Ok(Self {
                root,
                diagnostics: vec![format!(
                    "缺少资源清单 {}，已启用缺图可运行模式",
                    csv_path.display()
                )],
                ..Self::default()
            });
        }

        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .from_path(&csv_path)
            .map_err(|error| error.to_string())?;
        let headers = reader.headers().map_err(|error| error.to_string())?.clone();
        let id_idx = headers
            .iter()
            .position(|header| header == "id")
            .ok_or_else(|| "monster.csv missing 'id' column".to_string())?;
        let name_idx = headers
            .iter()
            .position(|header| header == "原始名称")
            .ok_or_else(|| "monster.csv missing '原始名称' column".to_string())?;

        let image_root = root.join("images");
        let mut templates = Vec::new();
        let mut skipped_templates = 0usize;
        let mut diagnostics = Vec::new();

        for record in reader.records() {
            let record = record.map_err(|error| error.to_string())?;
            let id = record
                .get(id_idx)
                .unwrap_or("0")
                .parse::<u32>()
                .unwrap_or_default();
            let name = record.get(name_idx).unwrap_or("").trim().to_string();
            if name.is_empty() {
                continue;
            }

            let path = image_root.join(format!("{name}.png"));
            let Some(thumbnail) = load_thumbnail(&path) else {
                skipped_templates += 1;
                continue;
            };

            templates.push(MonsterTemplate {
                id,
                name,
                thumbnail,
            });
        }

        let empty_thumbnail = load_thumbnail(&image_root.join("empty.png"));

        if templates.is_empty() {
            diagnostics.push(format!(
                "未加载到任何模板图片，当前以缺图可运行模式继续；请检查 {}",
                image_root.display()
            ));
        } else if skipped_templates > 0 {
            diagnostics.push(format!("有 {skipped_templates} 个模板图片缺失或损坏，已自动跳过"));
        }

        Ok(Self {
            root,
            monster_count: templates.len(),
            templates,
            empty_thumbnail,
            skipped_templates,
            diagnostics,
        })
    }

    pub fn monster_count(&self) -> usize {
        self.monster_count
    }

    pub fn summary(&self) -> String {
        let mode_hint = if self.templates.is_empty() {
            "缺图可运行"
        } else {
            "资源已就绪"
        };

        let diagnostics = if self.diagnostics.is_empty() {
            String::new()
        } else {
            format!(" | 提示: {}", self.diagnostics.join("；"))
        };

        format!(
            "资源目录: {} | 模板数: {} | 跳过模板: {} | 空模板: {} | {}{}",
            self.root.display(),
            self.templates.len(),
            self.skipped_templates,
            if self.empty_thumbnail.is_some() {
                "已加载"
            } else {
                "缺失"
            },
            mode_hint,
            diagnostics
        )
    }
}

fn load_thumbnail(path: &Path) -> Option<GrayImage> {
    let image = image::open(path).ok()?;
    let cropped = crop_avatar(image);
    Some(image::imageops::resize(
        &cropped,
        48,
        48,
        FilterType::Triangle,
    ))
}

fn crop_avatar(image: DynamicImage) -> GrayImage {
    let gray = image.to_luma8();
    let width = gray.width();
    let height = gray.height();

    let left = ((width as f32) * 0.18) as u32;
    let top = ((height as f32) * 0.16) as u32;
    let right = ((width as f32) * 0.82) as u32;
    let bottom = ((height as f32) * 0.80) as u32;

    image::imageops::crop_imm(
        &gray,
        left.min(width.saturating_sub(1)),
        top.min(height.saturating_sub(1)),
        right.saturating_sub(left).max(1),
        bottom.saturating_sub(top).max(1),
    )
    .to_image()
}

#[cfg(test)]
mod tests {
    use super::ResourceStore;
    use crate::config::{AppConfig, Win32InputMethodConfig};
    use crate::core::{CaptureSource, GameMode, Roi, TrainConfig};
    use crate::ocr::{DeepseekCliModel, OcrBackend};
    use std::path::PathBuf;

    fn empty_config_for_root(root: PathBuf) -> AppConfig {
        AppConfig {
            schema_version: AppConfig::schema_version(),
            last_capture_source: Some(CaptureSource::Monitor(1)),
            game_mode: GameMode::Pc,
            invest_mode: false,
            roi: Some(Roi::default()),
            model_path: PathBuf::from("model.safetensors"),
            resource_root: root,
            maa_library_path: PathBuf::from("maafw/MaaFramework.dll"),
            ocr_model_path: PathBuf::from("maafw/model/ocr"),
            ocr_backend: OcrBackend::Maa,
            deepseek_cli_path: PathBuf::from("deepseek-ocr-cli.exe"),
            deepseek_model: DeepseekCliModel::PaddleOcrVl,
            deepseek_device: "cpu".to_string(),
            win32_input_method: Win32InputMethodConfig::SendMessageWithCursorPos,
            train_config: TrainConfig::default(),
            monster_count: 60,
            field_feature_count: 0,
        }
    }

    #[test]
    fn load_is_graceful_when_resources_are_missing() {
        let root = std::env::temp_dir().join("cannot-max-rs-missing-resources");
        let _ = std::fs::remove_dir_all(&root);

        let config = empty_config_for_root(root.clone());
        let store = ResourceStore::load(&config).expect("missing resources should not fail");

        assert_eq!(store.root, root);
        assert!(store.templates.is_empty());
        assert_eq!(store.skipped_templates, 0);
        assert!(store.summary().contains("缺图可运行"));
    }
}
