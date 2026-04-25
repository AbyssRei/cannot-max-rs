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
}

impl ResourceStore {
    pub fn load(config: &AppConfig) -> Result<Self, String> {
        let root = config.resource_root.clone();
        let csv_path = root.join("monster.csv");
        if !csv_path.exists() {
            return Err(format!("missing resource file: {}", csv_path.display()));
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
                continue;
            };

            templates.push(MonsterTemplate {
                id,
                name,
                thumbnail,
            });
        }

        let empty_thumbnail = load_thumbnail(&image_root.join("empty.png"));

        Ok(Self {
            root,
            templates,
            empty_thumbnail,
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "资源目录: {} | 模板数: {} | 空模板: {}",
            self.root.display(),
            self.templates.len(),
            if self.empty_thumbnail.is_some() {
                "已加载"
            } else {
                "缺失"
            }
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
