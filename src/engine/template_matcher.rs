use std::collections::HashMap;
use std::path::Path;

use crate::error::AppResult;
use crate::types::Side;
use image::DynamicImage;

pub struct TemplateMatchResult {
    pub name: String,
    pub x: u32,
    pub y: u32,
    pub confidence: f64,
    pub side: Side,
}

pub struct TemplateMatcher {
    templates: HashMap<String, DynamicImage>,
}

impl TemplateMatcher {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn load_from_dir(&mut self, dir: &Path) -> AppResult<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                self.load_from_dir(&path)?;
            } else if path.extension().is_some_and(|ext| ext == "png") {
                if let Ok(img) = image::open(&path) {
                    let key = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    self.templates.insert(key, img);
                }
            }
        }
        Ok(())
    }

    pub fn match_template(
        &self,
        screenshot: &DynamicImage,
        template_name: &str,
        threshold: f64,
    ) -> Option<TemplateMatchResult> {
        let template = self.templates.get(template_name)?;
        let result = Self::template_match_ncc(screenshot, template);
        if result >= threshold {
            Some(TemplateMatchResult {
                name: template_name.to_string(),
                x: 0,
                y: 0,
                confidence: result,
                side: Side::Left,
            })
        } else {
            None
        }
    }

    pub fn match_all(
        &self,
        screenshot: &DynamicImage,
        threshold: f64,
    ) -> Vec<TemplateMatchResult> {
        let mut results = Vec::new();
        for (name, template) in &self.templates {
            let conf = Self::template_match_ncc(screenshot, template);
            if conf >= threshold {
                results.push(TemplateMatchResult {
                    name: name.clone(),
                    x: 0,
                    y: 0,
                    confidence: conf,
                    side: Side::Left,
                });
            }
        }
        results
    }

    fn template_match_ncc(source: &DynamicImage, template: &DynamicImage) -> f64 {
        let src_gray = source.to_luma8();
        let tpl_gray = template.to_luma8();
        let src_w = src_gray.width() as i32;
        let src_h = src_gray.height() as i32;
        let tpl_w = tpl_gray.width() as i32;
        let tpl_h = tpl_gray.height() as i32;
        if src_w < tpl_w || src_h < tpl_h {
            return 0.0;
        }
        let tpl_pixels: Vec<f64> = tpl_gray.iter().map(|&v| v as f64).collect();
        let tpl_mean: f64 = tpl_pixels.iter().sum::<f64>() / tpl_pixels.len() as f64;
        let tpl_norm: f64 = tpl_pixels
            .iter()
            .map(|&v| (v - tpl_mean).powi(2))
            .sum::<f64>()
            .sqrt();
        if tpl_norm < 1e-6 {
            return 0.0;
        }
        let mut best_score = 0.0_f64;
        for dy in 0..=(src_h - tpl_h) {
            for dx in 0..=(src_w - tpl_w) {
                let mut src_sum = 0.0_f64;
                let mut src_sq_sum = 0.0_f64;
                let mut cross = 0.0_f64;
                let mut count = 0usize;
                for ty in 0..tpl_h {
                    for tx in 0..tpl_w {
                        let sx = (dx + tx) as u32;
                        let sy = (dy + ty) as u32;
                        let src_val = src_gray.get_pixel(sx, sy).0[0] as f64;
                        let tpl_val = tpl_pixels[count];
                        src_sum += src_val;
                        src_sq_sum += src_val * src_val;
                        cross += src_val * tpl_val;
                        count += 1;
                    }
                }
                let n = count as f64;
                let src_mean = src_sum / n;
                let src_norm = (src_sq_sum - n * src_mean * src_mean).sqrt();
                if src_norm < 1e-6 {
                    continue;
                }
                let ncc = (cross - n * src_mean * tpl_mean) / (src_norm * tpl_norm);
                if ncc > best_score {
                    best_score = ncc;
                }
            }
        }
        best_score.max(0.0)
    }

    pub fn templates(&self) -> &HashMap<String, DynamicImage> {
        &self.templates
    }
}
