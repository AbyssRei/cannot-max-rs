use std::sync::Arc;

use crate::error::AppResult;
use crate::infra::maa_bridge::IMaaBridge;
use crate::infra::resource_manager::IResourceManager;
use crate::types::{RecognizedMonster, RecognitionResult, Rect, Side};
use image::DynamicImage;

pub trait IRecognitionEngine: Send + Sync {
    fn recognize(&self, image: &DynamicImage, roi: &crate::types::RoiRegion) -> AppResult<RecognitionResult>;
}

pub struct RecognitionEngine {
    maa: Arc<dyn IMaaBridge>,
    resource: Arc<dyn IResourceManager>,
    template_threshold: f64,
}

impl RecognitionEngine {
    pub fn new(
        maa: Arc<dyn IMaaBridge>,
        resource: Arc<dyn IResourceManager>,
        template_threshold: f64,
    ) -> Self {
        Self {
            maa,
            resource,
            template_threshold,
        }
    }

    fn recognize_monsters(&self, image: &DynamicImage) -> Vec<RecognizedMonster> {
        let mut results = Vec::new();
        let monster_list = self.resource.get_monster_list();
        for info in monster_list {
            if let Some(template) = self.resource.get_template(&info.name) {
                let confidence = Self::quick_match(image, template);
                if confidence >= self.template_threshold {
                    let count = self.recognize_count(image, &info.name);
                    let side = Side::Left;
                    results.push(RecognizedMonster {
                        name: info.name.clone(),
                        icon: Vec::new(),
                        count,
                        side,
                        confidence,
                        low_confidence: confidence < 0.8,
                        count_uncertain: false,
                        ocr_unavailable: false,
                    });
                }
            }
        }
        results
    }

    fn recognize_count(&self, _image: &DynamicImage, _monster_name: &str) -> i32 {
        let roi = Rect { x: 0.0, y: 0.0, width: 0.0, height: 0.0 };
        match self.maa.ocr(_image, &roi) {
            Ok(text) => text.trim().parse::<i32>().unwrap_or(1),
            Err(_) => 1,
        }
    }

    fn quick_match(source: &DynamicImage, template: &DynamicImage) -> f64 {
        let src_gray = source.to_luma8();
        let tpl_gray = template.to_luma8();
        let src_w = src_gray.width() as usize;
        let src_h = src_gray.height() as usize;
        let tpl_w = tpl_gray.width() as usize;
        let tpl_h = tpl_gray.height() as usize;
        if src_w < tpl_w || src_h < tpl_h {
            return 0.0;
        }
        0.5
    }
}

impl RecognitionEngine {
    fn crop_to_roi(&self, image: &DynamicImage, roi: &crate::types::RoiRegion) -> DynamicImage {
        if !roi.is_valid() {
            return image.clone();
        }
        let img_width = image.width();
        let img_height = image.height();
        let clipped = roi.clip_to_bounds(img_width, img_height);
        if clipped.width == 0 || clipped.height == 0 {
            return image.clone();
        }
        image.crop_imm(clipped.x, clipped.y, clipped.width, clipped.height)
    }
}

impl IRecognitionEngine for RecognitionEngine {
    fn recognize(&self, image: &DynamicImage, roi: &crate::types::RoiRegion) -> AppResult<RecognitionResult> {
        let cropped = self.crop_to_roi(image, roi);
        let mut all_monsters = self.recognize_monsters(&cropped);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for monster in all_monsters.drain(..) {
            match monster.side {
                Side::Left => left.push(monster),
                Side::Right => {
                    let mut m = monster;
                    m.side = Side::Right;
                    right.push(m);
                }
            }
        }
        Ok(RecognitionResult {
            left,
            right,
            left_terrain: Vec::new(),
            right_terrain: Vec::new(),
        })
    }
}
