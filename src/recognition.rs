use crate::config::AppConfig;
use crate::core::{BattleSnapshot, CaptureSource, CapturedFrame, RecognizedUnit, Roi, Side};
use crate::ocr::recognize_count;
use crate::resources::ResourceStore;
use image::imageops::FilterType;
use image::{GrayImage, RgbaImage};

const UNIT_REGIONS: [(f32, f32, f32, f32); 6] = [
    (0.0000, 0.10, 0.1200, 0.77),
    (0.1200, 0.10, 0.2400, 0.77),
    (0.2400, 0.10, 0.3600, 0.77),
    (0.6400, 0.10, 0.7600, 0.77),
    (0.7600, 0.10, 0.8800, 0.77),
    (0.8800, 0.10, 1.0000, 0.77),
];

pub fn analyze_frame(
    source: &CaptureSource,
    frame: &CapturedFrame,
    roi: Option<Roi>,
    resources: &ResourceStore,
    config: &AppConfig,
) -> BattleSnapshot {
    let width = frame.image.width();
    let height = frame.image.height();
    let effective_roi = roi
        .filter(|roi| !roi.is_empty())
        .map(|roi| roi.clamp(width, height))
        .unwrap_or_else(|| default_battle_roi(width, height));

    let region = crop_rgba(&frame.image, effective_roi);
    let gray = image::DynamicImage::ImageRgba8(region.clone()).to_luma8();
    let resized = image::imageops::resize(&gray, 969, 119, FilterType::Triangle);
    let units = recognize_units(&resized, resources, config);

    BattleSnapshot {
        source: source.clone(),
        frame_size: (width, height),
        roi: Some(effective_roi),
        units,
        terrain_features: Vec::new(),
    }
}

fn recognize_units(
    frame: &GrayImage,
    resources: &ResourceStore,
    config: &AppConfig,
) -> Vec<RecognizedUnit> {
    let mut units = Vec::new();

    for (index, region) in UNIT_REGIONS.iter().enumerate() {
        let x = ((frame.width() as f32) * region.0) as u32;
        let y = ((frame.height() as f32) * region.1) as u32;
        let width = (((frame.width() as f32) * region.2) as u32)
            .saturating_sub(x)
            .max(1);
        let height = (((frame.height() as f32) * region.3) as u32)
            .saturating_sub(y)
            .max(1);
        let slot = image::imageops::crop_imm(frame, x, y, width, height).to_image();
        let slot = image::imageops::resize(&slot, 48, 48, FilterType::Triangle);

        let (unit_id, confidence) = best_template_match(&slot, resources);
        if unit_id == "empty" || confidence < 0.45 {
            continue;
        }

        let count_image = crop_count_region(&slot);
        let count = recognize_count(&count_image, config)
            .ok()
            .flatten()
            .and_then(|value| {
                if value.confidence >= 0.20 {
                    value.text.parse::<u32>().ok()
                } else {
                    None
                }
            })
            .unwrap_or_else(|| estimate_count(&slot));

        units.push(RecognizedUnit {
            side: if index < 3 { Side::Left } else { Side::Right },
            slot: index,
            unit_id,
            count,
            confidence,
        });
    }

    units
}

fn best_template_match(slot: &GrayImage, resources: &ResourceStore) -> (String, f32) {
    let mut best_id = "empty".to_string();
    let mut best_score = resources
        .empty_thumbnail
        .as_ref()
        .map(|empty| compare(slot, empty))
        .unwrap_or(0.0);

    for template in &resources.templates {
        let score = compare(slot, &template.thumbnail);
        if score > best_score {
            best_score = score;
            best_id = template.id.to_string();
        }
    }

    (best_id, best_score)
}

fn compare(left: &GrayImage, right: &GrayImage) -> f32 {
    let mut total = 0f32;
    let pixels = left.width() * left.height();

    for (left_pixel, right_pixel) in left.pixels().zip(right.pixels()) {
        let diff = (left_pixel[0] as f32 - right_pixel[0] as f32).abs();
        total += diff / 255.0;
    }

    (1.0 - (total / pixels as f32)).clamp(0.0, 1.0)
}

fn crop_count_region(slot: &GrayImage) -> GrayImage {
    let x = ((slot.width() as f32) * 0.46) as u32;
    let y = ((slot.height() as f32) * 0.56) as u32;
    let width = slot.width().saturating_sub(x).max(1);
    let height = slot.height().saturating_sub(y).max(1);
    image::imageops::crop_imm(slot, x, y, width, height).to_image()
}

fn estimate_count(slot: &GrayImage) -> u32 {
    let bright = slot.pixels().filter(|pixel| pixel[0] > 200).count() as f32;
    let ratio = bright / (slot.width() * slot.height()) as f32;

    if ratio < 0.08 {
        1
    } else if ratio < 0.16 {
        2
    } else if ratio < 0.24 {
        3
    } else if ratio < 0.32 {
        4
    } else {
        5
    }
}

fn crop_rgba(source: &RgbaImage, roi: Roi) -> RgbaImage {
    image::imageops::crop_imm(source, roi.x, roi.y, roi.width.max(1), roi.height.max(1)).to_image()
}

pub fn default_battle_roi(width: u32, height: u32) -> Roi {
    let x = (width as f32 * 0.2479) as u32;
    let y = (height as f32 * 0.8410) as u32;
    let right = (width as f32 * 0.7526) as u32;
    let bottom = (height as f32 * 0.9510) as u32;

    Roi {
        x,
        y,
        width: right.saturating_sub(x).max(1),
        height: bottom.saturating_sub(y).max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::default_battle_roi;

    #[test]
    fn default_roi_has_area() {
        let roi = default_battle_roi(1920, 1080);
        assert!(roi.width > 0);
        assert!(roi.height > 0);
    }
}
