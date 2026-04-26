use crate::config::AppConfig;
use crate::core::{BattleSnapshot, CaptureSource, CapturedFrame, RecognizedUnit, RelativeRoi, Roi, Side};
use crate::ocr::recognize_count;
use crate::resources::ResourceStore;
use image::imageops::FilterType;
use image::{GrayImage, RgbaImage};

/// 6个头像区域（相对比例，基于战斗区域内部坐标）
pub const AVATAR_REGIONS_REL: [(f32, f32, f32, f32); 6] = [
    (0.0000, 0.10, 0.1200, 0.77),
    (0.1200, 0.10, 0.2400, 0.77),
    (0.2400, 0.10, 0.3600, 0.77),
    (0.6400, 0.10, 0.7600, 0.77),
    (0.7600, 0.10, 0.8800, 0.77),
    (0.8800, 0.10, 1.0000, 0.77),
];

/// 6个数字区域（相对比例，基于战斗区域内部坐标）
pub const COUNT_REGIONS_REL: [(f32, f32, f32, f32); 6] = [
    (0.0300, 0.70, 0.1400, 1.00),
    (0.1600, 0.70, 0.2700, 1.00),
    (0.2900, 0.70, 0.4000, 1.00),
    (0.6100, 0.70, 0.7200, 1.00),
    (0.7300, 0.70, 0.8400, 1.00),
    (0.8600, 0.70, 0.9700, 1.00),
];

/// 战斗区域 ROI（相对比例，基于全屏坐标）
const BATTLE_ROI_REL: RelativeRoi = RelativeRoi {
    x: 0.2479,
    y: 0.8410,
    width: 0.5047,
    height: 0.1100,
};

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
        .unwrap_or_else(|| {
            // 尝试自动定位怪物区域，失败则使用默认ROI
            find_monster_zone(&frame.image)
                .map(|rel_roi| rel_roi.to_absolute(width, height).clamp(width, height))
                .unwrap_or_else(|| default_battle_roi(width, height))
        });

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
        terrain_name: None,
    }
}

fn recognize_units(
    frame: &GrayImage,
    resources: &ResourceStore,
    config: &AppConfig,
) -> Vec<RecognizedUnit> {
    let mut units = Vec::new();

    for (index, (avatar_region, count_region)) in AVATAR_REGIONS_REL.iter().zip(COUNT_REGIONS_REL.iter()).enumerate() {
        // 从独立头像区域裁切
        let ax = ((frame.width() as f32) * avatar_region.0) as u32;
        let ay = ((frame.height() as f32) * avatar_region.1) as u32;
        let aw = (((frame.width() as f32) * avatar_region.2) as u32)
            .saturating_sub(ax)
            .max(1);
        let ah = (((frame.height() as f32) * avatar_region.3) as u32)
            .saturating_sub(ay)
            .max(1);
        let avatar = image::imageops::crop_imm(frame, ax, ay, aw, ah).to_image();
        let avatar = image::imageops::resize(&avatar, 48, 48, FilterType::Triangle);

        let (unit_id, confidence) = best_template_match(&avatar, resources);
        if unit_id == "empty" || confidence < 0.45 {
            continue;
        }

        // 从独立数字区域裁切
        let cx = ((frame.width() as f32) * count_region.0) as u32;
        let cy = ((frame.height() as f32) * count_region.1) as u32;
        let cw = (((frame.width() as f32) * count_region.2) as u32)
            .saturating_sub(cx)
            .max(1);
        let ch = (((frame.height() as f32) * count_region.3) as u32)
            .saturating_sub(cy)
            .max(1);
        let count_image = image::imageops::crop_imm(frame, cx, cy, cw, ch).to_image();

        // OCR预处理：二值化+去噪+加黑框
        let count_preprocessed = preprocess_for_ocr(&count_image);

        let ocr_value = recognize_count(&count_preprocessed, config)
            .ok()
            .flatten()
            .filter(|value| value.confidence >= 0.20)
            .and_then(|value| {
                value
                    .text
                    .parse::<u32>()
                    .ok()
                    .map(|count| (count, value.source_label, value.cached))
            });
        let (count, count_source, count_cached) =
            ocr_value.unwrap_or_else(|| (estimate_count(&avatar), "基线估算".to_string(), false));

        units.push(RecognizedUnit {
            side: if index < 3 { Side::Left } else { Side::Right },
            slot: index,
            unit_id,
            count,
            confidence,
            count_source,
            count_cached,
        });
    }

    units
}

fn best_template_match(slot: &GrayImage, resources: &ResourceStore) -> (String, f32) {
    // 将灰度槽位转为 RGBA 用于 NCC 匹配
    let slot_rgba = image::DynamicImage::ImageLuma8(slot.clone()).to_rgba8();

    let mut best_id = "empty".to_string();
    let mut best_score = resources
        .empty_thumbnail
        .as_ref()
        .map(|empty| {
            let empty_rgba = image::DynamicImage::ImageLuma8(empty.clone()).to_rgba8();
            compare_ncc(&slot_rgba, &empty_rgba)
        })
        .unwrap_or(0.0);

    for template in &resources.templates {
        let template_rgba = image::DynamicImage::ImageLuma8(template.thumbnail.clone()).to_rgba8();
        let score = compare_ncc(&slot_rgba, &template_rgba);
        if score > best_score {
            best_score = score;
            best_id = template.id.to_string();
        }
    }

    (best_id, best_score)
}

fn compare_ncc(left: &RgbaImage, right: &RgbaImage) -> f32 {
    let pixels = (left.width() * left.height()) as f32;
    if pixels == 0.0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (pa, pb) in left.pixels().zip(right.pixels()) {
        for c in 0..3 {
            let a = pa[c] as f32 / 255.0;
            let b = pb[c] as f32 / 255.0;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-8 {
        return 0.0;
    }

    (dot / denom).clamp(0.0, 1.0)
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

/// OCR预处理：二值化(OTSU) + 中值滤波去噪 + 加黑框
pub fn preprocess_for_ocr(img: &GrayImage) -> GrayImage {
    // 1. OTSU二值化
    let threshold = otsu_threshold(img);
    let binary: GrayImage = image::ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let val = if img.get_pixel(x, y)[0] > threshold { 255 } else { 0 };
        image::Luma([val])
    });

    // 2. 3x3中值滤波去噪
    let denoised = median_filter_3x3(&binary);

    // 3. 加2px黑色边框
    add_black_border(&denoised, 2)
}

/// 计算OTSU最佳阈值
fn otsu_threshold(img: &GrayImage) -> u8 {
    let mut hist = [0u32; 256];
    for pixel in img.pixels() {
        hist[pixel[0] as usize] += 1;
    }
    let total = (img.width() * img.height()) as f32;

    let mut sum = 0.0f32;
    for (i, &h) in hist.iter().enumerate() {
        sum += i as f32 * h as f32;
    }

    let mut sum_b = 0.0f32;
    let mut w_b = 0.0f32;
    let mut max_var = 0.0f32;
    let mut best_thresh = 0u8;

    for (t, &h) in hist.iter().enumerate() {
        w_b += h as f32;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f32 * h as f32;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let var = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var > max_var {
            max_var = var;
            best_thresh = t as u8;
        }
    }

    best_thresh
}

/// 3x3中值滤波
fn median_filter_3x3(img: &GrayImage) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    image::ImageBuffer::from_fn(w, h, |x, y| {
        let mut vals = Vec::with_capacity(9);
        for dy in 0i32..3 {
            for dx in 0i32..3 {
                let nx = x as i32 + dx - 1;
                let ny = y as i32 + dy - 1;
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    vals.push(img.get_pixel(nx as u32, ny as u32)[0]);
                }
            }
        }
        vals.sort_unstable();
        image::Luma([vals[vals.len() / 2]])
    })
}

/// 添加黑色边框
fn add_black_border(img: &GrayImage, border: u32) -> GrayImage {
    let new_w = img.width() + 2 * border;
    let new_h = img.height() + 2 * border;
    image::ImageBuffer::from_fn(new_w, new_h, |x, y| {
        if x >= border && x < new_w - border && y >= border && y < new_h - border {
            *img.get_pixel(x - border, y - border)
        } else {
            image::Luma([0])
        }
    })
}

/// 基于边缘检测+水平投影的怪物区域自动定位
pub fn find_monster_zone(frame: &RgbaImage) -> Option<RelativeRoi> {
    let gray = image::DynamicImage::ImageRgba8(frame.clone()).to_luma8();
    let (w, h) = (gray.width(), gray.height());
    if w < 64 || h < 64 {
        return None;
    }

    // Sobel边缘检测 - 水平梯度
    let mut edge_sum = vec![0.0f32; h as usize];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (gray.get_pixel(x + 1, y)[0] as i32 - gray.get_pixel(x - 1, y)[0] as i32).abs();
            let gy = (gray.get_pixel(x, y + 1)[0] as i32 - gray.get_pixel(x, y - 1)[0] as i32).abs();
            let edge = (gx * gx + gy * gy) as f32;
            edge_sum[y as usize] += edge;
        }
    }

    // 找到边缘密度最高的水平区域（战斗信息栏通常在画面下方1/3）
    let search_start = (h as f32 * 0.6) as usize;
    let search_end = h as usize;
    if search_start >= search_end {
        return None;
    }

    // 滑动窗口找最大边缘密度区域
    let window = (h as f32 * 0.12) as usize; // 约等于战斗信息栏高度比例
    let window = window.max(1);
    let mut best_y = search_start;
    let mut best_density = 0.0f32;

    for y_start in search_start..search_end.saturating_sub(window) {
        let density: f32 = edge_sum[y_start..y_start + window].iter().sum();
        if density > best_density {
            best_density = density;
            best_y = y_start;
        }
    }

    if best_density < 1.0 {
        return None;
    }

    let roi_y = best_y as f32 / h as f32;
    let roi_h = window as f32 / h as f32;
    // 战斗信息栏通常在画面中间偏下，宽度约占50%
    Some(RelativeRoi {
        x: 0.247,
        y: roi_y,
        width: 0.508,
        height: roi_h,
    })
}

fn crop_rgba(source: &RgbaImage, roi: Roi) -> RgbaImage {
    image::imageops::crop_imm(source, roi.x, roi.y, roi.width.max(1), roi.height.max(1)).to_image()
}

pub fn default_battle_roi(width: u32, height: u32) -> Roi {
    BATTLE_ROI_REL.to_absolute(width, height).clamp(width, height)
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

    #[test]
    fn default_roi_720p_has_area() {
        let roi = default_battle_roi(1280, 720);
        assert!(roi.width > 0);
        assert!(roi.height > 0);
        // 720p 下 ROI 应该比 1080p 小但比例一致
        let roi_1080 = default_battle_roi(1920, 1080);
        let ratio_w = roi.width as f32 / roi_1080.width as f32;
        let ratio_h = roi.height as f32 / roi_1080.height as f32;
        assert!((ratio_w - 1280.0 / 1920.0).abs() < 0.02);
        assert!((ratio_h - 720.0 / 1080.0).abs() < 0.02);
    }

    #[test]
    fn default_roi_1440p_has_area() {
        let roi = default_battle_roi(2560, 1440);
        assert!(roi.width > 0);
        assert!(roi.height > 0);
    }
}
