use crate::core::{BattleSnapshot, Roi, Side, UnitAnnotation, VisualizationOverlay};
use image::{Rgba, RgbaImage};
use crate::recognition::AVATAR_REGIONS_REL;

/// 可视化渲染器
pub struct VisualizationRenderer;

impl VisualizationRenderer {
    /// 从分析结果构建可视化标注数据
    pub fn build_overlay(
        snapshot: &BattleSnapshot,
        roi_size: (u32, u32),
    ) -> VisualizationOverlay {
        let unit_annotations: Vec<UnitAnnotation> = snapshot
            .units
            .iter()
            .map(|unit| {
                // 计算每个单位在ROI图中的边界框
                let bbox = unit_bbox_in_roi(unit.slot, roi_size);
                UnitAnnotation {
                    slot_index: unit.slot,
                    side: unit.side,
                    unit_id: unit.unit_id.clone(),
                    count: unit.count,
                    confidence: unit.confidence,
                    bbox,
                }
            })
            .collect();

        VisualizationOverlay {
            roi_rect: None,
            unit_annotations,
        }
    }

    /// 在 RgbaImage 上绘制可视化标注，返回标注后的图像
    pub fn render_overlay(
        frame: &RgbaImage,
        overlay: &VisualizationOverlay,
    ) -> RgbaImage {
        let mut annotated = frame.clone();

        // 兼容保留：如果传入roi_rect，仍可绘制
        if let Some(roi) = &overlay.roi_rect {
            draw_rect(
                &mut annotated,
                roi.x,
                roi.y,
                roi.width,
                roi.height,
                Rgba([0, 255, 0, 255]),
            );
        }

        // 绘制各单位边界框
        for annotation in &overlay.unit_annotations {
            let color = match annotation.side {
                Side::Left => Rgba([0, 100, 255, 255]),  // 蓝色 - 左方
                Side::Right => Rgba([255, 50, 50, 255]), // 红色 - 右方
            };

            draw_rect(
                &mut annotated,
                annotation.bbox.x,
                annotation.bbox.y,
                annotation.bbox.width,
                annotation.bbox.height,
                color,
            );
        }

        annotated
    }

    /// 从 ROI 图中裁剪 6 个槽位图（按 AVATAR_REGIONS_REL 顺序）
    pub fn extract_slot_images(roi_image: &RgbaImage) -> Vec<RgbaImage> {
        AVATAR_REGIONS_REL
            .iter()
            .map(|region| {
                let x = ((roi_image.width() as f32) * region.0) as u32;
                let y = ((roi_image.height() as f32) * region.1) as u32;
                let right = ((roi_image.width() as f32) * region.2) as u32;
                let bottom = ((roi_image.height() as f32) * region.3) as u32;

                let width = right.saturating_sub(x).max(1);
                let height = bottom.saturating_sub(y).max(1);

                image::imageops::crop_imm(roi_image, x, y, width, height).to_image()
            })
            .collect()
    }
}

/// 计算单个单位在ROI图中的边界框
fn unit_bbox_in_roi(slot: usize, roi_size: (u32, u32)) -> Roi {
    let (roi_w, roi_h) = roi_size;
    let index = slot.min(AVATAR_REGIONS_REL.len().saturating_sub(1));
    let region = AVATAR_REGIONS_REL[index];

    let x = ((roi_w as f32) * region.0) as u32;
    let y = ((roi_h as f32) * region.1) as u32;
    let right = ((roi_w as f32) * region.2) as u32;
    let bottom = ((roi_h as f32) * region.3) as u32;

    Roi {
        x,
        y,
        width: right.saturating_sub(x).max(1),
        height: bottom.saturating_sub(y).max(1),
    }
}

/// 在图像上绘制空心矩形框
fn draw_rect(
    image: &mut RgbaImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    color: Rgba<u8>,
) {
    let (img_w, img_h) = (image.width(), image.height());

    // 上边
    for dx in 0..w {
        let px = x + dx;
        if px < img_w && y < img_h {
            image.put_pixel(px, y, color);
        }
    }

    // 下边
    let bottom_y = y + h;
    for dx in 0..w {
        let px = x + dx;
        if px < img_w && bottom_y < img_h {
            image.put_pixel(px, bottom_y, color);
        }
    }

    // 左边
    for dy in 0..h {
        let py = y + dy;
        if x < img_w && py < img_h {
            image.put_pixel(x, py, color);
        }
    }

    // 右边
    let right_x = x + w;
    for dy in 0..h {
        let py = y + dy;
        if right_x < img_w && py < img_h {
            image.put_pixel(right_x, py, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CaptureSource, RecognizedUnit};

    fn make_test_snapshot() -> BattleSnapshot {
        BattleSnapshot {
            source: CaptureSource::Adb("127.0.0.1:5555".to_string()),
            frame_size: (1280, 720),
            roi: Some(Roi {
                x: 100,
                y: 100,
                width: 1080,
                height: 520,
            }),
            units: vec![
                RecognizedUnit {
                    side: Side::Left,
                    slot: 0,
                    unit_id: "1".to_string(),
                    count: 3,
                    confidence: 0.95,
                    count_source: "OCR".to_string(),
                    count_cached: false,
                },
                RecognizedUnit {
                    side: Side::Right,
                    slot: 1,
                    unit_id: "5".to_string(),
                    count: 2,
                    confidence: 0.88,
                    count_source: "OCR".to_string(),
                    count_cached: false,
                },
            ],
            terrain_features: Vec::new(),
            terrain_name: None,
        }
    }

    #[test]
    fn build_overlay_contains_roi_and_annotations() {
        let snapshot = make_test_snapshot();
        let overlay = VisualizationRenderer::build_overlay(&snapshot, (1080, 520));

        assert!(overlay.roi_rect.is_none());
        assert_eq!(overlay.unit_annotations.len(), 2);
        assert_eq!(overlay.unit_annotations[0].side, Side::Left);
        assert_eq!(overlay.unit_annotations[1].side, Side::Right);
    }

    #[test]
    fn render_overlay_produces_annotated_image() {
        let snapshot = make_test_snapshot();
        let overlay = VisualizationRenderer::build_overlay(&snapshot, (1080, 520));

        let frame = RgbaImage::from_pixel(1080, 520, Rgba([0, 0, 0, 255]));
        let annotated = VisualizationRenderer::render_overlay(&frame, &overlay);

        assert_eq!(annotated.width(), 1080);
        assert_eq!(annotated.height(), 520);

        // 左侧第一槽位左上角应该被绘制（蓝色）
        let left = &overlay.unit_annotations[0].bbox;
        let pixel = annotated.get_pixel(left.x, left.y);
        assert_eq!(pixel, &Rgba([0, 100, 255, 255]));
    }

    #[test]
    fn unit_bbox_left_side_slot0() {
        let bbox = unit_bbox_in_roi(0, (1080, 520));
        assert_eq!(bbox.x, 0);
        assert!(bbox.width > 0);
        assert!(bbox.height > 0);
    }

    #[test]
    fn unit_bbox_right_side_slot0() {
        let bbox = unit_bbox_in_roi(3, (1080, 520));
        assert!(bbox.x >= 680); // 1080 * 0.64
    }

    #[test]
    fn extract_slot_images_returns_six_crops() {
        let roi = RgbaImage::from_pixel(1080, 520, Rgba([12, 34, 56, 255]));
        let slots = VisualizationRenderer::extract_slot_images(&roi);

        assert_eq!(slots.len(), 6);
        assert!(slots.iter().all(|img| img.width() > 0 && img.height() > 0));
    }
}
