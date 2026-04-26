use crate::core::{BattleSnapshot, Roi, Side, UnitAnnotation, VisualizationOverlay};
use image::{Rgba, RgbaImage};

/// 可视化渲染器
pub struct VisualizationRenderer;

impl VisualizationRenderer {
    /// 从分析结果构建可视化标注数据
    pub fn build_overlay(
        snapshot: &BattleSnapshot,
        roi: Option<Roi>,
        frame_size: (u32, u32),
    ) -> VisualizationOverlay {
        let unit_annotations: Vec<UnitAnnotation> = snapshot
            .units
            .iter()
            .map(|unit| {
                // 计算每个单位在画面中的边界框
                // 使用相对比例估算单位位置
                let bbox = unit_bbox(unit.side, unit.slot, frame_size);
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
            roi_rect: roi,
            unit_annotations,
        }
    }

    /// 在 RgbaImage 上绘制可视化标注，返回标注后的图像
    pub fn render_overlay(
        frame: &RgbaImage,
        overlay: &VisualizationOverlay,
    ) -> RgbaImage {
        let mut annotated = frame.clone();

        // 绘制 ROI 矩形框（绿色）
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
}

/// 计算单个单位在画面中的边界框
fn unit_bbox(side: Side, slot: usize, frame_size: (u32, u32)) -> Roi {
    let (fw, fh) = frame_size;
    // 战斗区域大致在画面中部，左右各3个槽位
    // 每个槽位宽度约为画面宽度的 1/6
    let slot_width = fw / 6;
    let slot_height = fh / 4;
    let y_offset = fh / 3; // 从画面1/3处开始

    let x = match side {
        Side::Left => slot as u32 * slot_width,
        Side::Right => fw / 2 + slot as u32 * slot_width,
    };

    Roi {
        x,
        y: y_offset,
        width: slot_width,
        height: slot_height,
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
        }
    }

    #[test]
    fn build_overlay_contains_roi_and_annotations() {
        let snapshot = make_test_snapshot();
        let overlay = VisualizationRenderer::build_overlay(
            &snapshot,
            snapshot.roi,
            snapshot.frame_size,
        );

        assert!(overlay.roi_rect.is_some());
        assert_eq!(overlay.unit_annotations.len(), 2);
        assert_eq!(overlay.unit_annotations[0].side, Side::Left);
        assert_eq!(overlay.unit_annotations[1].side, Side::Right);
    }

    #[test]
    fn render_overlay_produces_annotated_image() {
        let snapshot = make_test_snapshot();
        let overlay = VisualizationRenderer::build_overlay(
            &snapshot,
            snapshot.roi,
            snapshot.frame_size,
        );

        let frame = RgbaImage::from_pixel(1280, 720, Rgba([0, 0, 0, 255]));
        let annotated = VisualizationRenderer::render_overlay(&frame, &overlay);

        assert_eq!(annotated.width(), 1280);
        assert_eq!(annotated.height(), 720);

        // ROI 矩形框的左上角应该是绿色
        let roi = overlay.roi_rect.unwrap();
        let pixel = annotated.get_pixel(roi.x, roi.y);
        assert_eq!(pixel, &Rgba([0, 255, 0, 255]));
    }

    #[test]
    fn unit_bbox_left_side_slot0() {
        let bbox = unit_bbox(Side::Left, 0, (1280, 720));
        assert_eq!(bbox.x, 0);
        assert!(bbox.width > 0);
        assert!(bbox.height > 0);
    }

    #[test]
    fn unit_bbox_right_side_slot0() {
        let bbox = unit_bbox(Side::Right, 0, (1280, 720));
        assert_eq!(bbox.x, 640); // 1280 / 2
    }
}
