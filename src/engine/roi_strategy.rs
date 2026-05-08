use crate::error::{AppError, AppResult};
use crate::types::{CaptureMode, RoiRegion};

pub struct RoiStrategy;

impl RoiStrategy {
    pub fn new() -> Self {
        Self
    }

    pub fn resolve_roi(
        mode: &CaptureMode,
        preset_roi: &RoiRegion,
        custom_roi: Option<&RoiRegion>,
    ) -> AppResult<RoiRegion> {
        match mode {
            CaptureMode::Adb | CaptureMode::Pc => {
                tracing::debug!("Using preset ROI: {:?}", preset_roi);
                Ok(preset_roi.clone())
            }
            CaptureMode::WinCapture => {
                match custom_roi {
                    Some(roi) if roi.is_valid() => {
                        tracing::debug!("Using custom ROI: {:?}", roi);
                        Ok(roi.clone())
                    }
                    Some(roi) => {
                        tracing::warn!("Custom ROI is invalid (area={}), falling back to preset", roi.area());
                        Ok(preset_roi.clone())
                    }
                    None => {
                        tracing::warn!("Custom ROI not set, falling back to preset");
                        Ok(preset_roi.clone())
                    }
                }
            }
        }
    }

    pub fn validate_roi(roi: &RoiRegion, image_width: u32, image_height: u32) -> AppResult<RoiRegion> {
        if !roi.is_valid() {
            return Err(AppError::RoiSelectionZeroArea);
        }
        Ok(roi.clip_to_bounds(image_width, image_height))
    }
}
