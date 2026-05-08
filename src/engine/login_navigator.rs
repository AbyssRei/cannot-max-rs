use std::sync::Arc;

use crate::error::{AppError, AppResult};
use crate::engine::template_matcher::TemplateMatcher;
use crate::infra::maa_bridge::IMaaBridge;
use image::DynamicImage;

pub trait ILoginNavigator: Send + Sync {
    fn detect_and_handle_login(&self, image: &DynamicImage) -> AppResult<bool>;
    fn detect_and_close_announcement(&self, image: &DynamicImage) -> AppResult<bool>;
    fn navigate_to_target(&self, image: &DynamicImage) -> AppResult<()>;
    fn restart_game(&self) -> AppResult<()>;
}

pub struct LoginNavigator {
    maa: Arc<dyn IMaaBridge>,
    template_matcher: Arc<std::sync::Mutex<TemplateMatcher>>,
}

impl LoginNavigator {
    pub fn new(maa: Arc<dyn IMaaBridge>, template_matcher: TemplateMatcher) -> Self {
        Self {
            maa,
            template_matcher: Arc::new(std::sync::Mutex::new(template_matcher)),
        }
    }
}

impl ILoginNavigator for LoginNavigator {
    fn detect_and_handle_login(&self, _image: &DynamicImage) -> AppResult<bool> {
        let matcher = self.template_matcher.lock().unwrap();
        if let Some(_result) = matcher.match_template(_image, "login", 0.7) {
            self.maa.click(360, 640)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn detect_and_close_announcement(&self, _image: &DynamicImage) -> AppResult<bool> {
        let matcher = self.template_matcher.lock().unwrap();
        if let Some(_result) = matcher.match_template(_image, "close_announcement", 0.7) {
            self.maa.click(680, 40)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn navigate_to_target(&self, _image: &DynamicImage) -> AppResult<()> {
        self.maa.click(360, 400)?;
        Ok(())
    }

    fn restart_game(&self) -> AppResult<()> {
        tracing::warn!("Game restart requested - requires platform-specific implementation");
        Err(AppError::Login("Game restart not implemented".into()))
    }
}
