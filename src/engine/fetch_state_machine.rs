use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::error::AppResult;
use crate::engine::login_navigator::ILoginNavigator;
use crate::engine::recognition::IRecognitionEngine;
use crate::infra::maa_bridge::IMaaBridge;
use crate::types::{FetchPhase, FetchRecord, FetchTransition, Winner};
use image::DynamicImage;

pub trait IFetchStateMachine: Send + Sync {
    fn tick(&mut self, image: &DynamicImage) -> AppResult<FetchTransition>;
    fn current_phase(&self) -> FetchPhase;
    fn request_stop(&self);
    fn get_record(&self) -> Option<&FetchRecord>;
}

pub struct FetchStateMachine {
    current_phase: FetchPhase,
    maa: Arc<dyn IMaaBridge>,
    recognition: Arc<dyn IRecognitionEngine>,
    login: Arc<dyn ILoginNavigator>,
    stop_requested: AtomicBool,
    record: Option<FetchRecord>,
    retry_count: u32,
    max_retries: u32,
    battle_timeout_secs: u64,
    battle_elapsed_secs: u64,
    preset_roi: crate::types::RoiRegion,
    custom_roi: Option<crate::types::RoiRegion>,
}

impl FetchStateMachine {
    pub fn new(
        maa: Arc<dyn IMaaBridge>,
        recognition: Arc<dyn IRecognitionEngine>,
        login: Arc<dyn ILoginNavigator>,
        preset_roi: crate::types::RoiRegion,
        custom_roi: Option<crate::types::RoiRegion>,
    ) -> Self {
        Self {
            current_phase: FetchPhase::MainMenu,
            maa,
            recognition,
            login,
            stop_requested: AtomicBool::new(false),
            record: None,
            retry_count: 0,
            max_retries: 3,
            battle_timeout_secs: 300,
            battle_elapsed_secs: 0,
            preset_roi,
            custom_roi,
        }
    }

    fn handle_main_menu(&mut self, _image: &DynamicImage) -> AppResult<FetchTransition> {
        self.login.navigate_to_target(_image)?;
        self.current_phase = FetchPhase::ModeSelect;
        self.retry_count = 0;
        Ok(FetchTransition::Continue)
    }

    fn handle_mode_select(&mut self, _image: &DynamicImage) -> AppResult<FetchTransition> {
        self.maa.click(360, 400)?;
        self.current_phase = FetchPhase::PreBattle;
        self.retry_count = 0;
        Ok(FetchTransition::Continue)
    }

    fn handle_pre_battle(&mut self, image: &DynamicImage) -> AppResult<FetchTransition> {
        let mode = crate::types::CaptureMode::Adb;
        let roi = crate::engine::roi_strategy::RoiStrategy::resolve_roi(
            &mode, &self.preset_roi, self.custom_roi.as_ref()
        ).unwrap_or_default();
        let result = self.recognition.recognize(image, &roi)?;
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        self.record = Some(FetchRecord {
            timestamp,
            left_monsters: result.left.iter().map(|m| (m.name.clone(), m.count)).collect(),
            right_monsters: result.right.iter().map(|m| (m.name.clone(), m.count)).collect(),
            left_terrain: result.left_terrain.clone(),
            right_terrain: result.right_terrain.clone(),
            winner: Winner::Unknown,
            screenshot_path: None,
        });
        self.current_phase = FetchPhase::InBattle;
        self.battle_elapsed_secs = 0;
        self.retry_count = 0;
        Ok(FetchTransition::Continue)
    }

    fn handle_in_battle(&mut self, _image: &DynamicImage) -> AppResult<FetchTransition> {
        self.battle_elapsed_secs += 1;
        if self.battle_elapsed_secs >= self.battle_timeout_secs {
            self.current_phase = FetchPhase::MainMenu;
            self.retry_count = 0;
            return Ok(FetchTransition::Timeout);
        }
        Ok(FetchTransition::Continue)
    }

    fn handle_settlement(&mut self, _image: &DynamicImage) -> AppResult<FetchTransition> {
        if let Some(ref mut record) = self.record {
            record.winner = Winner::Left;
        }
        self.current_phase = FetchPhase::Completed;
        Ok(FetchTransition::Continue)
    }

    fn handle_failure(&mut self) -> FetchTransition {
        self.retry_count += 1;
        if self.retry_count >= self.max_retries {
            self.current_phase = FetchPhase::MainMenu;
            self.retry_count = 0;
            FetchTransition::BackToMenu
        } else {
            FetchTransition::Retry
        }
    }
}

impl IFetchStateMachine for FetchStateMachine {
    fn tick(&mut self, image: &DynamicImage) -> AppResult<FetchTransition> {
        if self.stop_requested.load(Ordering::SeqCst) {
            return Ok(FetchTransition::Stop);
        }
        let result = match self.current_phase {
            FetchPhase::MainMenu => self.handle_main_menu(image),
            FetchPhase::ModeSelect => self.handle_mode_select(image),
            FetchPhase::PreBattle => self.handle_pre_battle(image),
            FetchPhase::InBattle => self.handle_in_battle(image),
            FetchPhase::Settlement => self.handle_settlement(image),
            FetchPhase::Completed => {
                self.current_phase = FetchPhase::MainMenu;
                return Ok(FetchTransition::Continue);
            }
        };
        match result {
            Ok(transition) => Ok(transition),
            Err(_) => Ok(self.handle_failure()),
        }
    }

    fn current_phase(&self) -> FetchPhase {
        self.current_phase.clone()
    }

    fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::SeqCst);
    }

    fn get_record(&self) -> Option<&FetchRecord> {
        self.record.as_ref()
    }
}
