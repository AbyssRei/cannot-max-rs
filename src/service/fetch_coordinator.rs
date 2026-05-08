use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::AppResult;
use crate::engine::fetch_state_machine::{FetchStateMachine, IFetchStateMachine};
use crate::infra::maa_bridge::IMaaBridge;
use crate::engine::recognition::IRecognitionEngine;
use crate::engine::login_navigator::ILoginNavigator;
use crate::types::{FetchPhase, FetchRecord, FetchTransition, RoiRegion};
use image::DynamicImage;

pub struct FetchInstance {
    state_machine: FetchStateMachine,
    running: AtomicBool,
    phase: std::sync::Mutex<FetchPhase>,
    record: std::sync::Mutex<Option<FetchRecord>>,
}

impl FetchInstance {
    pub fn new(
        maa: Arc<dyn IMaaBridge>,
        recognition: Arc<dyn IRecognitionEngine>,
        login: Arc<dyn ILoginNavigator>,
        preset_roi: RoiRegion,
        custom_roi: Option<RoiRegion>,
    ) -> Self {
        Self {
            state_machine: FetchStateMachine::new(maa, recognition, login, preset_roi, custom_roi),
            running: AtomicBool::new(false),
            phase: std::sync::Mutex::new(FetchPhase::MainMenu),
            record: std::sync::Mutex::new(None),
        }
    }

    pub fn tick(&mut self, image: &DynamicImage) -> AppResult<FetchTransition> {
        let transition = self.state_machine.tick(image)?;
        *self.phase.lock().unwrap() = self.state_machine.current_phase();
        if let Some(record) = self.state_machine.get_record() {
            *self.record.lock().unwrap() = Some(record.clone());
        }
        Ok(transition)
    }

    pub fn stop(&self) {
        self.state_machine.request_stop();
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn current_phase(&self) -> FetchPhase {
        self.phase.lock().unwrap().clone()
    }

    pub fn get_record(&self) -> Option<FetchRecord> {
        self.record.lock().unwrap().clone()
    }
}

pub struct FetchCoordinator {
    instances: std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<FetchInstance>>>>,
}

impl FetchCoordinator {
    pub fn new() -> Self {
        Self {
            instances: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub fn start(&self, instance_id: String, instance: FetchInstance) {
        instance.running.store(true, Ordering::SeqCst);
        let mut instances = self.instances.lock().unwrap();
        instances.insert(instance_id, Arc::new(std::sync::Mutex::new(instance)));
    }

    pub fn stop(&self, instance_id: &str) {
        let instances = self.instances.lock().unwrap();
        if let Some(instance) = instances.get(instance_id) {
            instance.lock().unwrap().stop();
        }
    }

    pub fn get_status(&self, instance_id: &str) -> Option<FetchPhase> {
        let instances = self.instances.lock().unwrap();
        instances.get(instance_id).map(|i| i.lock().unwrap().current_phase())
    }

    pub fn remove_stopped(&self) {
        let mut instances = self.instances.lock().unwrap();
        instances.retain(|_, instance| instance.lock().unwrap().is_running());
    }
}
