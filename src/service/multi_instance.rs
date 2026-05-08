use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use crate::error::{AppError, AppResult};
use crate::service::fetch_coordinator::FetchInstance;
use crate::types::DeviceDescriptor;

pub struct MultiInstanceManager {
    instances: Mutex<HashMap<String, Arc<std::sync::Mutex<FetchInstance>>>>,
    occupied_devices: Mutex<HashSet<String>>,
    coordinator: crate::service::fetch_coordinator::FetchCoordinator,
}

impl MultiInstanceManager {
    pub fn new() -> Self {
        Self {
            instances: Mutex::new(HashMap::new()),
            occupied_devices: Mutex::new(HashSet::new()),
            coordinator: crate::service::fetch_coordinator::FetchCoordinator::new(),
        }
    }

    pub fn create_instance(&self, id: String, device: &DeviceDescriptor, instance: FetchInstance) -> AppResult<()> {
        let device_key = format!("{}:{}", device.name, device.address);
        {
            let occupied = self.occupied_devices.lock().unwrap();
            if occupied.contains(&device_key) {
                return Err(AppError::Fetch(format!("Device {} already occupied", device.name)));
            }
        }
        self.coordinator.start(id.clone(), instance);
        self.occupied_devices.lock().unwrap().insert(device_key);
        Ok(())
    }

    pub fn stop_instance(&self, id: &str) {
        self.coordinator.stop(id);
    }

    pub fn is_device_occupied(&self, device: &DeviceDescriptor) -> bool {
        let device_key = format!("{}:{}", device.name, device.address);
        self.occupied_devices.lock().unwrap().contains(&device_key)
    }

    pub fn active_count(&self) -> usize {
        self.occupied_devices.lock().unwrap().len()
    }
}
