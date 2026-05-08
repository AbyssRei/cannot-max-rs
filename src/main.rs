#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use cannot_max_rs::types;
use cannot_max_rs::app_controller;
use cannot_max_rs::bridge::slint_convert;

slint::include_modules!();

fn create_placeholder_image(r: u8, g: u8, b: u8) -> slint::Image {
    let width = 80u32;
    let height = 80u32;
    let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
    for pixel in buffer.make_mut_slice().iter_mut() {
        *pixel = slint::Rgba8Pixel { r, g, b, a: 255 };
    }
    slint::Image::from_rgba8(buffer)
}

fn recognized_to_item(monster: &types::RecognizedMonster) -> RecognitionItem {
    RecognitionItem {
        image: create_placeholder_image(100, 100, 100),
        count: monster.count,
        label: monster.name.clone().into(),
        low_confidence: monster.low_confidence,
        count_uncertain: monster.count_uncertain,
        ocr_unavailable: monster.ocr_unavailable,
    }
}

fn prediction_to_item(result: &types::PredictionResult) -> PredictedItem {
    PredictedItem {
        model_serial: result.model_serial as i32,
        confidence: result.confidence as f32,
        terrain_included: result.terrain_included,
        load_failed: result.load_failed,
        inference_timeout: result.inference_timeout,
    }
}

fn apply_recognition_result(ui: &AppWindow, result: &types::RecognitionResult) {
    let left: Vec<_> = result.left.iter().map(recognized_to_item).collect();
    let right: Vec<_> = result.right.iter().map(recognized_to_item).collect();
    ui.set_left_items(slint::ModelRc::new(slint::VecModel::from(left)));
    ui.set_right_items(slint::ModelRc::new(slint::VecModel::from(right)));
}

fn apply_prediction_results(ui: &AppWindow, results: &[types::PredictionResult]) {
    let items: Vec<_> = results.iter().map(prediction_to_item).collect();
    ui.set_predicted_items(slint::ModelRc::new(slint::VecModel::from(items)));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();
    tracing::info!("CannotMax-RS starting");

    let resource_root = std::path::PathBuf::from("resource");
    let controller = match app_controller::AppController::new(&resource_root) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Failed to initialize: {}", e);
            std::process::exit(1);
        }
    };

    let ui = AppWindow::new()?;

    let device_list: Arc<std::sync::Mutex<Vec<types::DeviceDescriptor>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    ui.on_request_recognize({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        move || {
            let ui = ui_handle.unwrap();
            ui.set_app_state(AppState::Recognizing);
            let ctrl = ctrl.clone();
            let ui_handle2 = ui.as_weak();
            std::thread::spawn(move || {
                match ctrl.handle_recognize() {
                    Ok(result) => {
                        slint::invoke_from_event_loop(move || {
                            let ui = ui_handle2.unwrap();
                            ui.set_app_state(AppState::RecognizeSuccess);
                            apply_recognition_result(&ui, &result);
                        }).ok();
                    }
                    Err(_) => {
                        slint::invoke_from_event_loop(move || {
                            let ui = ui_handle2.unwrap();
                            ui.set_app_state(AppState::RecognizeFailed);
                        }).ok();
                    }
                }
            });
        }
    });

    ui.on_request_predict({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        move || {
            let ui = ui_handle.unwrap();
            ui.set_app_state(AppState::Predicting);
            let ctrl = ctrl.clone();
            let ui_handle2 = ui.as_weak();
            std::thread::spawn(move || {
                match ctrl.handle_predict() {
                    Ok(results) => {
                        slint::invoke_from_event_loop(move || {
                            let ui = ui_handle2.unwrap();
                            ui.set_app_state(AppState::PredictSuccess);
                            apply_prediction_results(&ui, &results);
                        }).ok();
                    }
                    Err(_) => {
                        slint::invoke_from_event_loop(move || {
                            let ui = ui_handle2.unwrap();
                            ui.set_app_state(AppState::PredictFailed);
                        }).ok();
                    }
                }
            });
        }
    });

    ui.on_request_recognize_and_predict({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        move || {
            let ui = ui_handle.unwrap();
            ui.set_app_state(AppState::Recognizing);
            let ctrl = ctrl.clone();
            let ui_handle2 = ui.as_weak();
            std::thread::spawn(move || {
                match ctrl.handle_recognize() {
                    Ok(rec_result) => {
                        match ctrl.handle_predict() {
                            Ok(pred_results) => {
                                slint::invoke_from_event_loop(move || {
                                    let ui = ui_handle2.unwrap();
                                    ui.set_app_state(AppState::RecognizeAndPredictSuccess);
                                    apply_recognition_result(&ui, &rec_result);
                                    apply_prediction_results(&ui, &pred_results);
                                }).ok();
                            }
                            Err(_) => {
                                slint::invoke_from_event_loop(move || {
                                    let ui = ui_handle2.unwrap();
                                    ui.set_app_state(AppState::PredictFailed);
                                    apply_recognition_result(&ui, &rec_result);
                                }).ok();
                            }
                        }
                    }
                    Err(_) => {
                        slint::invoke_from_event_loop(move || {
                            let ui = ui_handle2.unwrap();
                            ui.set_app_state(AppState::RecognizeFailed);
                        }).ok();
                    }
                }
            });
        }
    });

    ui.on_request_reset({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        move || {
            let ui = ui_handle.unwrap();
            ctrl.handle_reset();
            ui.set_app_state(AppState::Idle);
            ui.set_left_items(slint::ModelRc::new(slint::VecModel::default()));
            ui.set_right_items(slint::ModelRc::new(slint::VecModel::default()));
            ui.set_predicted_items(slint::ModelRc::new(slint::VecModel::default()));
        }
    });

    ui.on_request_change_data_source({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        let device_list = device_list.clone();
        move |source: DataSource| {
            let ui = ui_handle.unwrap();
            ui.set_data_source(source);
            let current_source = ui.get_data_source();
            let mode = match current_source {
                DataSource::Adb => types::CaptureMode::Adb,
                DataSource::Pc => types::CaptureMode::Pc,
                DataSource::WindowCapture => types::CaptureMode::WinCapture,
                _ => types::CaptureMode::Adb,
            };
            let ctrl = ctrl.clone();
            let ui_handle2 = ui.as_weak();
            let device_list = device_list.clone();
            std::thread::spawn(move || {
                let devices = ctrl.handle_refresh_devices(mode).unwrap_or_default();
                let names: Vec<slint::SharedString> = if devices.is_empty() {
                    vec!["未发现设备".into()]
                } else {
                    devices.iter().map(|d| d.name.clone().into()).collect()
                };
                *device_list.lock().unwrap() = devices;
                slint::invoke_from_event_loop(move || {
                    let ui = ui_handle2.unwrap();
                    ui.set_device_list(slint::ModelRc::new(slint::VecModel::from(names)));
                    ui.set_selected_device_index(0);
                }).ok();
            });
        }
    });

    ui.on_request_refresh_devices({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        let device_list = device_list.clone();
        move || {
            let ui = ui_handle.unwrap();
            let current_source = ui.get_data_source();
            let mode = match current_source {
                DataSource::Adb => types::CaptureMode::Adb,
                DataSource::Pc => types::CaptureMode::Pc,
                DataSource::WindowCapture => types::CaptureMode::WinCapture,
                _ => types::CaptureMode::Adb,
            };
            let ctrl = ctrl.clone();
            let ui_handle2 = ui.as_weak();
            let device_list = device_list.clone();
            std::thread::spawn(move || {
                let devices = ctrl.handle_refresh_devices(mode).unwrap_or_default();
                let names: Vec<slint::SharedString> = if devices.is_empty() {
                    vec!["未发现设备".into()]
                } else {
                    devices.iter().map(|d| d.name.clone().into()).collect()
                };
                *device_list.lock().unwrap() = devices;
                slint::invoke_from_event_loop(move || {
                    let ui = ui_handle2.unwrap();
                    ui.set_device_list(slint::ModelRc::new(slint::VecModel::from(names)));
                }).ok();
            });
        }
    });

    ui.on_request_connect_device({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        let device_list = device_list.clone();
        move || {
            let index;
            let screencap_idx;
            let mouse_idx;
            let keyboard_idx;
            {
                let ui = ui_handle.unwrap();
                index = ui.get_selected_device_index() as usize;
                screencap_idx = ui.get_screencap_method_index();
                mouse_idx = ui.get_mouse_method_index();
                keyboard_idx = ui.get_keyboard_method_index();
            }

            use maa_framework::common::{Win32ScreencapMethod, Win32InputMethod};

            let screencap_method = match screencap_idx {
                0 => Win32ScreencapMethod::FRAME_POOL.bits(),
                1 => Win32ScreencapMethod::PRINT_WINDOW.bits(),
                2 => Win32ScreencapMethod::SCREEN_DC.bits(),
                _ => Win32ScreencapMethod::DXGI_DESKTOP_DUP_WINDOW.bits(),
            };
            let mouse_method = match mouse_idx {
                0 => Win32InputMethod::SEIZE.bits(),
                1 => Win32InputMethod::SEND_MESSAGE_WITH_CURSOR_POS.bits(),
                _ => Win32InputMethod::SEND_MESSAGE_WITH_WINDOW_POS.bits(),
            };
            let keyboard_method = match keyboard_idx {
                0 => Win32InputMethod::SEIZE.bits(),
                1 => Win32InputMethod::SEND_MESSAGE.bits(),
                _ => Win32InputMethod::POST_MESSAGE.bits(),
            };

            let list = device_list.lock().unwrap();
            if let Some(device) = list.get(index) {
                let device = device.clone();
                let device_name = device.name.clone();
                drop(list);
                let ctrl = ctrl.clone();
                let ui_handle2 = ui_handle.clone();
                std::thread::spawn(move || {
                    match ctrl.handle_connect_device(device, screencap_method, mouse_method, keyboard_method) {
                        Ok(()) => {
                            tracing::info!("Device connected successfully");
                            slint::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_handle2.upgrade() {
                                    ui.set_connected_device_name(device_name.into());
                                }
                            }).ok();
                        }
                        Err(e) => {
                            tracing::error!("Failed to connect device: {}", e);
                        }
                    }
                });
            } else {
                drop(list);
                tracing::warn!("Selected device index {} out of range", index);
            }
        }
    });

    ui.on_request_select_roi({
        let ui_handle = ui.as_weak();
        let ctrl = controller.clone();
        move || {
            let eg_path = std::path::PathBuf::from("resource/images/eg.png");
            let eg_image = slint_convert::image_from_path(&eg_path)
                .unwrap_or_else(|| {
                    tracing::warn!("Failed to load eg.png from {:?}", eg_path);
                    create_placeholder_image(128, 128, 128)
                });

            let (capture_image, capture_error, capture_w, capture_h) = match ctrl.device_manager().capture_screenshot() {
                Ok(img) => {
                    let rgba = img.to_rgba8();
                    let w = rgba.width();
                    let h = rgba.height();
                    tracing::info!("Captured screenshot for ROI selection ({}x{})", w, h);
                    (slint_convert::image_from_rgba8(w, h, rgba.as_raw()), slint::SharedString::default(), w as i32, h as i32)
                }
                Err(e) => {
                    let err_msg = format!("{}", e);
                    tracing::warn!("Failed to capture screenshot for ROI selection: {}", err_msg);
                    (slint::Image::default(), slint::SharedString::from(err_msg.as_str()), 0, 0)
                }
            };

            tracing::info!("ROI selection window requested");
            let ui = ui_handle.unwrap();
            ui.invoke_show_roi_selection_window(eg_image, capture_image, capture_error, capture_w, capture_h);
        }
    });

    ui.on_roi_confirmed({
        let ctrl = controller.clone();
        move |x: i32, y: i32, w: i32, h: i32| {
            let roi = types::RoiRegion {
                x: x as u32,
                y: y as u32,
                width: w as u32,
                height: h as u32,
            };
            ctrl.update_custom_roi(roi);
            tracing::info!("Custom ROI updated: ({}, {}) {}x{}", x, y, w, h);
        }
    });

    ui.run()?;
    Ok(())
}
