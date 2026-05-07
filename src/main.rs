#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::error::Error;
use std::rc::Rc;
use std::cell::RefCell;

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

fn main() -> Result<(), Box<dyn Error>> {
    let ui = AppWindow::new()?;

    let img_a = create_placeholder_image(66, 133, 244);
    let img_b = create_placeholder_image(234, 67, 53);
    let img_c = create_placeholder_image(251, 188, 4);
    let img_d = create_placeholder_image(52, 168, 83);
    let img_e = create_placeholder_image(103, 58, 183);
    let img_f = create_placeholder_image(255, 112, 67);

    let demo_left_items = Rc::new(RefCell::new(None::<slint::ModelRc<RecognitionItem>>));
    let demo_right_items = Rc::new(RefCell::new(None::<slint::ModelRc<RecognitionItem>>));
    let demo_predicted_items = Rc::new(RefCell::new(None::<slint::ModelRc<PredictedItem>>));

    {
        let left = slint::ModelRc::new(slint::VecModel::from(vec![
            RecognitionItem { image: img_a, count: 5, label: "类别A".into() },
            RecognitionItem { image: img_b, count: 3, label: "类别B".into() },
            RecognitionItem { image: img_c, count: 8, label: "类别C".into() },
        ]));
        let right = slint::ModelRc::new(slint::VecModel::from(vec![
            RecognitionItem { image: img_d, count: 2, label: "类别D".into() },
            RecognitionItem { image: img_e, count: 7, label: "类别E".into() },
            RecognitionItem { image: img_f, count: 4, label: "类别F".into() },
        ]));
        *demo_left_items.borrow_mut() = Some(left);
        *demo_right_items.borrow_mut() = Some(right);
        let predicted = slint::ModelRc::new(slint::VecModel::from(vec![
            PredictedItem { model_serial: 1, confidence: 0.85 },
            PredictedItem { model_serial: 2, confidence: 0.65 },
            PredictedItem { model_serial: 3, confidence: 0.45 },
        ]));
        *demo_predicted_items.borrow_mut() = Some(predicted);
    }

    let timer = Rc::new(slint::Timer::default());


    ui.on_request_predict({
        let ui_handle = ui.as_weak();
        let predicted_rc = demo_predicted_items.clone();
        let timer_rc = timer.clone();
        move || {
            let ui = ui_handle.unwrap();
            ui.set_app_state(AppState::Predicting);
            let ui_handle2 = ui.as_weak();
            let predicted_rc2 = predicted_rc.clone();
            timer_rc.start(slint::TimerMode::SingleShot, std::time::Duration::from_secs(2), move || {
                let ui = ui_handle2.unwrap();
                ui.set_app_state(AppState::PredictSuccess);
                if let Some(ref predicted) = *predicted_rc2.borrow() {
                    ui.set_predicted_items(predicted.clone());
                }
            });
        }
    });

    ui.on_request_recognize_and_predict({
        let ui_handle = ui.as_weak();
        let left_rc = demo_left_items.clone();
        let right_rc = demo_right_items.clone();
        let predicted_rc = demo_predicted_items.clone();
        let timer_rc = timer.clone();
        move || {
            let ui = ui_handle.unwrap();
            ui.set_app_state(AppState::Recognizing);
            let ui_handle2 = ui.as_weak();
            let left_rc2 = left_rc.clone();
            let right_rc2 = right_rc.clone();
            let predicted_rc2 = predicted_rc.clone();
            timer_rc.start(slint::TimerMode::SingleShot, std::time::Duration::from_secs(2), move || {
                let ui = ui_handle2.unwrap();
                ui.set_app_state(AppState::RecognizeAndPredictSuccess);
                if let Some(ref left) = *left_rc2.borrow() {
                    ui.set_left_items(left.clone());
                }
                if let Some(ref right) = *right_rc2.borrow() {
                    ui.set_right_items(right.clone());
                }
                if let Some(ref predicted) = *predicted_rc2.borrow() {
                    ui.set_predicted_items(predicted.clone());
                }
            });
        }
    });

    ui.on_request_recognize({
        let ui_handle = ui.as_weak();
        let left_rc = demo_left_items.clone();
        let right_rc = demo_right_items.clone();
        let timer_rc = timer.clone();
        move || {
            let ui = ui_handle.unwrap();
            ui.set_app_state(AppState::Recognizing);
            let ui_handle2 = ui.as_weak();
            let left_rc2 = left_rc.clone();
            let right_rc2 = right_rc.clone();
            timer_rc.start(slint::TimerMode::SingleShot, std::time::Duration::from_secs(2), move || {
                let ui = ui_handle2.unwrap();
                ui.set_app_state(AppState::RecognizeSuccess);
                if let Some(ref left) = *left_rc2.borrow() {
                    ui.set_left_items(left.clone());
                }
                if let Some(ref right) = *right_rc2.borrow() {
                    ui.set_right_items(right.clone());
                }
            });
        }
    });

    ui.on_request_reset({
        let ui_handle = ui.as_weak();
        move || {
            let ui = ui_handle.unwrap();
            ui.set_app_state(AppState::Idle);
            ui.set_left_items(slint::ModelRc::new(slint::VecModel::default()));
            ui.set_right_items(slint::ModelRc::new(slint::VecModel::default()));
        }
    });

    ui.run()?;

    Ok(())
}
