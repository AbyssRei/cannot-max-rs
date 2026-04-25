use crate::capture::{capture_frame, discover_sources};
use crate::config::AppConfig;
use crate::core::{AnalysisOutput, CaptureCatalog, GameMode, PredictionResult, SourceChoice};
use crate::prediction::{CandlePredictor, Predictor};
use crate::recognition::{analyze_frame, default_battle_roi};
use crate::resources::ResourceStore;
use iced::widget::{
    button, column, container, image, pick_list, row, scrollable, text, text_input,
};
use iced::{Alignment, Element, Length, Task, Theme};

pub fn run() -> iced::Result {
    iced::application(boot, update, view)
        .title("cannot-max-rs")
        .theme(app_theme)
        .run()
}

#[derive(Debug, Clone)]
pub enum Message {
    SourcesLoaded(CaptureCatalog),
    RefreshSources,
    SourceSelected(SourceChoice),
    CaptureAndPredict,
    AnalysisFinished(Result<AnalysisOutput, String>),
    ResourceRootChanged(String),
    ModelPathChanged(String),
    RoiXChanged(String),
    RoiYChanged(String),
    RoiWidthChanged(String),
    RoiHeightChanged(String),
    ModeChanged(GameMode),
    SaveConfig,
}

pub struct CannotMaxApp {
    config: AppConfig,
    catalog: CaptureCatalog,
    resources_summary: String,
    selected_source: Option<SourceChoice>,
    preview: Option<image::Handle>,
    recognized_rows: Vec<String>,
    prediction: Option<PredictionResult>,
    status: String,
    roi_x: String,
    roi_y: String,
    roi_width: String,
    roi_height: String,
    resource_root_text: String,
    model_path_text: String,
    busy: bool,
}

impl CannotMaxApp {
    fn new() -> Self {
        let config = AppConfig::load();
        let default_roi = config.roi.unwrap_or_default();
        let resources_summary = match ResourceStore::load(&config) {
            Ok(store) => store.summary(),
            Err(error) => format!("资源尚未就绪: {error}"),
        };

        Self {
            resource_root_text: config.resource_root.display().to_string(),
            model_path_text: config.model_path.display().to_string(),
            roi_x: default_roi.x.to_string(),
            roi_y: default_roi.y.to_string(),
            roi_width: default_roi.width.to_string(),
            roi_height: default_roi.height.to_string(),
            config,
            catalog: CaptureCatalog::default(),
            resources_summary,
            selected_source: None,
            preview: None,
            recognized_rows: Vec::new(),
            prediction: None,
            status: "正在加载输入源…".to_string(),
            busy: false,
        }
    }

    fn current_roi(&self) -> Option<crate::core::Roi> {
        let x = self.roi_x.parse::<u32>().ok()?;
        let y = self.roi_y.parse::<u32>().ok()?;
        let width = self.roi_width.parse::<u32>().ok()?;
        let height = self.roi_height.parse::<u32>().ok()?;
        Some(crate::core::Roi {
            x,
            y,
            width,
            height,
        })
    }

    fn save_config(&mut self) {
        self.config.resource_root = self.resource_root_text.clone().into();
        self.config.model_path = self.model_path_text.clone().into();
        self.config.roi = self.current_roi().filter(|roi| !roi.is_empty());
        self.status = match self.config.save() {
            Ok(()) => "配置已保存".to_string(),
            Err(error) => format!("配置保存失败: {error}"),
        };
    }
}

fn boot() -> (CannotMaxApp, Task<Message>) {
    (
        CannotMaxApp::new(),
        Task::perform(async { discover_sources() }, Message::SourcesLoaded),
    )
}

fn app_theme(_: &CannotMaxApp) -> Theme {
    Theme::TokyoNight
}

pub fn update(app: &mut CannotMaxApp, message: Message) -> Task<Message> {
    match message {
        Message::SourcesLoaded(catalog) => {
            app.catalog = catalog;
            let choices = app.catalog.source_choices();
            app.selected_source = app
                .config
                .last_capture_source
                .as_ref()
                .and_then(|source| {
                    choices
                        .iter()
                        .find(|choice| &choice.source == source)
                        .cloned()
                })
                .or_else(|| choices.first().cloned());
            app.status = format!(
                "输入源已刷新: ADB {} | 窗口 {} | 显示器 {}",
                app.catalog.adb_devices.len(),
                app.catalog.windows.len(),
                app.catalog.monitors.len()
            );
            Task::none()
        }
        Message::RefreshSources => {
            app.status = "正在刷新输入源…".to_string();
            Task::perform(async { discover_sources() }, Message::SourcesLoaded)
        }
        Message::SourceSelected(choice) => {
            app.config.last_capture_source = Some(choice.source.clone());
            app.selected_source = Some(choice);
            let _ = app.config.save();
            Task::none()
        }
        Message::CaptureAndPredict => {
            let Some(source) = app.selected_source.clone() else {
                app.status = "请先选择一个输入源".to_string();
                return Task::none();
            };

            app.busy = true;
            app.status = "正在截屏、识别并预测…".to_string();
            app.config.resource_root = app.resource_root_text.clone().into();
            app.config.model_path = app.model_path_text.clone().into();
            app.config.roi = app.current_roi().filter(|roi| !roi.is_empty());
            let _ = app.config.save();

            let config = app.config.clone();
            let catalog = app.catalog.clone();
            Task::perform(
                async move {
                    let resources = ResourceStore::load(&config)?;
                    let frame = capture_frame(&source.source, &config, &catalog)?;
                    let snapshot = analyze_frame(&source.source, &frame, config.roi, &resources);
                    let predictor = CandlePredictor::new(config.model_path.clone());
                    let prediction = predictor.predict(&snapshot)?;

                    Ok(AnalysisOutput {
                        frame,
                        snapshot,
                        prediction,
                    })
                },
                Message::AnalysisFinished,
            )
        }
        Message::AnalysisFinished(result) => {
            app.busy = false;

            match result {
                Ok(output) => {
                    let image = output.frame.image.clone();
                    app.preview = Some(image::Handle::from_rgba(
                        image.width(),
                        image.height(),
                        image.into_raw(),
                    ));

                    app.recognized_rows = if output.snapshot.units.is_empty() {
                        vec!["未识别到有效单位，当前为一期模板匹配基线。".to_string()]
                    } else {
                        output
                            .snapshot
                            .units
                            .iter()
                            .map(|unit| {
                                format!(
                                    "{} 槽位{} | ID {} | 数量 {} | 置信度 {:.2}",
                                    unit.side, unit.slot, unit.unit_id, unit.count, unit.confidence
                                )
                            })
                            .collect()
                    };
                    app.prediction = Some(output.prediction.clone());
                    app.status = format!(
                        "{} | ROI {:?}",
                        output.frame.note,
                        output
                            .snapshot
                            .roi
                            .unwrap_or_else(|| default_battle_roi(1920, 1080))
                    );
                }
                Err(error) => {
                    app.status = format!("执行失败: {error}");
                }
            }

            Task::none()
        }
        Message::ResourceRootChanged(value) => {
            app.resource_root_text = value;
            Task::none()
        }
        Message::ModelPathChanged(value) => {
            app.model_path_text = value;
            Task::none()
        }
        Message::RoiXChanged(value) => {
            app.roi_x = value;
            Task::none()
        }
        Message::RoiYChanged(value) => {
            app.roi_y = value;
            Task::none()
        }
        Message::RoiWidthChanged(value) => {
            app.roi_width = value;
            Task::none()
        }
        Message::RoiHeightChanged(value) => {
            app.roi_height = value;
            Task::none()
        }
        Message::ModeChanged(mode) => {
            app.config.game_mode = mode;
            let _ = app.config.save();
            Task::none()
        }
        Message::SaveConfig => {
            app.save_config();
            app.resources_summary = match ResourceStore::load(&app.config) {
                Ok(store) => store.summary(),
                Err(error) => format!("资源尚未就绪: {error}"),
            };
            Task::none()
        }
    }
}

pub fn view(app: &CannotMaxApp) -> Element<'_, Message> {
    let source_choices = app.catalog.source_choices();

    let header = column![
        text("cannot-max-rs").size(32),
        text("一期核心闭环: 双输入源 + RON 配置 + MAA/Win32 接入 + Candle 预测").size(16),
        text(&app.status).size(14),
    ]
    .spacing(8);

    let controls = column![
        row![
            button("刷新输入源").on_press(Message::RefreshSources),
            button(if app.busy {
                "处理中…"
            } else {
                "识别并预测"
            })
            .on_press_maybe((!app.busy).then_some(Message::CaptureAndPredict)),
            button("保存配置").on_press(Message::SaveConfig),
        ]
        .spacing(12),
        pick_list(
            source_choices,
            app.selected_source.clone(),
            Message::SourceSelected
        )
        .placeholder("选择 ADB / 窗口 / 显示器"),
        row![
            button("模拟器模式").on_press(Message::ModeChanged(GameMode::Emulator)),
            button("PC 模式").on_press(Message::ModeChanged(GameMode::Pc)),
            text(format!("当前模式: {}", app.config.game_mode)),
        ]
        .spacing(12)
        .align_y(Alignment::Center),
    ]
    .spacing(12);

    let path_panel = column![
        text("路径与资源").size(20),
        text_input("资源根目录", &app.resource_root_text).on_input(Message::ResourceRootChanged),
        text_input("模型路径", &app.model_path_text).on_input(Message::ModelPathChanged),
        text(&app.resources_summary).size(14),
    ]
    .spacing(8);

    let roi_panel = column![
        text("ROI").size(20),
        row![
            text_input("x", &app.roi_x)
                .on_input(Message::RoiXChanged)
                .width(Length::FillPortion(1)),
            text_input("y", &app.roi_y)
                .on_input(Message::RoiYChanged)
                .width(Length::FillPortion(1)),
            text_input("w", &app.roi_width)
                .on_input(Message::RoiWidthChanged)
                .width(Length::FillPortion(1)),
            text_input("h", &app.roi_height)
                .on_input(Message::RoiHeightChanged)
                .width(Length::FillPortion(1)),
        ]
        .spacing(8),
        text("首期支持手填 ROI 与自动恢复；窗口框选工具作为二期交互增强预留。").size(14),
    ]
    .spacing(8);

    let preview_panel: Element<'_, _> = if let Some(handle) = &app.preview {
        column![
            text("当前预览").size(20),
            image(handle.clone()).width(Length::Fill).height(320),
        ]
        .spacing(8)
        .into()
    } else {
        column![
            text("当前预览").size(20),
            container(text("尚未生成预览"))
                .width(Length::Fill)
                .height(320),
        ]
        .spacing(8)
        .into()
    };

    let prediction_text = if let Some(prediction) = &app.prediction {
        format!(
            "胜方: {} | 左 {:.1}% | 右 {:.1}% | 信心 {:.1}%",
            prediction.winner,
            prediction.left_win_rate * 100.0,
            prediction.right_win_rate * 100.0,
            prediction.confidence_band * 100.0
        )
    } else {
        "暂无预测结果".to_string()
    };

    let recognized = if app.recognized_rows.is_empty() {
        vec![text("暂无识别结果").into()]
    } else {
        app.recognized_rows
            .iter()
            .map(|row| text(row).into())
            .collect::<Vec<Element<'_, Message>>>()
    };

    let result_panel = column![
        text("识别与预测").size(20),
        text(prediction_text).size(16),
        scrollable(column(recognized).spacing(6)).height(220),
    ]
    .spacing(8);

    container(scrollable(
        column![
            header,
            controls,
            row![
                container(column![path_panel, roi_panel].spacing(16)).width(Length::FillPortion(1)),
                container(column![preview_panel, result_panel].spacing(16))
                    .width(Length::FillPortion(2)),
            ]
            .spacing(20),
        ]
        .spacing(20)
        .padding(20),
    ))
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}
