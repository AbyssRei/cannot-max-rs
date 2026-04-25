use crate::automation::{AutomationAction, execute_action};
use crate::capture::discover_sources;
use crate::config::{AppConfig, Win32InputMethodConfig};
use crate::core::{AnalysisOutput, CaptureCatalog, GameMode, PredictionResult, SourceChoice};
use crate::ocr::{DeepseekCliModel, OcrBackend, library_hint, ocr_hint};
use crate::pipeline::AnalysisPipeline;
use crate::recognition::default_battle_roi;
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
    MaaLibraryPathChanged(String),
    OcrModelPathChanged(String),
    OcrBackendSelected(OcrBackend),
    DeepseekCliPathChanged(String),
    DeepseekModelSelected(DeepseekCliModel),
    DeepseekDeviceChanged(String),
    RoiXChanged(String),
    RoiYChanged(String),
    RoiWidthChanged(String),
    RoiHeightChanged(String),
    ModeChanged(GameMode),
    ActionXChanged(String),
    ActionYChanged(String),
    ActionTextChanged(String),
    SendClick,
    SendInputText,
    SendInactive,
    AutomationFinished(Result<String, String>),
    Win32InputMethodSelected(Win32InputMethodConfig),
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
    maa_library_path_text: String,
    ocr_model_path_text: String,
    deepseek_cli_path_text: String,
    deepseek_device_text: String,
    action_x_text: String,
    action_y_text: String,
    action_text: String,
    busy: bool,
}

impl CannotMaxApp {
    fn new() -> Self {
        let config = AppConfig::load();
        let default_roi = config.roi.unwrap_or_default();
        let resources_summary = summarize_runtime(&config);

        Self {
            resource_root_text: config.resource_root.display().to_string(),
            model_path_text: config.model_path.display().to_string(),
            maa_library_path_text: config.maa_library_path.display().to_string(),
            ocr_model_path_text: config.ocr_model_path.display().to_string(),
            deepseek_cli_path_text: config.deepseek_cli_path.display().to_string(),
            deepseek_device_text: config.deepseek_device.clone(),
            action_x_text: "960".to_string(),
            action_y_text: "540".to_string(),
            action_text: String::new(),
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
        self.config.maa_library_path = self.maa_library_path_text.clone().into();
        self.config.ocr_model_path = self.ocr_model_path_text.clone().into();
        self.config.deepseek_cli_path = self.deepseek_cli_path_text.clone().into();
        self.config.deepseek_device = self.deepseek_device_text.clone();
        self.config.roi = self.current_roi().filter(|roi| !roi.is_empty());
        self.status = match self.config.save() {
            Ok(()) => "配置已保存".to_string(),
            Err(error) => format!("配置保存失败: {error}"),
        };
    }
}

fn boot() -> (CannotMaxApp, Task<Message>) {
    let app = CannotMaxApp::new();
    let mode = app.config.game_mode;
    (
        app,
        Task::perform(async move { discover_sources(mode) }, Message::SourcesLoaded),
    )
}

fn app_theme(_: &CannotMaxApp) -> Theme {
    Theme::TokyoNight
}

fn summarize_runtime(config: &AppConfig) -> String {
    let resource_summary = match ResourceStore::load(config) {
        Ok(store) => store.summary(),
        Err(error) => format!("资源尚未就绪: {error}"),
    };

    format!(
        "{resource_summary}\n{}\n{}",
        library_hint(config),
        ocr_hint(config)
    )
}

pub fn update(app: &mut CannotMaxApp, message: Message) -> Task<Message> {
    match message {
        Message::SourcesLoaded(catalog) => {
            app.catalog = catalog;
            let choices = app.catalog.source_choices(app.config.game_mode);
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
            let mode = app.config.game_mode;
            Task::perform(async move { discover_sources(mode) }, Message::SourcesLoaded)
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
            app.config.maa_library_path = app.maa_library_path_text.clone().into();
            app.config.ocr_model_path = app.ocr_model_path_text.clone().into();
            app.config.deepseek_cli_path = app.deepseek_cli_path_text.clone().into();
            app.config.deepseek_device = app.deepseek_device_text.clone();
            app.config.roi = app.current_roi().filter(|roi| !roi.is_empty());
            let _ = app.config.save();

            let config = app.config.clone();
            let catalog = app.catalog.clone();
            Task::perform(
                async move {
                    AnalysisPipeline::new(config.model_path.clone())
                        .run(&source.source, &catalog, &config)
                },
                Message::AnalysisFinished,
            )
        }
        Message::ActionXChanged(value) => {
            app.action_x_text = value;
            Task::none()
        }
        Message::ActionYChanged(value) => {
            app.action_y_text = value;
            Task::none()
        }
        Message::ActionTextChanged(value) => {
            app.action_text = value;
            Task::none()
        }
        Message::SendClick => {
            let Some(source) = app.selected_source.clone() else {
                app.status = "请先选择一个输入源".to_string();
                return Task::none();
            };

            let Ok(x) = app.action_x_text.trim().parse::<i32>() else {
                app.status = "点击坐标 x 无效，请输入整数".to_string();
                return Task::none();
            };
            let Ok(y) = app.action_y_text.trim().parse::<i32>() else {
                app.status = "点击坐标 y 无效，请输入整数".to_string();
                return Task::none();
            };

            app.busy = true;
            app.status = "正在发送自动点击…".to_string();
            let config = app.config.clone();
            let catalog = app.catalog.clone();

            Task::perform(
                async move {
                    execute_action(
                        &source.source,
                        &catalog,
                        &config,
                        AutomationAction::Click { x, y },
                    )
                },
                Message::AutomationFinished,
            )
        }
        Message::SendInputText => {
            let Some(source) = app.selected_source.clone() else {
                app.status = "请先选择一个输入源".to_string();
                return Task::none();
            };

            if app.action_text.trim().is_empty() {
                app.status = "请输入要发送的文本".to_string();
                return Task::none();
            }

            app.busy = true;
            app.status = "正在发送文本输入…".to_string();
            let config = app.config.clone();
            let catalog = app.catalog.clone();
            let text = app.action_text.clone();

            Task::perform(
                async move {
                    execute_action(
                        &source.source,
                        &catalog,
                        &config,
                        AutomationAction::InputText { text },
                    )
                },
                Message::AutomationFinished,
            )
        }
        Message::SendInactive => {
            let Some(source) = app.selected_source.clone() else {
                app.status = "请先选择一个输入源".to_string();
                return Task::none();
            };

            app.busy = true;
            app.status = "正在发送 inactive 请求…".to_string();
            let config = app.config.clone();
            let catalog = app.catalog.clone();

            Task::perform(
                async move {
                    execute_action(&source.source, &catalog, &config, AutomationAction::Inactive)
                },
                Message::AutomationFinished,
            )
        }
        Message::Win32InputMethodSelected(method) => {
            app.config.win32_input_method = method;
            let _ = app.config.save();
            app.status = format!("Win32 输入方法已切换为: {method}");
            Task::none()
        }
        Message::AutomationFinished(result) => {
            app.busy = false;
            app.status = match result {
                Ok(message) => message,
                Err(error) => format!("自动化执行失败: {error}"),
            };
            Task::none()
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
                                    "{} 槽位{} | ID {} | 数量 {} | 置信度 {:.2} | 数量来源 {}{}",
                                    unit.side,
                                    unit.slot,
                                    unit.unit_id,
                                    unit.count,
                                    unit.confidence,
                                    unit.count_source,
                                    if unit.count_cached { " | 缓存" } else { "" }
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
        Message::MaaLibraryPathChanged(value) => {
            app.maa_library_path_text = value;
            Task::none()
        }
        Message::OcrModelPathChanged(value) => {
            app.ocr_model_path_text = value;
            Task::none()
        }
        Message::OcrBackendSelected(value) => {
            app.config.ocr_backend = value;
            let _ = app.config.save();
            app.resources_summary = summarize_runtime(&app.config);
            Task::none()
        }
        Message::DeepseekCliPathChanged(value) => {
            app.deepseek_cli_path_text = value;
            Task::none()
        }
        Message::DeepseekModelSelected(value) => {
            app.config.deepseek_model = value;
            let _ = app.config.save();
            app.resources_summary = summarize_runtime(&app.config);
            Task::none()
        }
        Message::DeepseekDeviceChanged(value) => {
            app.deepseek_device_text = value;
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
            app.status = format!("当前模式已切换为 {mode}，正在刷新输入源…");
            Task::perform(async move { discover_sources(mode) }, Message::SourcesLoaded)
        }
        Message::SaveConfig => {
            app.save_config();
            app.resources_summary = summarize_runtime(&app.config);
            Task::none()
        }
    }
}

pub fn view(app: &CannotMaxApp) -> Element<'_, Message> {
    let source_choices = app.catalog.source_choices(app.config.game_mode);

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

    let automation_panel = column![
        text("自动化操作（MAA）").size(20),
        pick_list(
            Win32InputMethodConfig::ALL.to_vec(),
            Some(app.config.win32_input_method),
            Message::Win32InputMethodSelected
        )
        .placeholder("选择 Win32 输入方法（仅窗口源）"),
        row![
            text_input("x", &app.action_x_text)
                .on_input(Message::ActionXChanged)
                .width(Length::FillPortion(1)),
            text_input("y", &app.action_y_text)
                .on_input(Message::ActionYChanged)
                .width(Length::FillPortion(1)),
            button("发送点击")
                .on_press_maybe((!app.busy).then_some(Message::SendClick)),
        ]
        .spacing(8),
        row![
            text_input("输入文本", &app.action_text)
                .on_input(Message::ActionTextChanged)
                .width(Length::FillPortion(3)),
            button("发送文本")
                .on_press_maybe((!app.busy).then_some(Message::SendInputText)),
            button("恢复窗口/输入")
                .on_press_maybe((!app.busy).then_some(Message::SendInactive)),
        ]
        .spacing(8),
        text("提示：仅 ADB/窗口源支持自动化；显示器源会返回失败。Win32 输入方法仅对窗口源生效。")
            .size(13),
    ]
    .spacing(8);

    let path_panel = column![
        text("路径与资源").size(20),
        text_input("资源根目录", &app.resource_root_text).on_input(Message::ResourceRootChanged),
        text_input("模型路径", &app.model_path_text).on_input(Message::ModelPathChanged),
        text_input("MAA 动态库路径", &app.maa_library_path_text)
            .on_input(Message::MaaLibraryPathChanged),
        text_input("OCR 模型目录", &app.ocr_model_path_text).on_input(Message::OcrModelPathChanged),
        text_input("deepseek-ocr-cli 路径", &app.deepseek_cli_path_text)
            .on_input(Message::DeepseekCliPathChanged),
        pick_list(
            DeepseekCliModel::ALL.to_vec(),
            Some(app.config.deepseek_model),
            Message::DeepseekModelSelected
        )
        .placeholder("选择 deepseek-ocr 模型"),
        text_input("PaddleOCR-VL 设备", &app.deepseek_device_text)
            .on_input(Message::DeepseekDeviceChanged),
        pick_list(
            vec![OcrBackend::Maa, OcrBackend::DeepseekCli],
            Some(app.config.ocr_backend),
            Message::OcrBackendSelected
        )
        .placeholder("选择 OCR 后端"),
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
                container(column![path_panel, roi_panel, automation_panel].spacing(16))
                    .width(Length::FillPortion(1)),
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
