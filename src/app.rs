use crate::automation::{is_automation_allowed, execute_action, AutomationAction};
use crate::auto_fetch::AutoFetch;
use crate::capture::{discover_sources, is_16by9};
use crate::config::{AppConfig, Win32InputMethodConfig};
use crate::core::{
    AnalysisOutput, CaptureCatalog, CaptureSource, GameMode, LrScheduler, ModelSelection, PredictionResult,
    RosterPanelState, RosterSource, Side, SourceChoice, TrainConfig, TrainProgress, TrainResult,
    UiMode,
};
use crate::model_scanner::ModelScanner;
use crate::ocr::{DeepseekCliModel, OcrBackend, library_hint, ocr_hint};
use crate::path_utils::normalize_path;
use crate::pipeline::AnalysisPipeline;
use crate::recognition::default_battle_roi;
use crate::resources::ResourceStore;
use crate::roster::RosterManager;
use crate::training::TrainingPipeline;
use crate::visualization::VisualizationRenderer;
use iced::widget::{
    button, checkbox, column, container, image, pick_list, row, scrollable, text, text_input,
    Space,
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
    // Training
    StartTraining,
    TrainingProgress(TrainProgress),
    TrainingFinished(Result<TrainResult, String>),
    PollTrainingProgress,
    TrainDataFileChanged(String),
    TrainBatchSizeChanged(String),
    TrainEmbedDimChanged(String),
    TrainLayersChanged(String),
    TrainHeadsChanged(String),
    TrainLrChanged(String),
    TrainEpochsChanged(String),
    TrainSeedChanged(String),
    TrainMaxFeatureValueChanged(String),
    // Auto fetch
    ToggleAutoFetch,
    InvestModeToggled(bool),
    // History
    ToggleHistoryPanel,
    SpecialMonsterMessage(String),
    // ── 新增：界面模式 ──
    UiModeChanged(UiMode),
    // ── 新增：手动阵容 ──
    ToggleRosterPanel,
    RosterSlotClicked(Side, usize),
    MonsterSelected(u32, String),
    MonsterCountInput(String),
    RosterSlotCleared(Side, usize),
    PredictFromRoster,
    MonsterPickerClosed,
    // ── 新增：模型扫描与选择 ──
    ModelSelectionChanged(ModelSelection),
    OpenModelFileDialog,
    ModelFilePicked(Option<std::path::PathBuf>),
    OpenResourceDirDialog,
    ResourceDirPicked(Option<std::path::PathBuf>),
    // ── 新增：可视化 ──
    ToggleVisualization,
    // ── 新增：ROI 图形化选择 ──
    RoiDragStart((f32, f32)),
    RoiDragging((f32, f32)),
    RoiDragEnd,
    // ── 新增：训练高级配置 ──
    TrainDropoutChanged(String),
    TrainLrSchedulerSelected(LrScheduler),
    TrainGradientClipNormChanged(String),
    // ── 新增：路径浏览按钮 ──
    OpenModelPathDialog,
    ModelPathPicked(Option<std::path::PathBuf>),
    OpenMaaLibraryDialog,
    MaaLibraryPicked(Option<std::path::PathBuf>),
    OpenOcrModelDialog,
    OcrModelPicked(Option<std::path::PathBuf>),
    OpenDeepseekCliDialog,
    DeepseekCliPicked(Option<std::path::PathBuf>),
    OpenFieldModelDialog,
    FieldModelPicked(Option<std::path::PathBuf>),
    OpenHistoryDataDialog,
    HistoryDataPicked(Option<std::path::PathBuf>),
    OpenScreenshotsDirDialog,
    ScreenshotsDirPicked(Option<std::path::PathBuf>),
    // ── 新增：路径文本输入 ──
    FieldModelPathChanged(String),
    HistoryDataPathChanged(String),
    ScreenshotsDirChanged(String),
    // ── 新增：怪物选择弹窗搜索 ──
    MonsterFilterChanged(String),
}

/// ROI 拖拽状态
#[derive(Debug, Clone)]
struct RoiDragState {
    start: (f32, f32),
    current: (f32, f32),
}

pub struct CannotMaxApp {
    config: AppConfig,
    catalog: CaptureCatalog,
    resources_summary: String,
    selected_source: Option<SourceChoice>,
    preview: Option<image::Handle>,
    slot_previews: Vec<image::Handle>,
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
    model_loaded: bool,
    // Training state
    train_data_file_text: String,
    train_batch_size_text: String,
    train_embed_dim_text: String,
    train_layers_text: String,
    train_heads_text: String,
    train_lr_text: String,
    train_epochs_text: String,
    train_seed_text: String,
    train_max_feature_value_text: String,
    training_progress: Option<TrainProgress>,
    training_busy: bool,
    training_receiver: Option<std::sync::mpsc::Receiver<TrainProgress>>,
    // Auto fetch state
    auto_fetch: AutoFetch,
    auto_fetch_running: bool,
    // History state
    history_visible: bool,
    history_results: Vec<String>,
    special_messages: String,
    // ── 新增字段 ──
    ui_mode: UiMode,
    roster_manager: RosterManager,
    monster_count_input: String,
    available_models: Vec<crate::core::ModelEntry>,
    current_model_selection: ModelSelection,
    visualization_enabled: bool,
    visualization_overlay: Option<crate::core::VisualizationOverlay>,
    roi_dragging: Option<RoiDragState>,
    // ── 新增：训练高级配置 ──
    train_dropout_text: String,
    train_gradient_clip_norm_text: String,
    // ── 新增：路径字段 ──
    field_model_path_text: String,
    history_data_path_text: String,
    screenshots_dir_text: String,
}

impl CannotMaxApp {
    fn new() -> Self {
        let config = AppConfig::load();
        let default_roi = config.roi.unwrap_or_default();
        let resources_summary = summarize_runtime(&config);
        let tc = &config.train_config;

        // 扫描可用模型
        let models_dir = ModelScanner::default_models_dir();
        let available_models = ModelScanner::scan(&models_dir);

        // 初始化模型选择
        let current_model_selection = if config.model_path.exists() {
            // 尝试匹配扫描到的模型
            available_models
                .iter()
                .find(|entry| entry.path == config.model_path)
                .map(|entry| ModelSelection::Scanned(entry.clone()))
                .unwrap_or_else(|| ModelSelection::Custom(config.model_path.clone()))
        } else if let Some(first) = available_models.first() {
            ModelSelection::Scanned(first.clone())
        } else {
            ModelSelection::OtherOption
        };

        // 初始化阵容面板
        let roster_panel = RosterPanelState {
            roster: config.last_roster.clone().unwrap_or_default(),
            source: RosterSource::AutoRecognized,
            expanded: config.roster_expanded,
            monster_picker_open: false,
            picker_target: None,
            monster_filter: String::new(),
        };

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
            train_data_file_text: tc.data_file.display().to_string(),
            train_batch_size_text: tc.batch_size.to_string(),
            train_embed_dim_text: tc.embed_dim.to_string(),
            train_layers_text: tc.n_layers.to_string(),
            train_heads_text: tc.num_heads.to_string(),
            train_lr_text: format!("{:.e}", tc.learning_rate),
            train_epochs_text: tc.epochs.to_string(),
            train_seed_text: tc.seed.to_string(),
            train_max_feature_value_text: tc.max_feature_value.to_string(),
            training_progress: None,
            training_busy: false,
            training_receiver: None,
            model_loaded: config.model_path.exists(),
            config: config.clone(),
            catalog: CaptureCatalog::default(),
            resources_summary,
            selected_source: None,
            preview: None,
            slot_previews: Vec::new(),
            recognized_rows: Vec::new(),
            prediction: None,
            status: "正在加载输入源…".to_string(),
            busy: false,
            auto_fetch: AutoFetch::new(),
            auto_fetch_running: false,
            history_visible: false,
            history_results: Vec::new(),
            special_messages: String::new(),
            // 新增字段
            ui_mode: config.ui_mode,
            roster_manager: RosterManager {
                panel: roster_panel,
                available_monsters: Vec::new(),
            },
            monster_count_input: String::new(),
            available_models,
            current_model_selection,
            visualization_enabled: config.visualization_enabled,
            visualization_overlay: None,
            roi_dragging: None,
            // 新增：训练高级配置
            train_dropout_text: format!("{:.2}", tc.dropout),
            train_gradient_clip_norm_text: format!("{:.2}", tc.gradient_clip_norm),
            // 新增：路径字段
            field_model_path_text: config.field_model_path.display().to_string(),
            history_data_path_text: config.history_data_path.display().to_string(),
            screenshots_dir_text: config.screenshots_dir.display().to_string(),
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

    fn current_train_config(&self) -> TrainConfig {
        TrainConfig {
            data_file: self.train_data_file_text.clone().into(),
            batch_size: self.train_batch_size_text.parse().unwrap_or(1024),
            test_size: 0.1,
            embed_dim: self.train_embed_dim_text.parse().unwrap_or(128),
            n_layers: self.train_layers_text.parse().unwrap_or(3),
            num_heads: self.train_heads_text.parse().unwrap_or(16),
            learning_rate: self.train_lr_text.parse().unwrap_or(3e-4),
            epochs: self.train_epochs_text.parse().unwrap_or(200),
            seed: self.train_seed_text.parse().unwrap_or(42),
            save_dir: "models".into(),
            max_feature_value: self.train_max_feature_value_text.parse().unwrap_or(100.0),
            weight_decay: 1e-1,
            dropout: self.train_dropout_text.parse().unwrap_or(0.2),
            lr_scheduler: self.config.train_config.lr_scheduler.clone(),
            gradient_clip_norm: self.train_gradient_clip_norm_text.parse().unwrap_or(1.0),
        }
    }

    fn save_config(&mut self) {
        self.config.resource_root = self.resource_root_text.clone().into();
        self.config.model_path = self.model_path_text.clone().into();
        self.config.maa_library_path = self.maa_library_path_text.clone().into();
        self.config.ocr_model_path = self.ocr_model_path_text.clone().into();
        self.config.deepseek_cli_path = self.deepseek_cli_path_text.clone().into();
        self.config.deepseek_device = self.deepseek_device_text.clone();
        self.config.roi = self.current_roi().filter(|roi| !roi.is_empty());
        self.config.train_config = self.current_train_config();
        self.config.ui_mode = self.ui_mode;
        self.config.visualization_enabled = self.visualization_enabled;
        self.config.roster_expanded = self.roster_manager.panel.expanded;
        self.config.last_roster = Some(self.roster_manager.panel.roster.clone());
        self.config.field_model_path = self.field_model_path_text.clone().into();
        self.config.history_data_path = self.history_data_path_text.clone().into();
        self.config.screenshots_dir = self.screenshots_dir_text.clone().into();
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

            // 检查16:9比例
            if let Some(CaptureSource::Monitor(idx)) = app.config.last_capture_source.as_ref() {
                if let Some(monitor) = app.catalog.monitors.iter().find(|m| m.index == *idx) {
                    if !is_16by9(monitor.width, monitor.height) {
                        app.status = format!(
                            "提示: 显示器分辨率 {}x{} 非 16:9 比例",
                            monitor.width, monitor.height
                        );
                    }
                }
            }

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
                    let history_csv = config.resource_root.join("arknights.csv");
                    let pipeline = if history_csv.exists() {
                        AnalysisPipeline::with_history(config.model_path.clone(), history_csv)
                    } else {
                        AnalysisPipeline::new(config.model_path.clone())
                    };
                    pipeline.run(&source.source, &catalog, &config)
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

            // 自动化操作模式限制检查
            if let Err(e) = is_automation_allowed(app.config.game_mode, &source.source) {
                app.status = e;
                return Task::none();
            }

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

            if let Err(e) = is_automation_allowed(app.config.game_mode, &source.source) {
                app.status = e;
                return Task::none();
            }

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

            if let Err(e) = is_automation_allowed(app.config.game_mode, &source.source) {
                app.status = e;
                return Task::none();
            }

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
                    let frame_image = output.frame.image.clone();
                    let roi_for_preview = output
                        .snapshot
                        .roi
                        .unwrap_or_else(|| crate::core::Roi {
                            x: 0,
                            y: 0,
                            width: frame_image.width().max(1),
                            height: frame_image.height().max(1),
                        })
                        .clamp(frame_image.width(), frame_image.height());
                    let roi_image = ::image::imageops::crop_imm(
                        &frame_image,
                        roi_for_preview.x,
                        roi_for_preview.y,
                        roi_for_preview.width.max(1),
                        roi_for_preview.height.max(1),
                    )
                    .to_image();

                    let slot_images = VisualizationRenderer::extract_slot_images(&roi_image);
                    app.slot_previews = slot_images
                        .into_iter()
                        .map(|img| {
                            image::Handle::from_rgba(img.width(), img.height(), img.into_raw())
                        })
                        .collect();

                    // 可视化渲染
                    if app.visualization_enabled {
                        let overlay = VisualizationRenderer::build_overlay(
                            &output.snapshot,
                            (roi_image.width(), roi_image.height()),
                        );
                        let annotated = VisualizationRenderer::render_overlay(&roi_image, &overlay);
                        app.visualization_overlay = Some(overlay);
                        app.preview = Some(image::Handle::from_rgba(
                            annotated.width(),
                            annotated.height(),
                            annotated.into_raw(),
                        ));
                    } else {
                        app.visualization_overlay = None;
                        app.preview = Some(image::Handle::from_rgba(
                            roi_image.width(),
                            roi_image.height(),
                            roi_image.into_raw(),
                        ));
                    }

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

                    // 自动同步阵容
                    app.roster_manager.sync_from_recognition(&output.snapshot.units);

                    app.special_messages = output.special_messages;
                    app.history_results = output.history_results;

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
        // Training messages
        Message::StartTraining => {
            if app.training_busy {
                return Task::none();
            }
            app.training_busy = true;
            app.training_progress = None;
            app.status = "训练已启动…".to_string();
            let train_config = app.current_train_config();

            let (sender, receiver) = std::sync::mpsc::channel();
            app.training_receiver = Some(receiver);
            std::thread::spawn(move || {
                let pipeline = TrainingPipeline::new(train_config);
                let result = pipeline.run(sender);
                result
            });

            Task::perform(
                async {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                },
                |_| Message::PollTrainingProgress,
            )
        }
        Message::TrainingProgress(progress) => {
            app.training_progress = Some(progress.clone());
            app.status = format!(
                "Epoch {}/{} | 训练损失 {:.4} | 验证准确率 {:.2}% | 设备 {}",
                progress.epoch, progress.total_epochs,
                progress.train_loss, progress.val_acc, progress.device_info
            );
            if progress.epoch < progress.total_epochs {
                Task::perform(
                    async {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    },
                    |_| Message::PollTrainingProgress,
                )
            } else {
                app.training_busy = false;
                Task::none()
            }
        }
        Message::PollTrainingProgress => {
            if let Some(receiver) = &app.training_receiver {
                let mut last_progress = None;
                while let Ok(progress) = receiver.try_recv() {
                    last_progress = Some(progress);
                }

                if let Some(progress) = last_progress {
                    if progress.epoch >= progress.total_epochs {
                        app.training_busy = false;
                        app.training_progress = Some(progress.clone());
                        app.status = format!(
                            "训练完成! 最佳准确率 {:.2}%, 最佳损失 {:.4}",
                            progress.best_acc, progress.best_loss
                        );
                        app.training_receiver = None;
                        Task::none()
                    } else {
                        app.training_progress = Some(progress.clone());
                        app.status = format!(
                            "Epoch {}/{} | 训练损失 {:.4} | 验证准确率 {:.2}% | 设备 {}",
                            progress.epoch, progress.total_epochs,
                            progress.train_loss, progress.val_acc, progress.device_info
                        );
                        Task::perform(
                            async {
                                std::thread::sleep(std::time::Duration::from_millis(200));
                            },
                            |_| Message::PollTrainingProgress,
                        )
                    }
                } else if app.training_busy {
                    Task::perform(
                        async {
                            std::thread::sleep(std::time::Duration::from_millis(200));
                        },
                        |_| Message::PollTrainingProgress,
                    )
                } else {
                    Task::none()
                }
            } else {
                Task::none()
            }
        }
        Message::TrainingFinished(result) => {
            app.training_busy = false;
            match result {
                Ok(train_result) => {
                    app.status = format!(
                        "训练完成! 最佳准确率 {:.2}%, 最佳损失 {:.4}",
                        train_result.best_acc, train_result.best_loss
                    );
                }
                Err(error) => {
                    app.status = format!("训练失败: {error}");
                }
            }
            Task::none()
        }
        Message::TrainDataFileChanged(value) => {
            app.train_data_file_text = value;
            Task::none()
        }
        Message::TrainBatchSizeChanged(value) => {
            app.train_batch_size_text = value;
            Task::none()
        }
        Message::TrainEmbedDimChanged(value) => {
            app.train_embed_dim_text = value;
            Task::none()
        }
        Message::TrainLayersChanged(value) => {
            app.train_layers_text = value;
            Task::none()
        }
        Message::TrainHeadsChanged(value) => {
            app.train_heads_text = value;
            Task::none()
        }
        Message::TrainLrChanged(value) => {
            app.train_lr_text = value;
            Task::none()
        }
        Message::TrainEpochsChanged(value) => {
            app.train_epochs_text = value;
            Task::none()
        }
        Message::TrainSeedChanged(value) => {
            app.train_seed_text = value;
            Task::none()
        }
        Message::TrainMaxFeatureValueChanged(value) => {
            app.train_max_feature_value_text = value;
            Task::none()
        }
        // Auto fetch
        Message::ToggleAutoFetch => {
            if app.auto_fetch_running {
                let _ = app.auto_fetch.stop();
                app.auto_fetch_running = false;
                app.status = "自动获取已停止".to_string();
            } else {
                let Some(source) = app.selected_source.clone() else {
                    app.status = "请先选择一个输入源".to_string();
                    return Task::none();
                };

                // 自动化操作模式限制检查
                if let Err(e) = is_automation_allowed(app.config.game_mode, &source.source) {
                    app.status = e;
                    return Task::none();
                }

                let config = app.config.clone();
                let catalog = app.catalog.clone();
                let game_mode = match app.config.game_mode {
                    GameMode::Pc => "PC".to_string(),
                    GameMode::WindowOnly => "WindowOnly".to_string(),
                    GameMode::Emulator => "模拟器".to_string(),
                };
                match app.auto_fetch.start(
                    source.source,
                    catalog,
                    config,
                    game_mode,
                    app.config.invest_mode,
                    -1.0,
                ) {
                    Ok(()) => {
                        app.auto_fetch_running = true;
                        app.status = "自动获取已启动".to_string();
                    }
                    Err(e) => app.status = format!("自动获取启动失败: {e}"),
                }
            }
            Task::none()
        }
        Message::InvestModeToggled(value) => {
            app.config.invest_mode = value;
            let _ = app.config.save();
            Task::none()
        }
        // History
        Message::ToggleHistoryPanel => {
            app.history_visible = !app.history_visible;
            Task::none()
        }
        Message::SpecialMonsterMessage(msg) => {
            app.special_messages = msg;
            Task::none()
        }
        // ── 新增：界面模式 ──
        Message::UiModeChanged(mode) => {
            // 切换模式时若自动获取运行中先停止
            if app.auto_fetch_running {
                let _ = app.auto_fetch.stop();
                app.auto_fetch_running = false;
                app.status = "自动获取已停止（模式切换）".to_string();
            }
            app.ui_mode = mode;
            app.config.ui_mode = mode;
            let _ = app.config.save();
            Task::none()
        }
        // ── 新增：手动阵容 ──
        Message::ToggleRosterPanel => {
            app.roster_manager.toggle_expanded();
            Task::none()
        }
        Message::RosterSlotClicked(side, index) => {
            app.roster_manager.open_monster_picker(side, index);
            Task::none()
        }
        Message::MonsterSelected(id, name) => {
            if let Some((side, index)) = app.roster_manager.panel.picker_target {
                let count: u32 = app.monster_count_input.parse().unwrap_or(1);
                app.roster_manager.set_slot(side, index, id, name, count);
            }
            app.roster_manager.close_monster_picker();
            app.monster_count_input.clear();
            Task::none()
        }
        Message::MonsterCountInput(value) => {
            app.monster_count_input = value;
            Task::none()
        }
        Message::RosterSlotCleared(side, index) => {
            app.roster_manager.clear_slot(side, index);
            Task::none()
        }
        Message::PredictFromRoster => {
            let Some(source) = app.selected_source.clone() else {
                app.status = "请先选择一个输入源".to_string();
                return Task::none();
            };

            app.busy = true;
            app.status = "正在从手动阵容预测…".to_string();

            let snapshot = app.roster_manager.to_battle_snapshot(
                &source.source,
                (1280, 720),
                app.current_roi().filter(|roi| !roi.is_empty()),
            );
            let config = app.config.clone();

            Task::perform(
                async move {
                    let pipeline = AnalysisPipeline::new(config.model_path.clone());
                    pipeline.predict_from_snapshot(&snapshot, &config)
                },
                Message::AnalysisFinished,
            )
        }
        Message::MonsterPickerClosed => {
            app.roster_manager.close_monster_picker();
            Task::none()
        }
        // ── 新增：模型扫描与选择 ──
        Message::ModelSelectionChanged(selection) => {
            match &selection {
                ModelSelection::Scanned(entry) => {
                    app.config.model_path = entry.path.clone();
                    app.model_path_text = entry.path.display().to_string();
                    app.model_loaded = entry.path.exists();
                    let _ = app.config.save();
                }
                ModelSelection::Custom(path) => {
                    app.config.model_path = path.clone();
                    app.model_path_text = path.display().to_string();
                    app.model_loaded = path.exists();
                    let _ = app.config.save();
                }
                ModelSelection::OtherOption => {
                    // 触发文件对话框
                    return Task::perform(
                        async {
                            let file = rfd::AsyncFileDialog::new()
                                .add_filter("safetensors", &["safetensors"])
                                .pick_file()
                                .await;
                            file.map(|f| f.path().to_path_buf())
                        },
                        Message::ModelFilePicked,
                    );
                }
            }
            app.current_model_selection = selection;
            Task::none()
        }
        Message::OpenModelFileDialog => {
            Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("safetensors", &["safetensors"])
                        .pick_file()
                        .await;
                    file.map(|f| f.path().to_path_buf())
                },
                Message::ModelFilePicked,
            )
        }
        Message::ModelFilePicked(path) => {
            if let Some(path) = path {
                let workspace_root = AppConfig::workspace_root();
                let normalized = normalize_path(&path, &workspace_root);
                app.config.model_path = normalized.stored.clone();
                app.model_path_text = normalized.display.display().to_string();
                app.model_loaded = path.exists();
                app.current_model_selection = ModelSelection::Custom(normalized.stored);
                let _ = app.config.save();
                app.status = format!("模型已加载: {}", normalized.display.display());
            }
            Task::none()
        }
        Message::OpenResourceDirDialog => {
            Task::perform(
                async {
                    let folder = rfd::AsyncFileDialog::new().pick_folder().await;
                    folder.map(|f| f.path().to_path_buf())
                },
                Message::ResourceDirPicked,
            )
        }
        Message::ResourceDirPicked(path) => {
            if let Some(path) = path {
                let workspace_root = AppConfig::workspace_root();
                let normalized = normalize_path(&path, &workspace_root);
                app.config.resource_root = normalized.stored.clone();
                app.resource_root_text = normalized.display.display().to_string();
                let _ = app.config.save();
                app.resources_summary = summarize_runtime(&app.config);
                app.status = format!("资源目录已设置: {}", normalized.display.display());
            }
            Task::none()
        }
        // ── 新增：可视化 ──
        Message::ToggleVisualization => {
            app.visualization_enabled = !app.visualization_enabled;
            app.config.visualization_enabled = app.visualization_enabled;
            let _ = app.config.save();
            Task::none()
        }
        // ── 新增：ROI 图形化选择 ──
        Message::RoiDragStart(pos) => {
            app.roi_dragging = Some(RoiDragState {
                start: pos,
                current: pos,
            });
            Task::none()
        }
        Message::RoiDragging(pos) => {
            if let Some(state) = &mut app.roi_dragging {
                state.current = pos;
            }
            Task::none()
        }
        Message::RoiDragEnd => {
            if let Some(state) = app.roi_dragging.take() {
                // 将相对坐标转换为ROI
                let x = (state.start.0.min(state.current.0) * 1280.0) as u32;
                let y = (state.start.1.min(state.current.1) * 720.0) as u32;
                let w = ((state.current.0 - state.start.0).abs() * 1280.0) as u32;
                let h = ((state.current.1 - state.start.1).abs() * 720.0) as u32;
                app.roi_x = x.to_string();
                app.roi_y = y.to_string();
                app.roi_width = w.to_string();
                app.roi_height = h.to_string();
            }
            Task::none()
        }
        // ── 新增：训练高级配置 ──
        Message::TrainDropoutChanged(value) => {
            app.train_dropout_text = value;
            Task::none()
        }
        Message::TrainLrSchedulerSelected(scheduler) => {
            app.config.train_config.lr_scheduler = scheduler;
            let _ = app.config.save();
            Task::none()
        }
        Message::TrainGradientClipNormChanged(value) => {
            app.train_gradient_clip_norm_text = value;
            Task::none()
        }
        // ── 新增：路径浏览按钮 ──
        Message::OpenModelPathDialog => {
            Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("safetensors", &["safetensors"])
                        .pick_file()
                        .await;
                    file.map(|f| f.path().to_path_buf())
                },
                Message::ModelPathPicked,
            )
        }
        Message::ModelPathPicked(path) => {
            if let Some(path) = path {
                let workspace_root = AppConfig::workspace_root();
                let normalized = normalize_path(&path, &workspace_root);
                app.config.model_path = normalized.stored.clone();
                app.model_path_text = normalized.display.display().to_string();
                app.model_loaded = path.exists();
                let _ = app.config.save();
            }
            Task::none()
        }
        Message::OpenMaaLibraryDialog => {
            Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("动态库", &["dll", "so", "dylib"])
                        .pick_file()
                        .await;
                    file.map(|f| f.path().to_path_buf())
                },
                Message::MaaLibraryPicked,
            )
        }
        Message::MaaLibraryPicked(path) => {
            if let Some(path) = path {
                app.config.maa_library_path = path.clone();
                app.maa_library_path_text = path.display().to_string();
                let _ = app.config.save();
            }
            Task::none()
        }
        Message::OpenOcrModelDialog => {
            Task::perform(
                async {
                    let folder = rfd::AsyncFileDialog::new().pick_folder().await;
                    folder.map(|f| f.path().to_path_buf())
                },
                Message::OcrModelPicked,
            )
        }
        Message::OcrModelPicked(path) => {
            if let Some(path) = path {
                app.config.ocr_model_path = path.clone();
                app.ocr_model_path_text = path.display().to_string();
                let _ = app.config.save();
            }
            Task::none()
        }
        Message::OpenDeepseekCliDialog => {
            Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("可执行文件", &["exe"])
                        .pick_file()
                        .await;
                    file.map(|f| f.path().to_path_buf())
                },
                Message::DeepseekCliPicked,
            )
        }
        Message::DeepseekCliPicked(path) => {
            if let Some(path) = path {
                app.config.deepseek_cli_path = path.clone();
                app.deepseek_cli_path_text = path.display().to_string();
                let _ = app.config.save();
            }
            Task::none()
        }
        Message::OpenFieldModelDialog => {
            Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("safetensors", &["safetensors"])
                        .pick_file()
                        .await;
                    file.map(|f| f.path().to_path_buf())
                },
                Message::FieldModelPicked,
            )
        }
        Message::FieldModelPicked(path) => {
            if let Some(path) = path {
                app.config.field_model_path = path.clone();
                app.field_model_path_text = path.display().to_string();
                let _ = app.config.save();
            }
            Task::none()
        }
        Message::OpenHistoryDataDialog => {
            Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("CSV", &["csv"])
                        .pick_file()
                        .await;
                    file.map(|f| f.path().to_path_buf())
                },
                Message::HistoryDataPicked,
            )
        }
        Message::HistoryDataPicked(path) => {
            if let Some(path) = path {
                app.config.history_data_path = path.clone();
                app.history_data_path_text = path.display().to_string();
                let _ = app.config.save();
            }
            Task::none()
        }
        Message::OpenScreenshotsDirDialog => {
            Task::perform(
                async {
                    let folder = rfd::AsyncFileDialog::new().pick_folder().await;
                    folder.map(|f| f.path().to_path_buf())
                },
                Message::ScreenshotsDirPicked,
            )
        }
        Message::ScreenshotsDirPicked(path) => {
            if let Some(path) = path {
                app.config.screenshots_dir = path.clone();
                app.screenshots_dir_text = path.display().to_string();
                let _ = app.config.save();
            }
            Task::none()
        }
        // ── 新增：路径文本输入 ──
        Message::FieldModelPathChanged(value) => {
            app.field_model_path_text = value;
            Task::none()
        }
        Message::HistoryDataPathChanged(value) => {
            app.history_data_path_text = value;
            Task::none()
        }
        Message::ScreenshotsDirChanged(value) => {
            app.screenshots_dir_text = value;
            Task::none()
        }
        // ── 新增：怪物选择弹窗搜索 ──
        Message::MonsterFilterChanged(value) => {
            app.roster_manager.panel.monster_filter = value;
            Task::none()
        }
    }
}

// ── 视图渲染 ──

pub fn view(app: &CannotMaxApp) -> Element<'_, Message> {
    let header = view_header(app);

    let content = match app.ui_mode {
        UiMode::Normal => view_normal_mode(app),
        UiMode::Developer => view_developer_mode(app),
    };

    container(scrollable(
        column![header, content].spacing(20).padding(20),
    ))
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn view_header(app: &CannotMaxApp) -> Element<'_, Message> {
    let mode_toggle = row![
        button(if app.ui_mode == UiMode::Normal {
            "普通模式"
        } else {
            "切换到普通模式"
        })
        .on_press(Message::UiModeChanged(UiMode::Normal)),
        button(if app.ui_mode == UiMode::Developer {
            "开发者模式"
        } else {
            "切换到开发者模式"
        })
        .on_press(Message::UiModeChanged(UiMode::Developer)),
    ]
    .spacing(8);

    column![
        text("cannot-max-rs").size(32),
        text("二期增强: 训练 + 自动获取 + 历史匹配 + 场地识别 + PC端兼容").size(16),
        text(&app.status).size(14),
        mode_toggle,
    ]
    .spacing(8)
    .into()
}

/// 普通模式视图
fn view_normal_mode(app: &CannotMaxApp) -> Element<'_, Message> {
    let source_choices = app.catalog.source_choices(app.config.game_mode);

    let controls = column![
        row![
            button("刷新输入源").on_press(Message::RefreshSources),
            button(if app.busy {
                "处理中…"
            } else if !app.model_loaded {
                "识别并预测（无模型，基线模式）"
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
        view_game_mode_selector(app),
    ]
    .spacing(12);

    // 模型下拉选择（含"选择其他模型…"）
    let mut model_options: Vec<ModelSelection> = app
        .available_models
        .iter()
        .map(|entry| ModelSelection::Scanned(entry.clone()))
        .collect();
    model_options.push(ModelSelection::OtherOption);

    let model_panel = column![
        text("模型选择").size(16),
        pick_list(
            model_options,
            Some(app.current_model_selection.clone()),
            Message::ModelSelectionChanged
        )
        .placeholder("选择模型文件"),
    ]
    .spacing(8);

    // 手动阵容面板
    let roster_panel = view_roster_panel(app);

    // 自动获取面板（仅在游戏模式下显示）
    let auto_fetch_panel = if app.config.game_mode != GameMode::WindowOnly {
        view_auto_fetch_panel(app)
    } else {
        column![].into()
    };

    // 识别结果可视化开关
    let visualization_toggle = view_visualization_toggle(app);

    // 预览与结果
    let preview_panel = view_preview(app);
    let result_panel = view_result_panel(app);
    let special_panel = view_special_panel(app);

    // 怪物选择弹窗覆盖层
    let monster_picker_overlay = if app.roster_manager.panel.monster_picker_open {
        view_monster_picker_overlay(app)
    } else {
        column![].into()
    };

    row![
        container(scrollable(column![
            controls,
            model_panel,
            roster_panel,
            auto_fetch_panel,
            visualization_toggle,
        ].spacing(16)).width(Length::Fill).height(Length::Fill))
            .width(Length::FillPortion(1)),
        container(scrollable(column![
            preview_panel,
            result_panel,
            special_panel,
        ].spacing(16)).width(Length::Fill).height(Length::Fill))
            .width(Length::FillPortion(2)),
    ]
    .spacing(20)
    .push(monster_picker_overlay)
    .into()
}

/// 开发者模式视图
fn view_developer_mode(app: &CannotMaxApp) -> Element<'_, Message> {
    let source_choices = app.catalog.source_choices(app.config.game_mode);

    let controls = column![
        row![
            button("刷新输入源").on_press(Message::RefreshSources),
            button(if app.busy {
                "处理中…"
            } else if !app.model_loaded {
                "识别并预测（无模型，基线模式）"
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
        view_game_mode_selector(app),
    ]
    .spacing(12);

    // 开发者模型下拉选择（含"选择其他模型"）
    let mut model_options: Vec<ModelSelection> = app
        .available_models
        .iter()
        .map(|entry| ModelSelection::Scanned(entry.clone()))
        .collect();
    model_options.push(ModelSelection::OtherOption);

    let model_panel = column![
        text("模型选择").size(16),
        pick_list(
            model_options,
            Some(app.current_model_selection.clone()),
            Message::ModelSelectionChanged
        )
        .placeholder("选择模型文件"),
    ]
    .spacing(8);

    let roster_panel = view_roster_panel(app);
    let auto_fetch_panel = view_auto_fetch_panel(app);
    let path_panel = view_path_panel(app);
    let roi_panel = view_roi_panel(app);
    let automation_panel = view_automation_panel(app);
    let training_panel = view_training_panel(app);
    let visualization_toggle = view_visualization_toggle(app);
    let history_panel = view_history_panel(app);

    let preview_panel = view_preview(app);
    let result_panel = view_result_panel(app);
    let special_panel = view_special_panel(app);

    // 怪物选择弹窗覆盖层
    let monster_picker_overlay = if app.roster_manager.panel.monster_picker_open {
        view_monster_picker_overlay(app)
    } else {
        column![].into()
    };

    row![
        container(scrollable(column![
            controls,
            model_panel,
            roster_panel,
            auto_fetch_panel,
            path_panel,
            roi_panel,
            automation_panel,
            training_panel,
            visualization_toggle,
            history_panel,
        ].spacing(16)).width(Length::Fill).height(Length::Fill))
            .width(Length::FillPortion(1)),
        container(scrollable(column![
            preview_panel,
            result_panel,
            special_panel,
        ].spacing(16)).width(Length::Fill).height(Length::Fill))
            .width(Length::FillPortion(2)),
    ]
    .spacing(20)
    .push(monster_picker_overlay)
    .into()
}

fn view_game_mode_selector(app: &CannotMaxApp) -> Element<'_, Message> {
    row![
        button("模拟器模式")
            .on_press(Message::ModeChanged(GameMode::Emulator)),
        button("PC 模式")
            .on_press(Message::ModeChanged(GameMode::Pc)),
        button("普通窗口模式")
            .on_press(Message::ModeChanged(GameMode::WindowOnly)),
        text(format!("当前: {}", app.config.game_mode)),
    ]
    .spacing(12)
    .align_y(Alignment::Center)
    .into()
}

fn view_roster_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    let toggle_text = if app.roster_manager.panel.expanded {
        "收起手动阵容"
    } else {
        "展开手动阵容"
    };

    let toggle = row![
        button(toggle_text).on_press(Message::ToggleRosterPanel),
        text(format!(
            "来源: {}",
            match app.roster_manager.panel.source {
                RosterSource::AutoRecognized => "自动识别",
                RosterSource::ManualInput => "手动输入",
            }
        ))
        .size(13),
    ]
    .spacing(8);

    if !app.roster_manager.panel.expanded {
        return column![text("手动输入阵容").size(16), toggle].spacing(8).into();
    }

    // 左方3个槽位
    let left_slots: Vec<Element<'_, Message>> = (0..3)
        .map(|i| {
            let slot = &app.roster_manager.panel.roster.left[i];
            let label = slot
                .monster_name
                .as_ref()
                .map(|name| format!("{} x{}", name, slot.count))
                .unwrap_or_else(|| "空".to_string());
            row![
                button(text(label.clone())).on_press(Message::RosterSlotClicked(Side::Left, i)),
                button("清空").on_press(Message::RosterSlotCleared(Side::Left, i)),
            ]
            .spacing(4)
            .into()
        })
        .collect();

    // 右方3个槽位
    let right_slots: Vec<Element<'_, Message>> = (0..3)
        .map(|i| {
            let slot = &app.roster_manager.panel.roster.right[i];
            let label = slot
                .monster_name
                .as_ref()
                .map(|name| format!("{} x{}", name, slot.count))
                .unwrap_or_else(|| "空".to_string());
            row![
                button(text(label.clone())).on_press(Message::RosterSlotClicked(Side::Right, i)),
                button("清空").on_press(Message::RosterSlotCleared(Side::Right, i)),
            ]
            .spacing(4)
            .into()
        })
        .collect();

    column![
        text("手动输入阵容").size(16),
        toggle,
        row![
            column![text("左方").size(14), column(left_slots).spacing(4)],
            column![text("右方").size(14), column(right_slots).spacing(4)],
        ]
        .spacing(20),
        row![
            text_input("数量", &app.monster_count_input)
                .on_input(Message::MonsterCountInput)
                .width(Length::FillPortion(1)),
            button("从阵容预测")
                .on_press_maybe((!app.busy).then_some(Message::PredictFromRoster)),
        ]
        .spacing(8),
    ]
    .spacing(8)
    .into()
}

fn view_auto_fetch_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    let auto_fetch_stats_text = if app.auto_fetch_running {
        let stats = app.auto_fetch.stats_snapshot();
        format!(
            "总填写: {} | 错误: {} | 运行: {:.0}s",
            stats.total_fill_count, stats.incorrect_fill_count, stats.elapsed_secs
        )
    } else {
        "暂无统计".to_string()
    };

    column![
        text("自动获取").size(20),
        row![
            button(if app.auto_fetch_running {
                "停止自动获取"
            } else {
                "启动自动获取"
            })
            .on_press(Message::ToggleAutoFetch),
            checkbox(app.config.invest_mode).on_toggle(Message::InvestModeToggled),
            text("投资模式").size(13),
        ]
        .spacing(12)
        .align_y(Alignment::Center),
        text(auto_fetch_stats_text).size(13),
    ]
    .spacing(8)
    .into()
}

fn view_path_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    column![
        text("路径与资源").size(20),
        row![
            text_input("资源根目录", &app.resource_root_text)
                .on_input(Message::ResourceRootChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenResourceDirDialog),
        ]
        .spacing(8),
        row![
            text_input("模型路径", &app.model_path_text)
                .on_input(Message::ModelPathChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenModelPathDialog),
        ]
        .spacing(8),
        row![
            text_input("MAA 动态库路径", &app.maa_library_path_text)
                .on_input(Message::MaaLibraryPathChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenMaaLibraryDialog),
        ]
        .spacing(8),
        row![
            text_input("OCR 模型目录", &app.ocr_model_path_text)
                .on_input(Message::OcrModelPathChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenOcrModelDialog),
        ]
        .spacing(8),
        row![
            text_input("deepseek-ocr-cli 路径", &app.deepseek_cli_path_text)
                .on_input(Message::DeepseekCliPathChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenDeepseekCliDialog),
        ]
        .spacing(8),
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
        row![
            text_input("场地识别模型路径", &app.field_model_path_text)
                .on_input(Message::FieldModelPathChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenFieldModelDialog),
        ]
        .spacing(8),
        row![
            text_input("历史数据路径", &app.history_data_path_text)
                .on_input(Message::HistoryDataPathChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenHistoryDataDialog),
        ]
        .spacing(8),
        row![
            text_input("截图保存目录", &app.screenshots_dir_text)
                .on_input(Message::ScreenshotsDirChanged)
                .width(Length::Fill),
            button("浏览…").on_press(Message::OpenScreenshotsDirDialog),
        ]
        .spacing(8),
        text(&app.resources_summary).size(14),
    ]
    .spacing(8)
    .into()
}

fn view_roi_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    column![
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
        text("提示：在普通窗口模式下可在预览画面上拖拽框选ROI").size(13),
    ]
    .spacing(8)
    .into()
}

fn view_automation_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    let is_window_only = app.config.game_mode == GameMode::WindowOnly;
    let mode_warning = if is_window_only {
        text("当前为普通窗口模式，不支持自动化操作").size(13)
    } else {
        text("").size(13)
    };

    column![
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
        mode_warning,
    ]
    .spacing(8)
    .into()
}

fn view_training_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    let train_progress_text = if let Some(p) = &app.training_progress {
        format!(
            "Epoch {}/{} | 训练损失 {:.4} | 训练准确率 {:.2}% | 验证损失 {:.4} | 验证准确率 {:.2}% | 最佳准确率 {:.2}% | 最佳损失 {:.4} | 已用 {:.1}s | 预估剩余 {:.1}s | {}",
            p.epoch, p.total_epochs,
            p.train_loss, p.train_acc,
            p.val_loss, p.val_acc,
            p.best_acc, p.best_loss,
            p.elapsed_secs, p.estimated_remaining_secs,
            p.device_info
        )
    } else {
        "暂无训练进度".to_string()
    };

    // 学习率调度器选项
    let lr_scheduler_options = vec![
        LrScheduler::Fixed,
        LrScheduler::CosineAnnealing { t_max: 200, eta_min: 1e-5 },
    ];

    column![
        text("模型训练（UnitAwareTransformer）").size(20),
        text_input("数据文件", &app.train_data_file_text).on_input(Message::TrainDataFileChanged),
        row![
            text_input("batch_size", &app.train_batch_size_text).on_input(Message::TrainBatchSizeChanged).width(Length::FillPortion(1)),
            text_input("embed_dim", &app.train_embed_dim_text).on_input(Message::TrainEmbedDimChanged).width(Length::FillPortion(1)),
            text_input("n_layers", &app.train_layers_text).on_input(Message::TrainLayersChanged).width(Length::FillPortion(1)),
        ].spacing(8),
        row![
            text_input("num_heads", &app.train_heads_text).on_input(Message::TrainHeadsChanged).width(Length::FillPortion(1)),
            text_input("lr", &app.train_lr_text).on_input(Message::TrainLrChanged).width(Length::FillPortion(1)),
            text_input("epochs", &app.train_epochs_text).on_input(Message::TrainEpochsChanged).width(Length::FillPortion(1)),
        ].spacing(8),
        row![
            text_input("seed", &app.train_seed_text).on_input(Message::TrainSeedChanged).width(Length::FillPortion(1)),
            text_input("max_feature_value", &app.train_max_feature_value_text).on_input(Message::TrainMaxFeatureValueChanged).width(Length::FillPortion(1)),
        ].spacing(8),
        text("高级配置").size(14),
        row![
            text("Dropout:").size(13),
            text_input("0.2", &app.train_dropout_text).on_input(Message::TrainDropoutChanged).width(80),
            Space::new().width(Length::Fill),
            text("梯度裁剪:").size(13),
            text_input("1.0", &app.train_gradient_clip_norm_text).on_input(Message::TrainGradientClipNormChanged).width(80),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
        row![
            text("学习率调度器:").size(13),
            pick_list(
                lr_scheduler_options,
                Some(app.config.train_config.lr_scheduler.clone()),
                Message::TrainLrSchedulerSelected
            )
            .placeholder("选择调度器"),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
        button(if app.training_busy { "训练中…" } else { "开始训练" })
            .on_press_maybe((!app.training_busy).then_some(Message::StartTraining)),
        text(train_progress_text).size(13),
    ]
    .spacing(8)
    .into()
}

fn view_visualization_toggle(app: &CannotMaxApp) -> Element<'_, Message> {
    row![
        text("识别结果可视化").size(16),
        checkbox(app.visualization_enabled).on_toggle(|_| Message::ToggleVisualization),
        text(if app.visualization_enabled { "已开启" } else { "已关闭" }).size(13),
    ]
    .spacing(8)
    .into()
}

fn view_history_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    if !app.history_visible {
        return row![
            button("显示历史面板").on_press(Message::ToggleHistoryPanel),
        ]
        .into();
    }

    let history_content = if app.history_results.is_empty() {
        vec![text("选择输入源并识别后，将自动匹配历史对局").size(13).into()]
    } else {
        app.history_results
            .iter()
            .map(|row| text(row).size(13).into())
            .collect::<Vec<Element<'_, Message>>>()
    };

    column![
        row![
            text("历史对局匹配").size(20),
            button("隐藏").on_press(Message::ToggleHistoryPanel),
        ]
        .spacing(8),
        scrollable(column(history_content).spacing(4)).height(200),
    ]
    .spacing(8)
    .into()
}

fn view_preview(app: &CannotMaxApp) -> Element<'_, Message> {
    if let Some(handle) = &app.preview {
        let slot_gallery: Element<'_, Message> = if app.slot_previews.is_empty() {
            text("暂无槽位截图").size(13).into()
        } else {
            let items = app
                .slot_previews
                .iter()
                .enumerate()
                .map(|(idx, slot)| {
                    column![
                        text(format!("槽位 {}", idx + 1)).size(12),
                        image(slot.clone()).width(120).height(80),
                    ]
                    .spacing(4)
                    .into()
                })
                .collect::<Vec<Element<'_, Message>>>();

            scrollable(row(items).spacing(8)).height(120).into()
        };

        // 普通窗口模式下显示ROI框提示
        let roi_hint: Element<_> = if app.config.game_mode == GameMode::WindowOnly {
            text("提示：可在预览画面上拖拽框选ROI").size(13).into()
        } else {
            column![].into()
        };

        // ROI拖拽中的视觉反馈
        let roi_drag_hint: Element<_> = if app.roi_dragging.is_some() {
            text("正在框选ROI区域…").size(13).into()
        } else {
            column![].into()
        };

        column![
            text("当前预览").size(20),
            image(handle.clone()).width(Length::Fill).height(Length::FillPortion(2)),
            roi_hint,
            roi_drag_hint,
            text("槽位截图").size(16),
            slot_gallery,
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
    }
}

fn view_result_panel(app: &CannotMaxApp) -> Element<'_, Message> {
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

    column![
        text("识别与预测").size(20),
        text(prediction_text).size(16),
        scrollable(column(recognized).spacing(6)).height(220),
    ]
    .spacing(8)
    .into()
}

fn view_special_panel(app: &CannotMaxApp) -> Element<'_, Message> {
    if app.special_messages.is_empty() {
        column![].into()
    } else {
        column![
            text("特殊怪物提示").size(16),
            text(&app.special_messages).size(13),
        ]
        .spacing(4)
        .into()
    }
}

/// 怪物选择弹窗覆盖层
fn view_monster_picker_overlay(app: &CannotMaxApp) -> Element<'_, Message> {
    // 搜索过滤
    let filter = app.roster_manager.panel.monster_filter.to_lowercase();
    let filtered_monsters: Vec<_> = app
        .roster_manager
        .available_monsters
        .iter()
        .filter(|m| filter.is_empty() || m.name.to_lowercase().contains(&filter))
        .collect();

    // 怪物网格：每行4个，使用弹性布局
    let monsters_per_row = 4;
    let mut grid_column = column![].spacing(8);

    let mut row_content = row![].spacing(8).width(Length::Fill);
    let mut count_in_row = 0;

    for monster in &filtered_monsters {
        let monster_id = monster.id;
        let monster_name = monster.name.clone();
        let card = button(
            column![
                image(monster.thumbnail.clone()).width(48).height(48),
                text(monster_name.clone()).size(12),
            ]
            .spacing(4)
            .align_x(Alignment::Center)
        )
        .on_press(Message::MonsterSelected(monster_id, monster_name))
        .width(Length::FillPortion(1));

        row_content = row_content.push(card);
        count_in_row += 1;

        if count_in_row >= monsters_per_row {
            grid_column = grid_column.push(row_content);
            row_content = row![].spacing(8).width(Length::Fill);
            count_in_row = 0;
        }
    }
    if count_in_row > 0 {
        grid_column = grid_column.push(row_content);
    }

    let grid_element: Element<_> = if filtered_monsters.is_empty() {
        text("无匹配怪物").size(14).into()
    } else {
        grid_column.into()
    };

    let overlay = container(
        column![
            row![
                text("选择怪物").size(20),
                Space::new().width(Length::Fill),
                button("关闭").on_press(Message::MonsterPickerClosed),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            text_input("搜索怪物名称…", &app.roster_manager.panel.monster_filter)
                .on_input(Message::MonsterFilterChanged)
                .width(Length::Fill),
            scrollable(column![grid_element].spacing(8)).height(400).width(Length::Fill),
        ]
        .spacing(12)
    )
    .width(500)
    .height(Length::Shrink)
    .center_x(Length::Fill)
    .center_y(Length::Fill);

    // 覆盖层背景
    container(overlay)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .into()
}
