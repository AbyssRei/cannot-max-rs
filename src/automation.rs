use crate::config::AppConfig;
use crate::core::{CaptureCatalog, CaptureSource, GameMode};
use crate::maa_controller::MaaControllerSession;

#[derive(Debug, Clone)]
pub enum AutomationAction {
    Click { x: i32, y: i32 },
    InputText { text: String },
    Inactive,
}

/// 检查当前模式是否支持自动化操作
pub fn is_automation_allowed(game_mode: GameMode, source: &CaptureSource) -> Result<(), String> {
    match game_mode {
        GameMode::WindowOnly => Err("普通窗口模式不支持自动化操作".to_string()),
        _ => match source {
            CaptureSource::Monitor(_) => Err("显示器源不支持自动化操作".to_string()),
            _ => Ok(()),
        },
    }
}

pub fn execute_action(
    source: &CaptureSource,
    catalog: &CaptureCatalog,
    config: &AppConfig,
    action: AutomationAction,
) -> Result<String, String> {
    let session = MaaControllerSession::for_source(source, catalog, config)?;

    match action {
        AutomationAction::Click { x, y } => {
            session.click(x, y)?;
            Ok(format!("自动点击成功: ({x}, {y})"))
        }
        AutomationAction::InputText { text } => {
            let preview = if text.chars().count() > 24 {
                format!("{}...", text.chars().take(24).collect::<String>())
            } else {
                text.clone()
            };
            session.input_text(&text)?;
            Ok(format!("自动输入成功: {preview}"))
        }
        AutomationAction::Inactive => {
            session.inactive()?;
            Ok("已发送 inactive 请求，窗口/输入状态已恢复".to_string())
        }
    }
}