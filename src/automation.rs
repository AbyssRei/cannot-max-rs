use crate::config::AppConfig;
use crate::core::{CaptureCatalog, CaptureSource};
use crate::maa_controller::MaaControllerSession;

#[derive(Debug, Clone)]
pub enum AutomationAction {
    Click { x: i32, y: i32 },
    InputText { text: String },
    Inactive,
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