use crate::core::{SpecialMonsterInfo, Winner};
use std::collections::HashMap;
use std::path::Path;

pub struct SpecialMonsterHandler {
    special_monsters: HashMap<u32, SpecialMonsterInfo>,
}

impl SpecialMonsterHandler {
    pub fn new() -> Self {
        Self::with_config(None)
    }

    /// 从 RON 配置文件加载特殊怪物数据，失败时使用内置默认数据
    pub fn with_config(config_path: Option<&Path>) -> Self {
        let special_monsters = if let Some(path) = config_path {
            if path.exists() {
                match Self::load_from_file(path) {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("加载特殊怪物配置失败: {e}，使用默认数据");
                        Self::default_monsters()
                    }
                }
            } else {
                Self::default_monsters()
            }
        } else {
            Self::default_monsters()
        };

        Self { special_monsters }
    }

    /// 从 RON 文件加载特殊怪物映射
    fn load_from_file(path: &Path) -> Result<HashMap<u32, SpecialMonsterInfo>, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("读取文件失败: {e}"))?;
        let data: HashMap<u32, SpecialMonsterInfo> = ron::from_str(&text)
            .map_err(|e| format!("解析 RON 失败: {e}"))?;
        Ok(data)
    }

    /// 内置默认特殊怪物数据
    fn default_monsters() -> HashMap<u32, SpecialMonsterInfo> {
        let mut m = HashMap::new();
        m.insert(1, SpecialMonsterInfo {
            name: "狗神".to_string(),
            win_message: "全军出击，我咬死你！".to_string(),
            lose_message: "牙崩了牙崩了".to_string(),
        });
        m.insert(2, SpecialMonsterInfo {
            name: "泥岩".to_string(),
            win_message: "大锤八十！".to_string(),
            lose_message: "锤不动了…".to_string(),
        });
        m.insert(3, SpecialMonsterInfo {
            name: "浮士德".to_string(),
            win_message: "紫箭降临！".to_string(),
            lose_message: "箭射歪了…".to_string(),
        });
        m.insert(4, SpecialMonsterInfo {
            name: "爱国者".to_string(),
            win_message: "为了乌萨斯！".to_string(),
            lose_message: "乌萨斯不会忘记…".to_string(),
        });
        m.insert(5, SpecialMonsterInfo {
            name: "塔露拉".to_string(),
            win_message: "整合运动，前进！".to_string(),
            lose_message: "整合运动…撤退…".to_string(),
        });
        m
    }
}

impl Default for SpecialMonsterHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecialMonsterHandler {
    pub fn check_special_monsters(
        &self,
        left_unit_ids: &[String],
        right_unit_ids: &[String],
        winner: &Winner,
    ) -> String {
        let mut messages = Vec::new();

        for (&monster_id, info) in &self.special_monsters {
            let id_str = monster_id.to_string();
            let left_has = left_unit_ids.iter().any(|id| *id == id_str);
            let right_has = right_unit_ids.iter().any(|id| *id == id_str);

            if !left_has && !right_has {
                continue;
            }

            match winner {
                Winner::Left if left_has && !info.win_message.is_empty() => {
                    messages.push(info.win_message.clone());
                }
                Winner::Right if right_has && !info.win_message.is_empty() => {
                    messages.push(info.win_message.clone());
                }
                Winner::Right if left_has && !info.lose_message.is_empty() => {
                    messages.push(info.lose_message.clone());
                }
                Winner::Left if right_has && !info.lose_message.is_empty() => {
                    messages.push(info.lose_message.clone());
                }
                _ => {}
            }
        }

        messages.join("\n")
    }
}
