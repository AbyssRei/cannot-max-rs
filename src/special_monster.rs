use crate::core::{SpecialMonsterInfo, Winner};
use std::collections::HashMap;

pub struct SpecialMonsterHandler {
    special_monsters: HashMap<u32, SpecialMonsterInfo>,
}

impl SpecialMonsterHandler {
    pub fn new() -> Self {
        let mut special_monsters = HashMap::new();
        special_monsters.insert(
            1,
            SpecialMonsterInfo {
                name: "狗神".to_string(),
                win_message: "全军出击，我咬死你！".to_string(),
                lose_message: "牙崩了牙崩了".to_string(),
            },
        );

        Self { special_monsters }
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
