use crate::core::{
    BattleSnapshot, CaptureSource, RecognizedUnit, Roster, RosterPanelState, RosterSlot, RosterSource,
    Side,
};
use crate::resources::ResourceStore;
use iced::widget::image;

/// 怪物信息（用于选择窗口显示）
#[derive(Debug, Clone)]
pub struct MonsterInfo {
    pub id: u32,
    pub name: String,
    pub thumbnail: image::Handle,
}

/// 手动阵容管理器
pub struct RosterManager {
    pub panel: RosterPanelState,
    pub available_monsters: Vec<MonsterInfo>,
}

impl RosterManager {
    pub fn new() -> Self {
        Self {
            panel: RosterPanelState::default(),
            available_monsters: Vec::new(),
        }
    }

    /// 从 ResourceStore 加载可选怪物列表
    pub fn load_monsters(&mut self, resources: &ResourceStore) {
        self.available_monsters = resources
            .templates
            .iter()
            .map(|template| {
                let thumbnail = image::Handle::from_rgba(
                    template.thumbnail.width(),
                    template.thumbnail.height(),
                    template.thumbnail.clone().into_raw(),
                );
                MonsterInfo {
                    id: template.id,
                    name: template.name.clone(),
                    thumbnail,
                }
            })
            .collect();
    }

    /// 从识别结果同步阵容
    pub fn sync_from_recognition(&mut self, units: &[RecognizedUnit]) {
        // 重置阵容为空
        self.panel.roster = Roster::default();
        self.panel.source = RosterSource::AutoRecognized;

        for unit in units {
            let slot_index = unit.slot.min(2); // 限制在0-2范围
            let slot = RosterSlot {
                monster_id: unit.unit_id.parse::<u32>().ok(),
                monster_name: Some(unit.unit_id.clone()),
                count: unit.count,
            };

            match unit.side {
                Side::Left => self.panel.roster.left[slot_index] = slot,
                Side::Right => self.panel.roster.right[slot_index] = slot,
            }
        }
    }

    /// 设置指定槽位的怪物
    pub fn set_slot(&mut self, side: Side, index: usize, monster_id: u32, name: String, count: u32) {
        let slot = RosterSlot {
            monster_id: Some(monster_id),
            monster_name: Some(name),
            count,
        };

        if index < 3 {
            match side {
                Side::Left => self.panel.roster.left[index] = slot,
                Side::Right => self.panel.roster.right[index] = slot,
            }
            self.panel.source = RosterSource::ManualInput;
        }
    }

    /// 清空指定槽位
    pub fn clear_slot(&mut self, side: Side, index: usize) {
        if index < 3 {
            let slot = RosterSlot::default();
            match side {
                Side::Left => self.panel.roster.left[index] = slot,
                Side::Right => self.panel.roster.right[index] = slot,
            }
        }
    }

    /// 从当前阵容构建 BattleSnapshot（用于预测）
    pub fn to_battle_snapshot(
        &self,
        source: &CaptureSource,
        frame_size: (u32, u32),
        roi: Option<crate::core::Roi>,
    ) -> BattleSnapshot {
        let mut units = Vec::new();

        for (idx, slot) in self.panel.roster.left.iter().enumerate() {
            if let Some(name) = &slot.monster_name {
                units.push(RecognizedUnit {
                    side: Side::Left,
                    slot: idx,
                    unit_id: name.clone(),
                    count: slot.count,
                    confidence: 1.0,
                    count_source: "手动输入".to_string(),
                    count_cached: false,
                });
            }
        }

        for (idx, slot) in self.panel.roster.right.iter().enumerate() {
            if let Some(name) = &slot.monster_name {
                units.push(RecognizedUnit {
                    side: Side::Right,
                    slot: idx,
                    unit_id: name.clone(),
                    count: slot.count,
                    confidence: 1.0,
                    count_source: "手动输入".to_string(),
                    count_cached: false,
                });
            }
        }

        BattleSnapshot {
            source: source.clone(),
            frame_size,
            roi,
            units,
            terrain_features: Vec::new(),
            terrain_name: None,
        }
    }

    /// 切换面板展开/折叠
    pub fn toggle_expanded(&mut self) {
        self.panel.expanded = !self.panel.expanded;
    }

    /// 打开怪物选择窗口
    pub fn open_monster_picker(&mut self, side: Side, index: usize) {
        self.panel.picker_target = Some((side, index));
        self.panel.monster_picker_open = true;
    }

    /// 关闭怪物选择窗口
    pub fn close_monster_picker(&mut self) {
        self.panel.monster_picker_open = false;
        self.panel.picker_target = None;
    }
}

impl Default for RosterManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_from_recognition_maps_units() {
        let mut manager = RosterManager::new();
        let units = vec![
            RecognizedUnit {
                side: Side::Left,
                slot: 0,
                unit_id: "1".to_string(),
                count: 3,
                confidence: 0.95,
                count_source: "OCR".to_string(),
                count_cached: false,
            },
            RecognizedUnit {
                side: Side::Right,
                slot: 1,
                unit_id: "5".to_string(),
                count: 2,
                confidence: 0.88,
                count_source: "OCR".to_string(),
                count_cached: false,
            },
        ];

        manager.sync_from_recognition(&units);

        assert_eq!(manager.panel.roster.left[0].monster_id, Some(1));
        assert_eq!(manager.panel.roster.left[0].count, 3);
        assert_eq!(manager.panel.roster.right[1].monster_id, Some(5));
        assert_eq!(manager.panel.roster.right[1].count, 2);
        assert_eq!(manager.panel.source, RosterSource::AutoRecognized);
    }

    #[test]
    fn set_slot_and_clear_slot() {
        let mut manager = RosterManager::new();

        manager.set_slot(Side::Left, 0, 1, "狗神".to_string(), 3);
        assert_eq!(manager.panel.roster.left[0].monster_id, Some(1));
        assert_eq!(manager.panel.roster.left[0].monster_name, Some("狗神".to_string()));
        assert_eq!(manager.panel.roster.left[0].count, 3);
        assert_eq!(manager.panel.source, RosterSource::ManualInput);

        manager.clear_slot(Side::Left, 0);
        assert!(manager.panel.roster.left[0].monster_id.is_none());
        assert_eq!(manager.panel.roster.left[0].count, 0);
    }

    #[test]
    fn to_battle_snapshot_converts_roster() {
        let mut manager = RosterManager::new();
        manager.set_slot(Side::Left, 0, 1, "狗神".to_string(), 3);
        manager.set_slot(Side::Right, 2, 5, "塔露拉".to_string(), 1);

        let source = CaptureSource::Adb("127.0.0.1:5555".to_string());
        let snapshot = manager.to_battle_snapshot(&source, (1280, 720), None);

        assert_eq!(snapshot.units.len(), 2);
        assert_eq!(snapshot.units[0].side, Side::Left);
        assert_eq!(snapshot.units[0].count, 3);
        assert_eq!(snapshot.units[1].side, Side::Right);
        assert_eq!(snapshot.units[1].count, 1);
    }

    #[test]
    fn toggle_expanded() {
        let mut manager = RosterManager::new();
        assert!(!manager.panel.expanded);
        manager.toggle_expanded();
        assert!(manager.panel.expanded);
        manager.toggle_expanded();
        assert!(!manager.panel.expanded);
    }

    #[test]
    fn open_and_close_monster_picker() {
        let mut manager = RosterManager::new();
        manager.open_monster_picker(Side::Left, 1);
        assert!(manager.panel.monster_picker_open);
        assert_eq!(manager.panel.picker_target, Some((Side::Left, 1)));

        manager.close_monster_picker();
        assert!(!manager.panel.monster_picker_open);
        assert!(manager.panel.picker_target.is_none());
    }
}
