use crate::error::AppResult;
use crate::types::{GridPosition, MonsterDef, SimResult, Winner};

pub trait IBattleSimulator: Send + Sync {
    fn initialize(&mut self, left: &[MonsterDef], right: &[MonsterDef]) -> AppResult<()>;
    fn step(&mut self) -> AppResult<SimState>;
    fn run_until_end(&mut self, max_rounds: u32) -> AppResult<SimResult>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimState {
    Running,
    LeftWin,
    RightWin,
    Draw,
}

#[derive(Debug, Clone)]
pub struct SimMonster {
    pub name: String,
    pub side: u8,
    pub hp: f64,
    pub max_hp: f64,
    pub attack: f64,
    pub defense: f64,
    pub magic_resist: f64,
    pub attack_interval: f64,
    pub attack_range: f64,
    pub move_speed: f64,
    pub attack_type: String,
    pub position: GridPosition,
    pub alive: bool,
    pub attack_timer: f64,
    pub special: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Projectile {
    pub from_idx: usize,
    pub target_idx: usize,
    pub damage: f64,
    pub progress: f64,
    pub speed: f64,
}

pub struct BattleSimulator {
    monsters: Vec<SimMonster>,
    projectiles: Vec<Projectile>,
    grid_width: usize,
    grid_height: usize,
    round: u32,
}

impl BattleSimulator {
    pub fn new() -> Self {
        Self {
            monsters: Vec::new(),
            projectiles: Vec::new(),
            grid_width: 13,
            grid_height: 9,
            round: 0,
        }
    }

    fn place_monsters(&mut self, left: &[MonsterDef], right: &[MonsterDef]) {
        self.monsters.clear();
        for (i, def) in left.iter().enumerate() {
            let row = i % self.grid_height;
            self.monsters.push(SimMonster {
                name: def.name.clone(),
                side: 0,
                hp: def.hp,
                max_hp: def.hp,
                attack: def.attack,
                defense: def.defense,
                magic_resist: def.magic_resist,
                attack_interval: def.attack_interval,
                attack_range: def.attack_range,
                move_speed: def.move_speed,
                attack_type: def.attack_type.clone(),
                position: GridPosition { col: 0, row },
                alive: true,
                attack_timer: 0.0,
                special: def.special.clone(),
            });
        }
        for (i, def) in right.iter().enumerate() {
            let row = i % self.grid_height;
            self.monsters.push(SimMonster {
                name: def.name.clone(),
                side: 1,
                hp: def.hp,
                max_hp: def.hp,
                attack: def.attack,
                defense: def.defense,
                magic_resist: def.magic_resist,
                attack_interval: def.attack_interval,
                attack_range: def.attack_range,
                move_speed: def.move_speed,
                attack_type: def.attack_type.clone(),
                position: GridPosition {
                    col: self.grid_width - 1,
                    row,
                },
                alive: true,
                attack_timer: 0.0,
                special: def.special.clone(),
            });
        }
    }

    fn move_monsters(&mut self) {
        for monster in &mut self.monsters {
            if !monster.alive {
                continue;
            }
            let direction: i32 = if monster.side == 0 { 1 } else { -1 };
            let new_col = monster.position.col as i32 + direction;
            if new_col >= 0 && (new_col as usize) < self.grid_width {
                monster.position.col = new_col as usize;
            }
        }
    }

    fn process_attacks(&mut self) {
        let n = self.monsters.len();
        let mut attacks: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            if !self.monsters[i].alive {
                continue;
            }
            self.monsters[i].attack_timer += 1.0;
            if self.monsters[i].attack_timer < self.monsters[i].attack_interval {
                continue;
            }
            let attacker_side = self.monsters[i].side;
            let attacker_col = self.monsters[i].position.col;
            let attacker_row = self.monsters[i].position.row;
            let attacker_range = self.monsters[i].attack_range;
            let attacker_attack = self.monsters[i].attack;
            let mut best_target: Option<usize> = None;
            let mut best_dist = f64::MAX;
            for j in 0..n {
                if !self.monsters[j].alive || self.monsters[j].side == attacker_side {
                    continue;
                }
                let dx = (self.monsters[j].position.col as f64 - attacker_col as f64).abs();
                let dy = (self.monsters[j].position.row as f64 - attacker_row as f64).abs();
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= attacker_range && dist < best_dist {
                    best_dist = dist;
                    best_target = Some(j);
                }
            }
            if let Some(target_idx) = best_target {
                attacks.push((i, target_idx, attacker_attack));
            }
        }
        for (attacker_idx, target_idx, damage) in attacks {
            self.monsters[attacker_idx].attack_timer = 0.0;
            let attack_type = self.monsters[attacker_idx].attack_type.clone();
            let target = &mut self.monsters[target_idx];
            let actual_damage = if attack_type == "物理" {
                (damage - target.defense).max(0.0)
            } else {
                (damage - target.magic_resist).max(0.0)
            };
            target.hp -= actual_damage;
            if target.hp <= 0.0 {
                target.alive = false;
            }
        }
    }

    fn check_winner(&self) -> SimState {
        let left_alive = self.monsters.iter().any(|m| m.alive && m.side == 0);
        let right_alive = self.monsters.iter().any(|m| m.alive && m.side == 1);
        match (left_alive, right_alive) {
            (true, true) => SimState::Running,
            (true, false) => SimState::LeftWin,
            (false, true) => SimState::RightWin,
            (false, false) => SimState::Draw,
        }
    }
}

impl IBattleSimulator for BattleSimulator {
    fn initialize(&mut self, left: &[MonsterDef], right: &[MonsterDef]) -> AppResult<()> {
        self.round = 0;
        self.projectiles.clear();
        self.place_monsters(left, right);
        Ok(())
    }

    fn step(&mut self) -> AppResult<SimState> {
        self.round += 1;
        self.move_monsters();
        self.process_attacks();
        Ok(self.check_winner())
    }

    fn run_until_end(&mut self, max_rounds: u32) -> AppResult<SimResult> {
        for _ in 0..max_rounds {
            match self.step()? {
                SimState::Running => continue,
                SimState::LeftWin => {
                    return Ok(SimResult {
                        winner: Winner::Left,
                        rounds: self.round,
                    })
                }
                SimState::RightWin => {
                    return Ok(SimResult {
                        winner: Winner::Right,
                        rounds: self.round,
                    })
                }
                SimState::Draw => {
                    return Ok(SimResult {
                        winner: Winner::Draw,
                        rounds: self.round,
                    })
                }
            }
        }
        Ok(SimResult {
            winner: Winner::Unknown,
            rounds: self.round,
        })
    }
}
