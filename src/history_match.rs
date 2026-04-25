use ndarray::{Array1, Array2, Axis};
use std::path::Path;

const MONSTER_COUNT: usize = 61;
const FIELD_FEATURE_COUNT: usize = 12;

pub struct HistoryMatch {
    past_left: Array2<f32>,
    past_right: Array2<f32>,
    #[allow(dead_code)]
    past_left_terrain: Array2<f32>,
    #[allow(dead_code)]
    past_right_terrain: Array2<f32>,
    labels: Vec<String>,
    feat_past: Array2<f32>,
    n_history: usize,
}

impl HistoryMatch {
    pub fn new(csv_path: &Path) -> Result<Self, String> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(csv_path)
            .map_err(|e| format!("无法打开历史数据文件: {e}"))?;

        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        let mut left_terrain_rows = Vec::new();
        let mut right_terrain_rows = Vec::new();
        let mut label_rows = Vec::new();

        let total_features = (MONSTER_COUNT + FIELD_FEATURE_COUNT) * 2;

        for result in reader.records() {
            let record = result.map_err(|e| format!("读取CSV行失败: {e}"))?;
            if record.len() < total_features + 1 {
                continue;
            }

            let mut left_monster = vec![0.0f32; MONSTER_COUNT];
            let mut right_monster = vec![0.0f32; MONSTER_COUNT];
            let mut left_terrain = vec![0.0f32; FIELD_FEATURE_COUNT];
            let mut right_terrain = vec![0.0f32; FIELD_FEATURE_COUNT];

            for i in 0..MONSTER_COUNT {
                left_monster[i] = record.get(i).and_then(|v| v.parse().ok()).unwrap_or(0.0);
            }
            for i in 0..FIELD_FEATURE_COUNT {
                left_terrain[i] = record
                    .get(MONSTER_COUNT + i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
            }
            for i in 0..MONSTER_COUNT {
                right_monster[i] = record
                    .get(MONSTER_COUNT + FIELD_FEATURE_COUNT + i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
            }
            for i in 0..FIELD_FEATURE_COUNT {
                right_terrain[i] = record
                    .get(MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT + i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
            }

            let label = record
                .get(total_features)
                .unwrap_or("L")
                .to_string();

            left_rows.push(left_monster);
            right_rows.push(right_monster);
            left_terrain_rows.push(left_terrain);
            right_terrain_rows.push(right_terrain);
            label_rows.push(label);
        }

        let n = left_rows.len();
        if n == 0 {
            return Ok(Self {
                past_left: Array2::zeros((0, MONSTER_COUNT)),
                past_right: Array2::zeros((0, MONSTER_COUNT)),
                past_left_terrain: Array2::zeros((0, FIELD_FEATURE_COUNT)),
                past_right_terrain: Array2::zeros((0, FIELD_FEATURE_COUNT)),
                labels: Vec::new(),
                feat_past: Array2::zeros((0, MONSTER_COUNT * 2)),
                n_history: 0,
            });
        }

        let past_left = rows_to_array2(&left_rows, MONSTER_COUNT);
        let past_right = rows_to_array2(&right_rows, MONSTER_COUNT);
        let past_left_terrain = rows_to_array2(&left_terrain_rows, FIELD_FEATURE_COUNT);
        let past_right_terrain = rows_to_array2(&right_terrain_rows, FIELD_FEATURE_COUNT);

        // Build feature: [left+right, |left-right|]
        let sum = &past_left + &past_right;
        let diff = &past_left - &past_right;
        let abs_diff = diff.mapv(|v| v.abs());
        let feat_past = ndarray::concatenate![Axis(1), sum, abs_diff];

        Ok(Self {
            past_left,
            past_right,
            past_left_terrain,
            past_right_terrain,
            labels: label_rows,
            feat_past,
            n_history: n,
        })
    }

    pub fn render_similar_matches(
        &self,
        left_counts: &[f32],
        right_counts: &[f32],
    ) -> (Vec<usize>, f32, f32) {
        if self.n_history == 0 {
            return (Vec::new(), 0.0, 0.0);
        }

        let cur_left = Array1::from_vec(left_counts.to_vec());
        let cur_right = Array1::from_vec(right_counts.to_vec());

        // Current feature
        let cur_sum = &cur_left + &cur_right;
        let cur_diff = &cur_left - &cur_right;
        let cur_abs_diff = cur_diff.mapv(|v| v.abs());
        let feat_cur = ndarray::concatenate![Axis(0), cur_sum, cur_abs_diff];

        // Cosine similarity
        let sims = cosine_similarity(&feat_cur, &self.feat_past);

        // Presence vectors
        let pres_l = cur_left.mapv(|v| v > 0.0);
        let pres_r = cur_right.mapv(|v| v > 0.0);
        let _need_l_idx: Vec<usize> = pres_l.iter().enumerate().filter(|(_, v)| **v).map(|(i, _)| i).collect();
        let _need_r_idx: Vec<usize> = pres_r.iter().enumerate().filter(|(_, v)| **v).map(|(i, _)| i).collect();

        let hist_pres_l = self.past_left.mapv(|v| v > 0.0);
        let hist_pres_r = self.past_right.mapv(|v| v > 0.0);

        // Compute miss and cnt for non-mirrored and mirrored
        let mut miss_a: Array1<f32> = Array1::zeros(self.n_history);
        let mut cnt_a: Array1<f32> = Array1::zeros(self.n_history);
        let mut miss_b: Array1<f32> = Array1::zeros(self.n_history);
        let mut cnt_b: Array1<f32> = Array1::zeros(self.n_history);

        for i in 0..self.n_history {
            for j in 0..MONSTER_COUNT {
                let pl = hist_pres_l[[i, j]];
                let pr = hist_pres_r[[i, j]];
                let cl = pres_l[j];
                let cr = pres_r[j];

                // XOR for miss
                miss_a[i] += ((pl != cl) as usize + (pr != cr) as usize) as f32;
                miss_b[i] += ((pr != cl) as usize + (pl != cr) as usize) as f32;

                // Abs diff for cnt
                cnt_a[i] += (self.past_left[[i, j]] - cur_left[j]).abs()
                    + (self.past_right[[i, j]] - cur_right[j]).abs();
                cnt_b[i] += (self.past_right[[i, j]] - cur_left[j]).abs()
                    + (self.past_left[[i, j]] - cur_right[j]).abs();
            }
        }

        // Determine swap
        let swap: Array1<bool> = miss_b
            .iter()
            .zip(miss_a.iter())
            .zip(cnt_b.iter())
            .zip(cnt_a.iter())
            .map(|(((&mb, &ma), &cb), &ca)| mb < ma || (mb == ma && cb < ca))
            .collect();

        // Sort by similarity (descending)
        let mut indices: Vec<usize> = (0..self.n_history).collect();
        indices.sort_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap_or(std::cmp::Ordering::Equal));

        let top20: Vec<usize> = indices.into_iter().take(20).collect();

        // Compute win rates from top 5
        let top5 = &top20[..5.min(top20.len())];
        let mut left_wins = 0usize;
        let mut right_wins = 0usize;

        for &idx in top5 {
            let label = &self.labels[idx];
            let effective_label = if swap[idx] {
                if label == "L" { "R" } else { "L" }
            } else {
                label.as_str()
            };
            if effective_label == "L" {
                left_wins += 1;
            } else {
                right_wins += 1;
            }
        }

        let n_top5 = top5.len().max(1) as f32;
        let left_rate = left_wins as f32 / n_top5;
        let right_rate = right_wins as f32 / n_top5;

        (top20, left_rate, right_rate)
    }

    pub fn len(&self) -> usize {
        self.n_history
    }
}

fn rows_to_array2(rows: &[Vec<f32>], cols: usize) -> Array2<f32> {
    let n = rows.len();
    let mut data = Vec::with_capacity(n * cols);
    for row in rows {
        data.extend_from_slice(row);
    }
    Array2::from_shape_vec((n, cols), data).unwrap_or_else(|_| Array2::zeros((n, cols)))
}

fn cosine_similarity(query: &Array1<f32>, matrix: &Array2<f32>) -> Array1<f32> {
    let n = matrix.nrows();
    let mut sims = Array1::zeros(n);

    let q_norm: f32 = query.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if q_norm < 1e-8 {
        return sims;
    }

    for i in 0..n {
        let row = matrix.row(i);
        let dot: f32 = query.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
        let r_norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if r_norm > 1e-8 {
            sims[i] = dot / (q_norm * r_norm);
        }
    }

    sims
}
