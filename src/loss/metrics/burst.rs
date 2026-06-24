//! Burst and guardspace histograms from loss bitmaps.

use alloc::collections::BTreeMap;

pub(super) fn get_bursts_and_guardspaces(
    losses: &[bool],
) -> (BTreeMap<u16, u16>, BTreeMap<u16, u16>) {
    let mut bursts = BTreeMap::new();
    let mut guardspaces = BTreeMap::new();
    let mut start = 0u16;
    let mut curr_burst = false;
    let mut buffer_start = 0u16;
    for (i, &loss) in losses.iter().enumerate() {
        let i = i as u16;
        if curr_burst {
            if !loss {
                let length = i - start;
                *bursts.entry(length).or_insert(0) += 1;
                buffer_start = i;
            }
        } else if loss {
            start = i;
            let length = i - buffer_start;
            if length > 0 {
                *guardspaces.entry(length).or_insert(0) += 1;
            }
        }
        curr_burst = loss;
    }
    if !losses.is_empty() {
        if curr_burst {
            let length = losses.len() as u16 - start;
            *bursts.entry(length).or_insert(0) += 1;
        } else {
            let length = losses.len() as u16 - buffer_start;
            *guardspaces.entry(length).or_insert(0) += 1;
        }
    }
    (bursts, guardspaces)
}

pub(super) fn eliminate_singleton_bursts(burst_indices: Vec<(u16, u16)>) -> Vec<(u16, u16)> {
    burst_indices
        .into_iter()
        .filter(|&(first, second)| second > first)
        .collect()
}

pub(super) fn get_burst_indices(losses: &[bool], g_min: u16) -> Vec<(u16, u16)> {
    let length = losses.len() as u16;
    let mut burst_indices = Vec::new();
    let mut i = 0u16;
    while i < length {
        if !losses[i as usize] {
            i += 1;
        } else {
            let mut j = i;
            while j < length
                && super::multi_frame::sum_bool(
                    &losses[j as usize..=((j + g_min - 1).min(length - 1)) as usize],
                ) > 0
            {
                j += 1;
            }
            burst_indices.push((i, j.saturating_sub(1)));
            i = j;
        }
    }
    burst_indices
}
