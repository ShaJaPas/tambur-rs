//! RFC burst/gap density metrics.

use alloc::vec::Vec;

use super::burst::{eliminate_singleton_bursts, get_burst_indices};

pub(super) fn get_rfc_densities_inner(losses: &[bool], g_min: u16) -> (f32, f32) {
    assert!(g_min > 0);
    let num_lost: f32 = losses.iter().map(|&l| u8::from(l) as f32).sum();
    if num_lost == 0.0 {
        return (0.0, 0.0);
    }
    let burst_indices = eliminate_singleton_bursts(get_burst_indices(losses, g_min));
    if burst_indices.is_empty() {
        return (0.0, num_lost / losses.len().max(1) as f32);
    }

    let burst_density = density_over_ranges(
        losses,
        burst_indices.iter().map(|&(first, second)| (first, second)),
    );
    let mut guardspace_ranges = Vec::new();
    if burst_indices[0].0 > 0 {
        guardspace_ranges.push((0, burst_indices[0].0 - 1));
    }
    let mut end_burst = burst_indices[0].1;
    for &(first, second) in burst_indices.iter().skip(1) {
        guardspace_ranges.push((end_burst + 1, first - 1));
        end_burst = second;
    }
    if end_burst + 1 < losses.len() as u16 {
        guardspace_ranges.push((end_burst + 1, losses.len() as u16 - 1));
    }
    let gap_density = density_over_ranges(losses, guardspace_ranges.into_iter());

    (burst_density, gap_density)
}

fn density_over_ranges(losses: &[bool], ranges: impl Iterator<Item = (u16, u16)>) -> f32 {
    let mut density = 0f32;
    let mut total = 0f32;
    for (first, second) in ranges {
        let slice = &losses[first as usize..=second as usize];
        density += slice.iter().map(|&l| u8::from(l) as f32).sum::<f32>();
        total += slice.len() as f32;
    }
    density / total.max(1.0)
}
