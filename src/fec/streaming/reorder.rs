//! Reordering helpers (`wrap_helpers.hh`).

use alloc::vec::Vec;

/// Map a timeslot-ordered vector to matrix frame positions `0..num_frames`.
pub(super) fn reorder_to_matrix_positions<T: Copy>(
    timeslot: u16,
    unordered: &[T],
    num_frames: u16,
) -> Vec<T> {
    let mut result = vec![unordered[0]; num_frames as usize];
    for (j, &val) in unordered.iter().enumerate().take(num_frames as usize) {
        let frame_idx = ((timeslot as usize + 1 + j) % num_frames as usize) as u16;
        result[frame_idx as usize] = val;
    }
    result
}
