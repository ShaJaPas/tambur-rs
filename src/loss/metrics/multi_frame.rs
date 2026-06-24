//! Multi-frame consecutive loss metrics.

use alloc::vec::Vec;

pub(super) fn sum_bool(slice: &[bool]) -> u32 {
    slice.iter().filter(|&&b| b).count() as u32
}

pub(super) fn consecutive_burst_info_inner(losses: &[bool], frame_indices: &[u16]) -> (u16, u16) {
    let mut num_frames = 0u16;
    let mut num_pkts = 0u16;
    let mut index = 0usize;
    let mut curr_burst = true;
    while index < frame_indices.len() && curr_burst {
        let start = frame_indices[index] as usize;
        index += 1;
        let end = frame_indices
            .get(index)
            .map(|&x| x as usize)
            .unwrap_or(losses.len());
        curr_burst = curr_burst && sum_bool(&losses[start..end]) > 0;
        if curr_burst {
            num_frames += 1;
            num_pkts += (end - start) as u16;
        }
    }
    (num_frames, num_pkts)
}

pub(super) fn get_positions_consecutive_losses_inner(
    losses: &[bool],
    frame_indices: &[u16],
    min_frames: u16,
) -> (Vec<u16>, Vec<u16>) {
    assert!(min_frames > 0);
    let mut positions = Vec::new();
    let mut lengths = Vec::new();
    let mut index_pos = 0usize;
    let num_indices = frame_indices.len();
    while index_pos + min_frames as usize <= num_indices {
        let index = frame_indices[index_pos];
        let packet_losses = &losses[index as usize..];
        let adjusted: Vec<u16> = frame_indices[index_pos..]
            .iter()
            .map(|&x| x - index)
            .collect();
        let info = consecutive_burst_info_inner(packet_losses, &adjusted);
        if info.0 >= min_frames {
            positions.push(index);
            lengths.push(info.1);
            index_pos += info.0 as usize;
        } else {
            index_pos += 1;
        }
    }
    (positions, lengths)
}

pub(super) fn compute_multi_frame_loss_fraction_inner(
    losses: &[bool],
    frame_indices: &[u16],
    min_frames: u16,
) -> f32 {
    assert!(min_frames > 0);
    let (positions, lengths) =
        get_positions_consecutive_losses_inner(losses, frame_indices, min_frames);
    if positions.is_empty() {
        return 0.0;
    }
    let mut average = 0f32;
    for (pos, num_pkts) in positions.iter().zip(lengths.iter()) {
        let lost = sum_bool(&losses[*pos as usize..*pos as usize + *num_pkts as usize]) as f32;
        average += lost / *num_pkts as f32 / positions.len() as f32;
    }
    average
}
