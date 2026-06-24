//! Puncture helpers for sliding-window encode/decode (`puncture.hh` / `.cc`).

use alloc::vec::Vec;

use crate::fec::CodingMatrixInfo;
use crate::fec::Matrix;

pub(super) type RowRange = (u16, u16);

pub(crate) fn num_frames_for_delay(delay: u16) -> u16 {
    2 * delay + 1
}

pub(super) fn get_puncture_dim(delay: u16, num_rows: u16, retained_frames: &[bool]) -> u16 {
    let num_frames = num_frames_for_delay(delay);
    let num_frames_used = retained_frames.iter().filter(|&&u| u).count() as u16;
    (num_rows / num_frames) * num_frames_used
}

pub(super) fn get_kept_rows(
    info: CodingMatrixInfo,
    delay: u16,
    retained_frames: &[bool],
    n_rows: u16,
) -> Vec<RowRange> {
    let num_frames = num_frames_for_delay(delay);
    if n_rows == 0 {
        return Vec::new();
    }
    let mut kept = Vec::new();
    for frame in 0..num_frames {
        if retained_frames
            .get(frame as usize)
            .copied()
            .unwrap_or(false)
        {
            let (first, second) = info.rows_of_frame(frame, delay);
            kept.push((first, second));
        }
    }
    kept
}

pub(super) fn set_decode_matrix(
    kept_rows: &[RowRange],
    decode_matrix: &Matrix,
    new_matrix: &mut Matrix,
) -> Result<(), super::error::FecError> {
    let mut curr_row = 0usize;
    for &(start, end) in kept_rows {
        for row in start..=end {
            for col in 0..decode_matrix.cols() {
                new_matrix.set(curr_row, col, decode_matrix.get(row as usize, col)?)?;
            }
            curr_row += 1;
        }
    }
    Ok(())
}

pub(super) fn get_puncture(
    info: CodingMatrixInfo,
    delay: u16,
    decode_matrix: &Matrix,
    new_matrix: &mut Matrix,
    retained_frames: &[bool],
) -> Result<(), super::error::FecError> {
    let kept = get_kept_rows(info, delay, retained_frames, new_matrix.max_rows() as u16);
    set_decode_matrix(&kept, decode_matrix, new_matrix)
}

pub(super) fn get_puncture_payload(
    info: CodingMatrixInfo,
    delay: u16,
    decode_matrix: &super::payload_matrix::PayloadMatrix,
    new_matrix: &mut super::payload_matrix::PayloadMatrix,
    retained_frames: &[bool],
) -> Result<(), super::error::FecError> {
    let kept = get_kept_rows(info, delay, retained_frames, new_matrix.max_rows() as u16);
    set_decode_payload(&kept, decode_matrix, new_matrix)
}

fn set_decode_payload(
    kept_rows: &[RowRange],
    decode_matrix: &super::payload_matrix::PayloadMatrix,
    new_matrix: &mut super::payload_matrix::PayloadMatrix,
) -> Result<(), super::error::FecError> {
    let mut curr_row = 0usize;
    for &(start, end) in kept_rows {
        for row in start..=end {
            for col in 0..decode_matrix.cols() {
                new_matrix.set(curr_row, col, decode_matrix.get(row as usize, col)?)?;
            }
            curr_row += 1;
        }
    }
    Ok(())
}
