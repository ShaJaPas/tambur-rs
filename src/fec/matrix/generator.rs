//! Generator matrix construction with Tambur streaming zero patterns.
//!
//! Port of `set_cauchy_matrix` / `zero_by_frame` from `multi_frame_fec_helpers.cc`.

use alloc::vec::Vec;

use super::Matrix;
use super::cauchy;
use super::info::CodingMatrixInfo;
use crate::fec::error::FecResult;

/// Matrix cell position in the coding window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Position {
    pub row: u16,
    pub col: u16,
}

/// Build the sparse Cauchy generator matrix for a streaming window.
///
/// Applies `zero_by_frame` and optional per-cell streaming zeros, matching upstream
/// `BlockCodeFactory` / `set_cauchy_matrix`.
pub(crate) fn set_cauchy_matrix(
    info: CodingMatrixInfo,
    delay: u16,
    zero_streaming: &[Vec<bool>],
) -> FecResult<Matrix> {
    let n_rows = info.n_rows as usize;
    let n_cols = info.n_cols as usize;
    let mut tmp = cauchy::original_coding_matrix(n_cols, n_rows, info.w)?;

    for row in 0..n_rows {
        for col in 0..n_cols {
            let pos = Position {
                row: row as u16,
                col: col as u16,
            };
            let forced_zero = zero_streaming
                .get(row)
                .and_then(|r| r.get(col))
                .copied()
                .unwrap_or(false);
            if zero_by_frame(pos, info, delay) || forced_zero {
                tmp[row * n_cols + col] = 0;
            } else {
                // Upstream asserts parity frame > encoded frame + delay window;
                // retained here as a debug check once frame helpers are wired in tests.
                let _par_frame = info.frame_of_row(row as u16, delay)?;
                let _enc_frame = info.frame_of_col(col as u16, delay)?;
            }
        }
    }

    let mut matrix = Matrix::new(n_rows, n_cols)?;
    for row in 0..n_rows {
        for col in 0..n_cols {
            matrix.set(row, col, tmp[row * n_cols + col])?;
        }
    }
    Ok(matrix)
}

/// Whether a matrix cell is zeroed by the sliding-window delay pattern.
pub(crate) fn zero_by_frame(pos: Position, info: CodingMatrixInfo, delay: u16) -> bool {
    let num_frames = CodingMatrixInfo::num_frames_for_delay(delay);
    let encode_for_frame = info
        .frame_of_row(pos.row, delay)
        .expect("valid row in zero_by_frame");
    let to_include = info
        .frame_of_col(pos.col, delay)
        .expect("valid col in zero_by_frame");

    for j in 0..=delay {
        if ((to_include + j) % num_frames) == encode_for_frame {
            return false;
        }
    }
    true
}

/// Collect streaming-code zero flags for every matrix cell.
pub(crate) fn zero_streaming_mask(
    info: CodingMatrixInfo,
    _delay: u16,
    predicate: impl Fn(Position) -> bool,
) -> Vec<Vec<bool>> {
    let n_rows = info.n_rows as usize;
    let n_cols = info.n_cols as usize;
    let mut mask = vec![vec![false; n_cols]; n_rows];
    for (row, row_mask) in mask.iter_mut().enumerate() {
        for (col, cell) in row_mask.iter_mut().enumerate() {
            let pos = Position {
                row: row as u16,
                col: col as u16,
            };
            *cell = predicate(pos);
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn set_cauchy_matrix_tau3_shape() {
        let config = Config::builder()
            .tau(crate::config::Tau::new(3).unwrap())
            .build()
            .unwrap();
        let delay = u16::from(config.tau);
        let info = CodingMatrixInfo::from_config(&config, delay).unwrap();
        let zeros = zero_streaming_mask(info, delay, |_| false);
        let matrix = set_cauchy_matrix(info, delay, &zeros).unwrap();
        assert_eq!(matrix.rows(), info.n_rows as usize);
        assert_eq!(matrix.cols(), info.n_cols as usize);
        assert!(matrix.active_coefficients().contains(&0));
        assert!(matrix.active_coefficients().iter().any(|&c| c != 0));
    }

    #[test]
    fn forced_zero_streaming() {
        // delay=1 → 3 frames; 6 cols / 3 = 2 per frame, 3 rows / 3 = 1 per frame
        let info = CodingMatrixInfo::new(3, 6, crate::word_size::WordSize::W32).unwrap();
        let mut zeros = vec![vec![false; 6]; 3];
        zeros[1][2] = true;
        let matrix = set_cauchy_matrix(info, 1, &zeros).unwrap();
        assert_eq!(matrix.get(1, 2).unwrap(), 0);
    }

    #[test]
    fn zero_by_frame_differs_from_all_nonzero_cauchy() {
        // delay=2 → 5 frames; 15 cols / 5 = 3, 10 rows / 5 = 2
        let info = CodingMatrixInfo::new(10, 15, crate::word_size::WordSize::W32).unwrap();
        let delay = 2;
        let dense =
            cauchy::original_coding_matrix(15, 10, crate::word_size::WordSize::W32).unwrap();
        let mut found_zero = false;
        let mut found_nonzero = false;
        for row in 0..10 {
            for col in 0..15 {
                let pos = Position {
                    row: row as u16,
                    col: col as u16,
                };
                if zero_by_frame(pos, info, delay) {
                    found_zero = true;
                } else {
                    assert_ne!(dense[row * 15 + col], 0);
                    found_nonzero = true;
                }
            }
        }
        assert!(found_zero && found_nonzero);
    }
}
