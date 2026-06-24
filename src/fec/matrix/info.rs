//! Dimensions of the Jerasure coding matrix for one streaming window.
//!
//! Port of Tambur `coding_matrix_info.hh` and inline helpers from
//! `multi_frame_fec_helpers.hh`.

use crate::word_size::WordSize;

use crate::fec::error::{FecError, FecResult};

/// Size and Galois-field word width of the generator matrix passed to Jerasure.
///
/// This struct holds three numbers:
///
/// - `n_cols` — number of **data** stripes in the coding window (columns)
/// - `n_rows` — number of **parity** stripes (rows)
/// - `w` — GF word size in bits (Jerasure `w`, typically 32)
///
/// Together they describe the matrix that [`super::Matrix`] holds before
/// `jerasure_matrix_encode` / `jerasure_matrix_decode`. The streaming code
/// fills this matrix with Cauchy coefficients and forced zeros; it does not
/// encode payload by itself.
///
/// Window layout helpers (`frame_of_row`, `cols_of_frame`, …) assume the matrix
/// is split evenly across `2 * delay + 1` source frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CodingMatrixInfo {
    pub n_rows: u16,
    pub n_cols: u16,
    pub w: WordSize,
}

/// Inclusive `(start, end)` index range in a matrix row or column dimension.
pub(crate) type IndexRange = (u16, u16);

impl CodingMatrixInfo {
    pub(crate) fn new(n_rows: u16, n_cols: u16, w: WordSize) -> FecResult<Self> {
        if n_rows == 0 || n_cols == 0 {
            return Err(FecError::InvalidMatrixDimensions);
        }
        Ok(Self { n_rows, n_cols, w })
    }

    /// Build dimensions from [`Config`] and latency deadline `delay` (usually `config.tau`).
    ///
    /// Uses the same stripe caps as upstream Tambur (`max_fec_stripes` / `max_data_stripes`
    /// per frame × frames in the window).
    #[cfg(test)]
    pub(crate) fn from_config(config: &crate::config::Config, delay: u16) -> FecResult<Self> {
        let frames = num_frames_for_delay(delay);
        let n_cols = config
            .max_data_stripes
            .get()
            .checked_mul(frames)
            .ok_or(FecError::InvalidMatrixDimensions)?;
        let n_rows = config
            .max_fec_stripes
            .checked_mul(frames)
            .ok_or(FecError::InvalidMatrixDimensions)?;
        Self::new(n_rows, n_cols, config.w)
    }

    /// Number of source frames in the sliding window: `2 * delay + 1`.
    pub(crate) fn num_frames_for_delay(delay: u16) -> u16 {
        num_frames_for_delay(delay)
    }

    pub(crate) fn num_cols_per_frame(&self, delay: u16) -> u16 {
        self.n_cols / num_frames_for_delay(delay)
    }

    pub(crate) fn num_rows_per_frame(&self, delay: u16) -> u16 {
        self.n_rows / num_frames_for_delay(delay)
    }

    pub(crate) fn rows_of_frame(&self, frame_num: u16, delay: u16) -> IndexRange {
        let frame_position = frame_num % num_frames_for_delay(delay);
        let per_frame = self.num_rows_per_frame(delay);
        let start = frame_position * per_frame;
        (start, start + per_frame - 1)
    }

    pub(crate) fn cols_of_frame(&self, frame_num: u16, delay: u16) -> IndexRange {
        let frame_position = frame_num % num_frames_for_delay(delay);
        let per_frame = self.num_cols_per_frame(delay);
        let start = frame_position * per_frame;
        (start, start + per_frame - 1)
    }

    pub(crate) fn frame_of_row(&self, row: u16, delay: u16) -> FecResult<u16> {
        if row >= self.n_rows {
            return Err(FecError::InvalidMatrixRow { row });
        }
        let frames = num_frames_for_delay(delay);
        if frames == 1 {
            return Ok(0);
        }
        for frame_num in 0..frames {
            let (start, end) = self.rows_of_frame(frame_num, delay);
            if row >= start && row <= end {
                return Ok(frame_num);
            }
        }
        Err(FecError::InvalidMatrixRow { row })
    }

    pub(crate) fn frame_of_col(&self, col: u16, delay: u16) -> FecResult<u16> {
        if col >= self.n_cols {
            return Err(FecError::InvalidMatrixColumn { col });
        }
        let frames = num_frames_for_delay(delay);
        for frame_num in 0..frames {
            let (start, end) = self.cols_of_frame(frame_num, delay);
            if col >= start && col <= end {
                return Ok(frame_num);
            }
        }
        Err(FecError::InvalidMatrixColumn { col })
    }
}

fn num_frames_for_delay(delay: u16) -> u16 {
    2 * delay + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn from_config_tau3_defaults() {
        let config = Config::builder()
            .tau(crate::config::Tau::new(3).unwrap())
            .build()
            .unwrap();
        let info = CodingMatrixInfo::from_config(&config, u16::from(config.tau)).unwrap();
        // window = 7 frames; 64 data stripes + 32 fec stripes per frame (validated defaults)
        assert_eq!(info.n_cols, 64 * 7);
        assert_eq!(info.n_rows, 32 * 7);
        assert_eq!(info.w, WordSize::W32);
    }

    #[test]
    fn frame_index_helpers() {
        let info = CodingMatrixInfo::new(15, 48, WordSize::W32).unwrap();
        let delay = 3;
        assert_eq!(info.num_cols_per_frame(delay), 48 / 7);
        assert_eq!(info.frame_of_col(0, delay).unwrap(), 0);
        assert_eq!(info.frame_of_row(0, delay).unwrap(), 0);
    }
}
