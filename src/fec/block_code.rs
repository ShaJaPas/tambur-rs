//! Single sliding-window FEC block (`block_code.*`).
#![cfg_attr(not(feature = "bench"), allow(unreachable_pub))]

use alloc::vec::Vec;

use super::error::{FecError, FecResult};
use super::payload_matrix::PayloadMatrix;
use super::puncture::{get_puncture, get_puncture_dim, num_frames_for_delay};
use crate::fec::{
    CodingMatrixInfo, Matrix, matrix_decode_payloads, set_cauchy_matrix, zero_streaming_mask,
};

#[cfg(not(feature = "parallel"))]
use crate::fec::matrix_encode_payloads_strided;
use crate::fec::{StreamingCodeHelper, VuRatio};
#[cfg(feature = "std")]
use crate::word_size::WordSize;

/// Reusable buffers for encode/decode (optimization §2.3, §2.5).
#[derive(Clone)]
struct BlockCodeScratch {
    erased_indices: Vec<i32>,
    erasures: Vec<i32>,
    kept_frame_parities: Vec<bool>,
    decode_mat: Matrix,
    decode_coeffs: Vec<i32>,
    punc: PayloadMatrix,
    punctured_encode: PayloadMatrix,
    encode_retained: Vec<bool>,
    cached_puncture_key: usize,
}

impl BlockCodeScratch {
    fn new(
        coding_matrix_info: &CodingMatrixInfo,
        delay: u16,
        matrix: &Matrix,
        stripe_bytes: usize,
    ) -> FecResult<Self> {
        let num_frames = num_frames_for_delay(delay) as usize;
        let max_rows = matrix.max_rows();
        let max_cols = matrix.max_cols();
        Ok(Self {
            erased_indices: Vec::new(),
            erasures: Vec::new(),
            kept_frame_parities: vec![false; num_frames],
            decode_mat: Matrix::new(max_rows, max_cols)?,
            decode_coeffs: vec![0; max_rows * max_cols],
            punc: PayloadMatrix::new(coding_matrix_info.n_rows as usize, stripe_bytes)?,
            punctured_encode: PayloadMatrix::new(coding_matrix_info.n_rows as usize, stripe_bytes)?,
            encode_retained: vec![false; num_frames],
            cached_puncture_key: usize::MAX,
        })
    }
}

#[inline]
fn retained_frames_key(frames: &[bool]) -> usize {
    frames.iter().enumerate().fold(
        0usize,
        |key, (i, &kept)| {
            if kept { key | (1usize << i) } else { key }
        },
    )
}

#[derive(Clone)]
pub struct BlockCode {
    matrix: Matrix,
    data_matrix: PayloadMatrix,
    parity_matrix: PayloadMatrix,
    helper: StreamingCodeHelper,
    coding_matrix_info: CodingMatrixInfo,
    delay: u16,
    packet_size: u16,
    erased: Vec<bool>,
    timeslot: u16,
    scratch: BlockCodeScratch,
}

impl BlockCode {
    pub(crate) fn new(
        coding_matrix_info: CodingMatrixInfo,
        delay: u16,
        v_to_u_ratio: VuRatio,
        packet_size: u16,
    ) -> FecResult<Self> {
        if !(packet_size as usize).is_multiple_of(size_of::<u64>()) {
            return Err(FecError::InvalidPacketSize);
        }

        let helper = StreamingCodeHelper::new(coding_matrix_info, delay, v_to_u_ratio);
        let zero_streaming =
            zero_streaming_mask(coding_matrix_info, delay, |pos| helper.zero_by_frame_u(pos));
        let matrix = set_cauchy_matrix(coding_matrix_info, delay, &zero_streaming)?;

        #[cfg(feature = "std")]
        if coding_matrix_info.w == WordSize::W32 {
            crate::fec::gf::table::warm_gf32_coeff_cache(matrix.coefficients());
        }

        let stripe_bytes = (packet_size * u16::from(coding_matrix_info.w)) as usize;
        let scratch = BlockCodeScratch::new(&coding_matrix_info, delay, &matrix, stripe_bytes)?;
        let mut bc = Self {
            data_matrix: PayloadMatrix::new(coding_matrix_info.n_cols as usize, stripe_bytes)?,
            parity_matrix: PayloadMatrix::new(coding_matrix_info.n_rows as usize, stripe_bytes)?,
            helper,
            coding_matrix_info,
            delay,
            packet_size,
            erased: Vec::new(),
            timeslot: 0,
            scratch,
            matrix,
        };
        bc.initialize_data();
        bc.validate_matrix(v_to_u_ratio)?;
        Ok(bc)
    }

    pub(crate) fn coding_matrix_info(&self) -> CodingMatrixInfo {
        self.coding_matrix_info
    }

    pub(crate) fn delay(&self) -> u16 {
        self.delay
    }

    pub(crate) fn packet_size(&self) -> u16 {
        self.packet_size
    }

    pub(crate) fn erased(&self) -> &[bool] {
        &self.erased
    }

    pub(crate) fn place_payload(
        &mut self,
        row: u16,
        is_parity: bool,
        vals: &[u8],
    ) -> FecResult<()> {
        let matrix = if is_parity {
            &mut self.parity_matrix
        } else {
            &mut self.data_matrix
        };
        matrix.fill_row_from_slice(row as usize, vals)?;
        let idx = row as usize
            + if is_parity {
                self.coding_matrix_info.n_cols as usize
            } else {
                0
            };
        self.erased[idx] = false;
        Ok(())
    }

    pub(crate) fn row_slice(&self, row: u16, is_parity: bool) -> FecResult<&[u8]> {
        let matrix = if is_parity {
            &self.parity_matrix
        } else {
            &self.data_matrix
        };
        matrix.row(row as usize)
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn get_row(&self, row: u16, is_parity: bool) -> FecResult<Vec<u8>> {
        Ok(self.row_slice(row, is_parity)?.to_vec())
    }

    pub(crate) fn update_timeslot(&mut self, timeslot: u16) -> FecResult<()> {
        if timeslot != self.timeslot.wrapping_add(1) && !(timeslot == 0 && self.timeslot == 0) {
            return Err(FecError::InvalidTimeslot {
                expected: self.timeslot.wrapping_add(1),
                actual: timeslot,
            });
        }
        let cols = self.coding_matrix_info.cols_of_frame(timeslot, self.delay);
        for col in cols.0..=cols.1 {
            let idx = col as usize;
            if !self.erased[idx] {
                self.data_matrix.fill_row(idx, 0)?;
                self.erased[idx] = true;
            }
        }
        let rows = self.coding_matrix_info.rows_of_frame(timeslot, self.delay);
        let n_cols = self.coding_matrix_info.n_cols as usize;
        for row in rows.0..=rows.1 {
            let idx = n_cols + row as usize;
            if !self.erased[idx] {
                self.parity_matrix.fill_row(row as usize, 0)?;
                self.erased[idx] = true;
            }
        }
        self.timeslot = timeslot;
        Ok(())
    }

    pub(crate) fn pad(&mut self, frame_num: u16, num_data_used: u16) -> FecResult<()> {
        let endpoints = self.coding_matrix_info.cols_of_frame(frame_num, self.delay);
        for pos in endpoints.0 + num_data_used..=endpoints.1 {
            self.erased[pos as usize] = false;
        }
        Ok(())
    }

    pub(crate) fn encode(&mut self) -> FecResult<()> {
        let n_cols = self.matrix.cols();
        let rows = self
            .coding_matrix_info
            .rows_of_frame(self.timeslot, self.delay);
        let num_frames = num_frames_for_delay(self.delay) as usize;
        self.scratch.encode_retained.fill(false);
        self.scratch.encode_retained[(self.timeslot as usize) % num_frames] = true;

        let dim = get_puncture_dim(
            self.delay,
            self.parity_matrix.max_rows() as u16,
            &self.scratch.encode_retained,
        ) as usize;
        self.scratch
            .punctured_encode
            .resize(dim, self.parity_matrix.max_cols())?;
        super::puncture::get_puncture_payload(
            self.coding_matrix_info,
            self.delay,
            &self.parity_matrix,
            &mut self.scratch.punctured_encode,
            &self.scratch.encode_retained,
        )?;

        let stripe_size = (self.packet_size * u16::from(self.coding_matrix_info.w)) as usize;

        #[cfg(feature = "parallel")]
        {
            let w = self.coding_matrix_info.w;
            let k = n_cols;
            let m = dim;
            crate::fec::matrix::jerasure::matrix_encode_payloads_strided_dispatch(
                w,
                k,
                m,
                &self.matrix,
                rows.0 as usize,
                &mut self.data_matrix,
                &mut self.scratch.punctured_encode,
                stripe_size,
            )?;
        }
        #[cfg(not(feature = "parallel"))]
        matrix_encode_payloads_strided(
            self.coding_matrix_info.w,
            n_cols,
            dim,
            &self.matrix,
            rows.0 as usize,
            &mut self.data_matrix,
            &mut self.scratch.punctured_encode,
            stripe_size,
        )?;

        self.copy_punctured_encode(rows.0, rows.1)?;
        Ok(())
    }

    fn copy_punctured_encode(&mut self, first_pos: u16, final_pos: u16) -> FecResult<()> {
        let n_cols = self.coding_matrix_info.n_cols as usize;
        for row in first_pos..=final_pos {
            let rel = (row - first_pos) as usize;
            self.erased[n_cols + row as usize] = false;
            let src = self.scratch.punctured_encode.row(rel)?;
            self.parity_matrix.fill_row_from_slice(row as usize, src)?;
        }
        Ok(())
    }

    pub(crate) fn can_recover(&mut self) -> bool {
        let (recoverable, _) = self
            .helper
            .recoverable_data(&self.erased, self.timeslot, true);
        recoverable.iter().any(|&r| r)
    }

    pub(crate) fn decode(&mut self) -> FecResult<Vec<bool>> {
        let (recoverable_data, retained_parities) =
            self.helper
                .recoverable_data(&self.erased, self.timeslot, false);

        self.scratch.erased_indices.clear();
        for (j, &recover) in recoverable_data.iter().enumerate() {
            if recover {
                self.scratch.erased_indices.push(j as i32);
            }
        }
        if self.scratch.erased_indices.is_empty() {
            return Ok(vec![false; recoverable_data.len()]);
        }

        let num_frames = num_frames_for_delay(self.delay) as usize;
        self.scratch.kept_frame_parities.fill(false);
        for (j, &retain) in retained_parities.iter().enumerate() {
            if retain {
                let frame = self.coding_matrix_info.frame_of_row(j as u16, self.delay)?;
                self.scratch.kept_frame_parities[frame as usize] = true;
            }
        }

        let mut ind = 0u16;
        for frame in 0..num_frames as u16 {
            if !self.scratch.kept_frame_parities[frame as usize] {
                continue;
            }
            let row_range = self.coding_matrix_info.rows_of_frame(frame, self.delay);
            for row in row_range.0..=row_range.1 {
                if !retained_parities[row as usize] {
                    self.scratch
                        .erased_indices
                        .push((recoverable_data.len() as i32) + ind as i32);
                }
                ind += 1;
            }
        }

        let dim = get_puncture_dim(
            self.delay,
            self.matrix.max_rows() as u16,
            &self.scratch.kept_frame_parities,
        ) as usize;
        if dim == 0 {
            return Ok(vec![false; recoverable_data.len()]);
        }

        let puncture_key = retained_frames_key(&self.scratch.kept_frame_parities);
        self.scratch.decode_mat.resize(dim, self.matrix.cols())?;
        if puncture_key != self.scratch.cached_puncture_key {
            get_puncture(
                self.coding_matrix_info,
                self.delay,
                &self.matrix,
                &mut self.scratch.decode_mat,
                &self.scratch.kept_frame_parities,
            )?;

            let n_cols = self.scratch.decode_mat.cols();
            let n_rows = self.scratch.decode_mat.rows();
            let coeff_len = n_rows * n_cols;
            if self.scratch.decode_coeffs.len() < coeff_len {
                self.scratch.decode_coeffs.resize(coeff_len, 0);
            }
            for row in 0..n_rows {
                for col in 0..n_cols {
                    self.scratch.decode_coeffs[row * n_cols + col] =
                        self.scratch.decode_mat.get(row, col)?;
                }
            }
            self.scratch.cached_puncture_key = puncture_key;

            // Pre-warm GF caches for inverse-matrix coefficients
            // (differ from the original Cauchy matrix coefficients)
            #[cfg(feature = "std")]
            if self.coding_matrix_info.w == WordSize::W32 {
                crate::fec::gf::table::warm_gf32_coeff_cache(
                    &self.scratch.decode_coeffs[..coeff_len],
                );
            }
        }

        self.scratch.erasures.clear();
        self.scratch
            .erasures
            .extend_from_slice(&self.scratch.erased_indices);
        self.scratch.erasures.push(-1);

        let parity_dim = get_puncture_dim(
            self.delay,
            self.parity_matrix.max_rows() as u16,
            &self.scratch.kept_frame_parities,
        ) as usize;
        self.scratch
            .punc
            .resize(parity_dim, self.parity_matrix.max_cols())?;
        super::puncture::get_puncture_payload(
            self.coding_matrix_info,
            self.delay,
            &self.parity_matrix,
            &mut self.scratch.punc,
            &self.scratch.kept_frame_parities,
        )?;

        let n_cols = self.scratch.decode_mat.cols();
        let coeff_len = dim * n_cols;
        let k = self.coding_matrix_info.n_cols as usize;
        let stripe_size = (self.packet_size * u16::from(self.coding_matrix_info.w)) as usize;
        #[cfg(feature = "parallel")]
        {
            let num_erasures = self
                .scratch
                .erasures
                .iter()
                .take_while(|&&e| e != -1)
                .count();
            if num_erasures >= 12 && stripe_size >= 4096 {
                let w = self.coding_matrix_info.w;
                if crate::fec::matrix::jerasure::matrix_decode_payloads_parallel_dispatch(
                    w,
                    k,
                    dim,
                    &self.scratch.decode_coeffs[..coeff_len],
                    false,
                    &self.scratch.erasures,
                    &mut self.data_matrix,
                    &mut self.scratch.punc,
                    stripe_size,
                )
                .is_err()
                {
                    return Ok(vec![false; recoverable_data.len()]);
                }
                for &el in &self.scratch.erasures {
                    if el == -1 {
                        break;
                    }
                    if (el as usize) < recoverable_data.len() {
                        self.erased[el as usize] = false;
                    }
                }
                return Ok(recoverable_data);
            }
        }
        if matrix_decode_payloads(
            self.coding_matrix_info.w,
            k,
            dim,
            &self.scratch.decode_coeffs[..coeff_len],
            false,
            &self.scratch.erasures,
            &mut self.data_matrix,
            &mut self.scratch.punc,
            stripe_size,
        )
        .is_err()
        {
            return Ok(vec![false; recoverable_data.len()]);
        }

        for &el in &self.scratch.erasures {
            if el == -1 {
                break;
            }
            if (el as usize) < recoverable_data.len() {
                self.erased[el as usize] = false;
            }
        }

        Ok(recoverable_data)
    }

    fn initialize_data(&mut self) {
        let n_cols = self.coding_matrix_info.n_cols as usize;
        let n_rows = self.coding_matrix_info.n_rows as usize;
        self.erased = vec![true; n_cols + n_rows];
        for pos in 0..self.coding_matrix_info.n_cols {
            let _ = self.place_payload(pos, false, &[]);
            self.erased[pos as usize] = false;
        }
        for pos in 0..self.coding_matrix_info.n_rows {
            let _ = self.place_payload(pos, true, &[]);
            self.erased[n_cols + pos as usize] = true;
        }
    }

    fn validate_matrix(&self, v_to_u_ratio: VuRatio) -> FecResult<()> {
        let num_frames = num_frames_for_delay(self.delay);
        for frame in 0..num_frames {
            let rows = self.coding_matrix_info.rows_of_frame(frame, self.delay);
            let mut zeros = vec![false; num_frames as usize];
            for z in frame + 1..=frame + self.delay {
                zeros[(z % num_frames) as usize] = true;
            }
            for row in rows.0..=rows.1 {
                for col in 0..self.coding_matrix_info.n_cols {
                    let other_frame = self.coding_matrix_info.frame_of_col(col, self.delay)?;
                    let zerod = self.matrix.get(row as usize, col as usize)? == 0;
                    let mut u_conflict = false;
                    for j in frame + self.delay + 2..frame + num_frames {
                        if (other_frame % num_frames) == (j % num_frames) {
                            u_conflict = true;
                        }
                    }
                    let per_frame = self.coding_matrix_info.num_cols_per_frame(self.delay);
                    u_conflict = u_conflict
                        && v_to_u_ratio.1 > 0
                        && ((col % per_frame) % (v_to_u_ratio.0 + v_to_u_ratio.1))
                            >= v_to_u_ratio.0;
                    let should_zero = zeros[other_frame as usize] || u_conflict;
                    if zerod != should_zero {
                        return Err(FecError::MatrixValidationFailed {
                            row,
                            col,
                            zerod,
                            should_zero,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
