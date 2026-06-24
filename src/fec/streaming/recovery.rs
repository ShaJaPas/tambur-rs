//! Which data/parity stripes can be recovered (`streaming_code_helper.*`).
#![allow(clippy::too_many_arguments)]

use alloc::borrow::Cow;
use alloc::vec::Vec;

use super::flow_decode::{FlowDecodeScratch, get_used_parity_counts};
use super::reorder::reorder_to_matrix_positions;
use super::theoretical::StreamingCodeTheoretical;
use crate::fec::num_frames_for_delay;
use crate::fec::{CodingMatrixInfo, Position, zero_by_frame};

pub(crate) type VuRatio = (u16, u16);

struct ReorderedTheoretical {
    used: Vec<u16>,
    unusable: Vec<bool>,
    recovered_us: Vec<bool>,
    recovered_vs: Vec<bool>,
}

#[derive(Debug)]
struct RetainedWindowScratch {
    lost_u: Vec<u16>,
    lost_v: Vec<u16>,
    received: Vec<u16>,
}

#[derive(Debug)]
pub(crate) struct StreamingCodeHelper {
    coding_matrix_info: CodingMatrixInfo,
    delay: u16,
    positions_of_us: Vec<bool>,
    flow_scratch: FlowDecodeScratch,
    window_scratch: RetainedWindowScratch,
}

impl Clone for StreamingCodeHelper {
    fn clone(&self) -> Self {
        Self {
            coding_matrix_info: self.coding_matrix_info,
            delay: self.delay,
            positions_of_us: self.positions_of_us.clone(),
            flow_scratch: FlowDecodeScratch::default(),
            window_scratch: RetainedWindowScratch {
                lost_u: Vec::new(),
                lost_v: Vec::new(),
                received: Vec::new(),
            },
        }
    }
}

impl StreamingCodeHelper {
    pub(crate) fn new(
        coding_matrix_info: CodingMatrixInfo,
        delay: u16,
        v_to_u_ratio: VuRatio,
    ) -> Self {
        let mut helper = Self {
            coding_matrix_info,
            delay,
            positions_of_us: vec![false; coding_matrix_info.n_cols as usize],
            flow_scratch: FlowDecodeScratch::default(),
            window_scratch: RetainedWindowScratch {
                lost_u: Vec::new(),
                lost_v: Vec::new(),
                received: Vec::new(),
            },
        };
        helper.set_vs_and_us(v_to_u_ratio);
        helper
    }

    pub(crate) fn zero_by_frame_u(&self, position: Position) -> bool {
        if zero_by_frame(position, self.coding_matrix_info, self.delay) {
            return true;
        }
        if !self.positions_of_us[position.col as usize] {
            return false;
        }
        let parity_frame = self
            .coding_matrix_info
            .frame_of_row(position.row, self.delay)
            .expect("valid row");
        let data_frame = self
            .coding_matrix_info
            .frame_of_col(position.col, self.delay)
            .expect("valid col");
        let num_frames = num_frames_for_delay(self.delay);
        parity_frame != data_frame && parity_frame != ((self.delay + data_frame) % num_frames)
    }

    pub(crate) fn recoverable_data(
        &mut self,
        erased: &[bool],
        timeslot: u16,
        check_recover: bool,
    ) -> (Vec<bool>, Vec<bool>) {
        let num_frames = num_frames_for_delay(self.delay);
        let lost_us = self.lost_data_count(true, erased);
        let lost_vs = self.lost_data_count(false, erased);
        let received = self.received_parities_count(erased);
        let mut theoretical = StreamingCodeTheoretical::new(self.delay);
        theoretical.add_frames(&lost_us, &lost_vs, &received, timeslot);
        let u_rec = theoretical.frame_u_recovered();
        let v_rec = theoretical.frame_v_recovered();
        self.get_retained_info(
            num_frames,
            erased,
            timeslot,
            &lost_us,
            &lost_vs,
            &received,
            &u_rec,
            &v_rec,
            theoretical.frame_unusable_parities(),
            check_recover,
        )
    }

    #[cfg(any(test, feature = "bench"))]
    #[cfg_attr(all(feature = "bench", not(test)), allow(dead_code))]
    pub(crate) fn positions_of_us(&self) -> &[bool] {
        &self.positions_of_us
    }

    fn set_vs_and_us(&mut self, v_to_u_ratio: VuRatio) {
        self.positions_of_us.fill(false);
        let num_frames = num_frames_for_delay(self.delay);
        for frame_num in 0..num_frames {
            let cols = self.coding_matrix_info.cols_of_frame(frame_num, self.delay);
            for col in cols.0..=cols.1 {
                let offset = col - cols.0;
                if offset % (v_to_u_ratio.0 + v_to_u_ratio.1) >= v_to_u_ratio.0 {
                    self.positions_of_us[col as usize] = true;
                }
            }
        }
    }

    fn lost_data_count(&self, is_u: bool, erased: &[bool]) -> Vec<u16> {
        let num_frames = num_frames_for_delay(self.delay);
        let mut lost = vec![0u16; num_frames as usize];
        for frame in 0..num_frames {
            let cols = self.coding_matrix_info.cols_of_frame(frame, self.delay);
            for col in cols.0..=cols.1 {
                if erased[col as usize] && self.positions_of_us[col as usize] == is_u {
                    lost[frame as usize] += 1;
                }
            }
        }
        lost
    }

    fn received_parities_count(&self, erased: &[bool]) -> Vec<u16> {
        let num_frames = num_frames_for_delay(self.delay);
        let n_cols = self.coding_matrix_info.n_cols as usize;
        let mut received = vec![0u16; num_frames as usize];
        for frame in 0..num_frames {
            let rows = self.coding_matrix_info.rows_of_frame(frame, self.delay);
            for row in rows.0..=rows.1 {
                if !erased[n_cols + row as usize] {
                    received[frame as usize] += 1;
                }
            }
        }
        received
    }

    fn get_retained_parities_zero_by_memory(&self, timeslot: u16) -> Vec<bool> {
        let mut retained = vec![true; self.coding_matrix_info.n_rows as usize];
        for j in 1..=self.delay {
            let endpoints = self
                .coding_matrix_info
                .rows_of_frame(timeslot + j, self.delay);
            for row in endpoints.0..=endpoints.1 {
                retained[row as usize] = false;
            }
        }
        retained
    }

    fn get_recoverable_data_info(
        &self,
        num_frames: u16,
        erased: &[bool],
        recovered_us: &[bool],
        recovered_vs: &[bool],
    ) -> Vec<bool> {
        let mut recoverable = vec![false; self.coding_matrix_info.n_cols as usize];
        for frame in 0..num_frames {
            if recovered_us[frame as usize] || recovered_vs[frame as usize] {
                let cols = self.coding_matrix_info.cols_of_frame(frame, self.delay);
                for col in cols.0..=cols.1 {
                    let idx = col as usize;
                    let frame_ok = if self.positions_of_us[idx] {
                        recovered_us[frame as usize]
                    } else {
                        recovered_vs[frame as usize]
                    };
                    recoverable[idx] = frame_ok && erased[idx];
                }
            }
        }
        recoverable
    }

    fn get_retained_parity_info(
        &self,
        erased: &[bool],
        timeslot: u16,
        unusable_parities: &[bool],
        used_parities_counts: &mut [u16],
    ) -> Vec<bool> {
        let n_cols = self.coding_matrix_info.n_cols as usize;
        let mut retained = self.get_retained_parities_zero_by_memory(timeslot);
        for row in 0..retained.len() {
            let frame = self
                .coding_matrix_info
                .frame_of_row(row as u16, self.delay)
                .expect("valid row");
            if erased[n_cols + row]
                || unusable_parities[frame as usize]
                || used_parities_counts[frame as usize] == 0
            {
                retained[row] = false;
            } else {
                used_parities_counts[frame as usize] -= 1;
            }
        }
        retained
    }

    fn reorder_theoretical_window(
        timeslot: u16,
        num_frames: u16,
        used_unordered: &[u16],
        unusable: &[bool],
        u_rec: &[bool],
        v_rec: &[bool],
    ) -> ReorderedTheoretical {
        ReorderedTheoretical {
            used: reorder_to_matrix_positions(timeslot, used_unordered, num_frames),
            unusable: reorder_to_matrix_positions(timeslot, unusable, num_frames),
            recovered_us: reorder_to_matrix_positions(timeslot, u_rec, num_frames),
            recovered_vs: reorder_to_matrix_positions(timeslot, v_rec, num_frames),
        }
    }

    fn fill_window_scratch(
        &mut self,
        timeslot: u16,
        num_frames: u16,
        lost_u: &[u16],
        lost_v: &[u16],
        received: &[u16],
    ) {
        let scratch = &mut self.window_scratch;
        scratch.lost_u.clear();
        scratch.lost_v.clear();
        scratch.received.clear();
        scratch.lost_u.reserve(num_frames as usize);
        scratch.lost_v.reserve(num_frames as usize);
        scratch.received.reserve(num_frames as usize);
        for j in timeslot + 1..timeslot + 1 + num_frames {
            let idx = (j % num_frames) as usize;
            scratch.lost_u.push(lost_u[idx]);
            scratch.lost_v.push(lost_v[idx]);
            scratch.received.push(received[idx]);
        }
    }

    fn get_retained_info(
        &mut self,
        num_frames: u16,
        erased: &[bool],
        timeslot: u16,
        lost_u: &[u16],
        lost_v: &[u16],
        received: &[u16],
        u_rec: &[bool],
        v_rec: &[bool],
        unusable: &[bool],
        check_recover: bool,
    ) -> (Vec<bool>, Vec<bool>) {
        self.fill_window_scratch(timeslot, num_frames, lost_u, lost_v, received);
        let used_unordered: Cow<'_, [u16]> = if check_recover {
            Cow::Owned(self.window_scratch.received.clone())
        } else {
            Cow::Owned(get_used_parity_counts(
                &self.window_scratch.lost_u,
                &self.window_scratch.lost_v,
                &self.window_scratch.received,
                u_rec,
                v_rec,
                unusable,
                &mut self.flow_scratch,
            ))
        };

        let mut reordered = Self::reorder_theoretical_window(
            timeslot,
            num_frames,
            used_unordered.as_ref(),
            unusable,
            u_rec,
            v_rec,
        );
        let retained = self.get_retained_parity_info(
            erased,
            timeslot,
            &reordered.unusable,
            &mut reordered.used,
        );
        let recoverable = self.get_recoverable_data_info(
            num_frames,
            erased,
            &reordered.recovered_us,
            &reordered.recovered_vs,
        );
        (recoverable, retained)
    }
}

#[cfg(test)]
impl StreamingCodeHelper {
    pub(crate) fn test_lost_data_count(&self, is_u: bool, erased: &[bool]) -> Vec<u16> {
        self.lost_data_count(is_u, erased)
    }

    pub(crate) fn test_received_parities_count(&self, erased: &[bool]) -> Vec<u16> {
        self.received_parities_count(erased)
    }

    pub(crate) fn test_retained_parities_zero_by_memory(&self, timeslot: u16) -> Vec<bool> {
        self.get_retained_parities_zero_by_memory(timeslot)
    }

    pub(crate) fn test_recoverable_data_info(
        &self,
        num_frames: u16,
        erased: &[bool],
        recovered_us: &[bool],
        recovered_vs: &[bool],
    ) -> Vec<bool> {
        self.get_recoverable_data_info(num_frames, erased, recovered_us, recovered_vs)
    }

    pub(crate) fn test_retained_parity_info(
        &self,
        erased: &[bool],
        timeslot: u16,
        unusable_parities: &[bool],
        used_parities_counts: &mut [u16],
    ) -> Vec<bool> {
        self.get_retained_parity_info(erased, timeslot, unusable_parities, used_parities_counts)
    }
}

// C++ `test_streaming_code_helper.cc` tests.

#[cfg(test)]
mod tests {
    use super::StreamingCodeHelper;
    use crate::fec::CodingMatrixInfo;
    use crate::fec::num_frames_for_delay;
    use crate::fec::test_util::{GlibcRand, get_missing};
    use crate::fec::{Position, zero_by_frame};

    #[test]
    fn recoverable_data_info() {
        let mut rng = GlibcRand::new(0);
        for _trial in 0..100 {
            for n_par in 1..3u16 {
                for n_data in 1..6u16 {
                    for delay in 0..4u16 {
                        for w in [
                            crate::word_size::WordSize::W8,
                            crate::word_size::WordSize::W32,
                        ] {
                            let num_frames = num_frames_for_delay(delay);
                            let n_cols = n_data * num_frames;
                            if n_cols >= 256 {
                                continue;
                            }
                            let info =
                                CodingMatrixInfo::new(n_par * num_frames, n_cols, w).unwrap();
                            let helper = StreamingCodeHelper::new(info, delay, (1, 1));

                            let erased = get_missing(
                                n_cols as u8,
                                rng.rand_usize(n_cols as usize) as u8,
                                &mut rng,
                            );
                            let recovered_us = get_missing(
                                num_frames as u8,
                                rng.rand_usize(num_frames as usize) as u8,
                                &mut rng,
                            );
                            let recovered_vs = get_missing(
                                num_frames as u8,
                                rng.rand_usize(num_frames as usize) as u8,
                                &mut rng,
                            );

                            let mut expected = vec![false; n_cols as usize];
                            for col in 0..n_cols {
                                let frame = info.frame_of_col(col, delay).unwrap();
                                let recover_part = if helper.positions_of_us()[col as usize] {
                                    recovered_us[frame as usize]
                                } else {
                                    recovered_vs[frame as usize]
                                };
                                expected[col as usize] = recover_part && erased[col as usize];
                            }

                            let test_val = helper.test_recoverable_data_info(
                                num_frames,
                                &erased,
                                &recovered_us,
                                &recovered_vs,
                            );
                            assert_eq!(test_val.len(), expected.len());
                            for j in 0..n_cols as usize {
                                assert_eq!(test_val[j], expected[j]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_by_frame_u() {
        for n_par in 1..4u16 {
            for n_data in 1..4u16 {
                for delay in 0..4u16 {
                    for w in [
                        crate::word_size::WordSize::W8,
                        crate::word_size::WordSize::W32,
                    ] {
                        let num_frames = num_frames_for_delay(delay);
                        let n_cols = n_data * num_frames;
                        if n_cols >= 256 {
                            continue;
                        }
                        let info = CodingMatrixInfo::new(n_par * num_frames, n_cols, w).unwrap();
                        let helper = StreamingCodeHelper::new(info, delay, (1, 1));

                        for row in 0..info.n_rows {
                            for col in 0..info.n_cols {
                                let position = Position { row, col };
                                let test_val = helper.zero_by_frame_u(position);
                                if zero_by_frame(position, info, delay) {
                                    assert!(test_val);
                                } else if !helper.positions_of_us()[col as usize] {
                                    assert!(!test_val);
                                } else {
                                    let frame_row = info.frame_of_row(row, delay).unwrap();
                                    let frame_col = info.frame_of_col(col, delay).unwrap();
                                    assert_eq!(
                                        !test_val,
                                        frame_row == frame_col
                                            || frame_row % num_frames
                                                == (frame_col + delay) % num_frames
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn set_vs_and_us() {
        for n_par in 1..4u16 {
            for n_data in 2..10u16 {
                for delay in 0..7u16 {
                    for w in [
                        crate::word_size::WordSize::W8,
                        crate::word_size::WordSize::W32,
                    ] {
                        for v in 1..n_data {
                            for u in 0..n_data {
                                let num_frames = num_frames_for_delay(delay);
                                let info = CodingMatrixInfo::new(
                                    n_par * num_frames,
                                    n_data * num_frames,
                                    w,
                                )
                                .unwrap();
                                let helper = StreamingCodeHelper::new(info, delay, (v, u));

                                let mut expected = vec![false; info.n_cols as usize];
                                for frame in 0..num_frames {
                                    let cols = info.cols_of_frame(frame, delay);
                                    for col in cols.0..=cols.1 {
                                        let offset = col - cols.0;
                                        if offset % (v + u) >= v {
                                            expected[col as usize] = true;
                                        }
                                    }
                                }

                                assert_eq!(helper.positions_of_us(), expected.as_slice());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lost_data_count() {
        let mut rng = GlibcRand::new(0);
        for _trial in 0..1000 {
            for n_par in 1..4u16 {
                for n_data in 1..4u16 {
                    for delay in 0..4u16 {
                        for w in [
                            crate::word_size::WordSize::W8,
                            crate::word_size::WordSize::W32,
                        ] {
                            let num_frames = num_frames_for_delay(delay);
                            let n_cols = n_data * num_frames;
                            let n_rows = n_par * num_frames;
                            if n_cols >= 256 {
                                continue;
                            }
                            let info = CodingMatrixInfo::new(n_rows, n_cols, w).unwrap();
                            let helper = StreamingCodeHelper::new(info, delay, (1, 1));
                            let total = n_cols + n_rows;
                            let erased = get_missing(
                                total as u8,
                                rng.rand_usize(total as usize) as u8,
                                &mut rng,
                            );
                            let u_pos = helper.positions_of_us();

                            let mut lost_u = vec![0u16; num_frames as usize];
                            let mut lost_v = vec![0u16; num_frames as usize];
                            for pos in 0..n_cols {
                                let frame = info.frame_of_col(pos, delay).unwrap();
                                if erased[pos as usize] && u_pos[pos as usize] {
                                    lost_u[frame as usize] += 1;
                                }
                                if erased[pos as usize] && !u_pos[pos as usize] {
                                    lost_v[frame as usize] += 1;
                                }
                            }

                            let test_lost_u = helper.test_lost_data_count(true, &erased);
                            let test_lost_v = helper.test_lost_data_count(false, &erased);
                            for frame in 0..num_frames as usize {
                                assert_eq!(test_lost_u[frame], lost_u[frame]);
                                assert_eq!(test_lost_v[frame], lost_v[frame]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn received_parities_count() {
        let mut rng = GlibcRand::new(0);
        for n_par in 1..4u16 {
            for n_data in 1..4u16 {
                for delay in 0..4u16 {
                    for w in [
                        crate::word_size::WordSize::W8,
                        crate::word_size::WordSize::W32,
                    ] {
                        let num_frames = num_frames_for_delay(delay);
                        let n_cols = n_data * num_frames;
                        let n_rows = n_par * num_frames;
                        if n_cols >= 256 {
                            continue;
                        }
                        let info = CodingMatrixInfo::new(n_rows, n_cols, w).unwrap();
                        let helper = StreamingCodeHelper::new(info, delay, (1, 1));
                        let total = n_cols + n_rows;
                        let erased = get_missing(
                            total as u8,
                            rng.rand_usize(total as usize) as u8,
                            &mut rng,
                        );

                        let mut parities = vec![n_par; num_frames as usize];
                        for row in 0..n_rows {
                            if erased[(n_cols + row) as usize] {
                                let frame = info.frame_of_row(row, delay).unwrap();
                                parities[frame as usize] -= 1;
                            }
                        }

                        let test_parities = helper.test_received_parities_count(&erased);
                        for frame in 0..num_frames as usize {
                            assert_eq!(test_parities[frame], parities[frame]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn retained_parities_zero_by_memory() {
        for n_par in 1..4u16 {
            for n_data in 1..4u16 {
                for delay in 0..4u16 {
                    for w in [
                        crate::word_size::WordSize::W8,
                        crate::word_size::WordSize::W32,
                    ] {
                        let num_frames = num_frames_for_delay(delay);
                        let n_cols = n_data * num_frames;
                        if n_cols >= 256 {
                            continue;
                        }
                        let info = CodingMatrixInfo::new(n_par * num_frames, n_cols, w).unwrap();
                        let helper = StreamingCodeHelper::new(info, delay, (1, 1));

                        for timeslot in 0..(4 * num_frames) {
                            let test_vals = helper.test_retained_parities_zero_by_memory(timeslot);
                            let mut frames_retained = vec![true; num_frames as usize];
                            for shift in 1..=delay {
                                frames_retained[((timeslot + shift) % num_frames) as usize] = false;
                            }
                            let mut expected = vec![false; info.n_rows as usize];
                            for pos in timeslot..timeslot + num_frames {
                                let endpoints = info.rows_of_frame(pos, delay);
                                let val = frames_retained[(pos % num_frames) as usize];
                                for par in endpoints.0..=endpoints.1 {
                                    expected[par as usize] = val;
                                }
                            }
                            let total: u16 = test_vals.iter().map(|&b| b as u16).sum();
                            assert_eq!(test_vals, expected);
                            assert_eq!(total, (delay + 1) * n_par);
                        }
                    }
                }
            }
        }
    }

    fn helper_burst_full_recover(timeslot: u16, w: crate::word_size::WordSize) {
        let delay = 3u16;
        let n_data = 8u16;
        let n_par = 8u16;
        let num_frames = num_frames_for_delay(delay);
        let n_cols = n_data * num_frames;
        let n_rows = n_par * num_frames;
        let info = CodingMatrixInfo::new(n_rows, n_cols, w).unwrap();
        let mut helper = StreamingCodeHelper::new(info, delay, (1, 0));

        let mut parity = helper.test_retained_parities_zero_by_memory(timeslot);
        let mut erased = vec![false; (n_rows + n_cols) as usize];

        for t in timeslot - 3..=timeslot - 2 {
            let endpoints_col = info.cols_of_frame(t, delay);
            for i in endpoints_col.0..=endpoints_col.1 {
                erased[i as usize] = true;
            }
            let endpoints_row = info.rows_of_frame(t, delay);
            for i in endpoints_row.0..=endpoints_row.1 {
                erased[(n_cols + i) as usize] = true;
                parity[i as usize] = false;
            }
        }
        for t in timeslot - 6..=timeslot - 4 {
            let endpoints_row = info.rows_of_frame(t, delay);
            for i in endpoints_row.0..=endpoints_row.1 {
                parity[i as usize] = false;
            }
        }

        let mut data = vec![false; n_cols as usize];
        for t in [timeslot - 2, timeslot - 3] {
            let endpoints = info.cols_of_frame(t, delay);
            for i in endpoints.0..=endpoints.1 {
                data[i as usize] = true;
            }
        }

        let test_val = helper.recoverable_data(&erased, timeslot, false);
        assert_eq!(test_val.0, data);
        assert_eq!(test_val.1, parity);
    }

    #[test]
    fn recoverable_data_burst_full_recover() {
        for timeslot in 6..70u16 {
            for w in [
                crate::word_size::WordSize::W8,
                crate::word_size::WordSize::W32,
            ] {
                helper_burst_full_recover(timeslot, w);
            }
        }
    }

    fn helper_burst_partial_recover(w: crate::word_size::WordSize) {
        let delay = 3u16;
        let n_data = 8u16;
        let n_par = 6u16;
        let num_frames = num_frames_for_delay(delay);
        let timeslot = 2 * num_frames - 1;
        let n_cols = n_data * num_frames;
        let n_rows = n_par * num_frames;
        let info = CodingMatrixInfo::new(n_rows, n_cols, w).unwrap();
        let mut helper = StreamingCodeHelper::new(info, delay, (1, 1));

        let mut parity = helper.test_retained_parities_zero_by_memory(timeslot);
        let mut erased = vec![false; (n_rows + n_cols) as usize];

        for t in timeslot - 3..=timeslot - 2 {
            let endpoints_col = info.cols_of_frame(t, delay);
            for i in endpoints_col.0..=endpoints_col.1 {
                erased[i as usize] = true;
            }
            let endpoints_row = info.rows_of_frame(t, delay);
            for i in endpoints_row.0..=endpoints_row.1 {
                erased[(n_cols + i) as usize] = true;
                parity[i as usize] = false;
            }
        }

        let mut data = vec![false; n_cols as usize];
        let endpoints = info.cols_of_frame(timeslot - 3, delay);
        for i in endpoints.0..=endpoints.1 {
            data[i as usize] = true;
        }
        let endpoints = info.cols_of_frame(timeslot - 2, delay);
        for i in endpoints.0..=endpoints.1 {
            if !helper.positions_of_us()[i as usize] {
                data[i as usize] = true;
            }
        }

        let test_val = helper.recoverable_data(&erased, timeslot, false);
        assert_eq!(test_val.0, data);
        assert_eq!(test_val.1, parity);
    }

    #[test]
    fn recoverable_data_burst_partial_recover() {
        for _timeslot in 6..70u16 {
            helper_burst_partial_recover(crate::word_size::WordSize::W8);
        }
    }

    fn helper_burst_v_recover(w: crate::word_size::WordSize) {
        let delay = 3u16;
        let n_data = 8u16;
        let n_par = 8u16;
        let num_frames = num_frames_for_delay(delay);
        let timeslot = 2 * num_frames - 1;
        let n_cols = n_data * num_frames;
        let n_rows = n_par * num_frames;
        let info = CodingMatrixInfo::new(n_rows, n_cols, w).unwrap();
        let mut helper = StreamingCodeHelper::new(info, delay, (1, 1));

        let mut parity = helper.test_retained_parities_zero_by_memory(timeslot);
        let mut erased = vec![false; (n_rows + n_cols) as usize];

        for t in timeslot - 2..=timeslot - 1 {
            let endpoints_col = info.cols_of_frame(t, delay);
            for i in endpoints_col.0..=endpoints_col.1 {
                erased[i as usize] = true;
            }
            let endpoints_row = info.rows_of_frame(t, delay);
            for i in endpoints_row.0..=endpoints_row.1 {
                erased[(n_cols + i) as usize] = true;
                parity[i as usize] = false;
            }
        }
        for t in 0..num_frames {
            if t != timeslot % num_frames {
                let endpoints_row = info.rows_of_frame(t, delay);
                for i in endpoints_row.0..=endpoints_row.1 {
                    parity[i as usize] = false;
                }
            }
        }

        let mut data = vec![false; n_cols as usize];
        for t in [timeslot - 2, timeslot - 1] {
            let endpoints = info.cols_of_frame(t, delay);
            for i in endpoints.0..=endpoints.1 {
                if !helper.positions_of_us()[i as usize] {
                    data[i as usize] = true;
                }
            }
        }

        let test_val = helper.recoverable_data(&erased, timeslot, false);
        assert_eq!(test_val.0, data);
        assert_eq!(test_val.1, parity);
    }

    #[test]
    fn recoverable_data_burst_v_recover() {
        for _timeslot in 6..70u16 {
            helper_burst_v_recover(crate::word_size::WordSize::W8);
        }
    }

    fn helper_burst_not_recovered(w: crate::word_size::WordSize) {
        let delay = 3u16;
        let n_data = 8u16;
        let n_par = 4u16;
        let num_frames = num_frames_for_delay(delay);
        let timeslot = 2 * num_frames - 1;
        let n_cols = n_data * num_frames;
        let n_rows = n_par * num_frames;
        let info = CodingMatrixInfo::new(n_rows, n_cols, w).unwrap();
        let mut helper = StreamingCodeHelper::new(info, delay, (1, 1));

        let parity = vec![false; n_rows as usize];
        let mut erased = vec![false; (n_rows + n_cols) as usize];

        for t in timeslot - 3..=timeslot - 2 {
            let endpoints_col = info.cols_of_frame(t, delay);
            for i in endpoints_col.0..=endpoints_col.1 {
                erased[i as usize] = true;
            }
            let endpoints_row = info.rows_of_frame(t, delay);
            for i in endpoints_row.0..=endpoints_row.1 {
                erased[(n_cols + i) as usize] = true;
            }
        }

        let data = vec![false; n_cols as usize];
        let test_val = helper.recoverable_data(&erased, timeslot, false);
        assert_eq!(test_val.0, data);
        assert_eq!(test_val.1, parity);
    }

    #[test]
    fn recoverable_data_burst_not_recovered() {
        for _timeslot in 6..70u16 {
            helper_burst_not_recovered(crate::word_size::WordSize::W8);
        }
    }

    #[test]
    fn retained_parity_info() {
        let mut rng = GlibcRand::new(0);
        for _trial in 0..1000 {
            for n_par in 1..4u16 {
                for n_data in 1..4u16 {
                    for delay in 0..4u16 {
                        for w in [
                            crate::word_size::WordSize::W8,
                            crate::word_size::WordSize::W32,
                        ] {
                            let num_frames = num_frames_for_delay(delay);
                            let n_cols = n_data * num_frames;
                            let n_rows = n_par * num_frames;
                            if n_cols + n_rows >= 256 {
                                continue;
                            }
                            let info = CodingMatrixInfo::new(n_rows, n_cols, w).unwrap();
                            let helper = StreamingCodeHelper::new(info, delay, (1, 1));

                            for timeslot in 0..(4 * num_frames) {
                                let retained_by_mem =
                                    helper.test_retained_parities_zero_by_memory(timeslot);
                                let erased = get_missing(
                                    (n_cols + n_rows) as u8,
                                    rng.rand_usize((n_cols + n_rows) as usize) as u8,
                                    &mut rng,
                                );
                                let unusable_by_frame = get_missing(
                                    num_frames as u8,
                                    rng.rand_usize(num_frames as usize) as u8,
                                    &mut rng,
                                );

                                let mut expected = vec![false; n_rows as usize];
                                for j in 0..num_frames {
                                    let endpoints = info.rows_of_frame(j, delay);
                                    for i in endpoints.0..=endpoints.1 {
                                        expected[i as usize] = !erased[(n_cols + i) as usize]
                                            && !unusable_by_frame[j as usize]
                                            && retained_by_mem[i as usize];
                                    }
                                }

                                let mut used_counts = vec![1000u16; n_rows as usize];
                                let test_val = helper.test_retained_parity_info(
                                    &erased,
                                    timeslot,
                                    &unusable_by_frame,
                                    &mut used_counts,
                                );
                                assert_eq!(test_val, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}
