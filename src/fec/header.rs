//! FEC for frame-size metadata in datagram headers (`multi_fec_header_code.*`).

use alloc::vec::Vec;

use crate::util::{get_u64, put_u64};

use super::block_code::BlockCode;
use super::error::{FecError, FecResult};
use super::puncture::num_frames_for_delay;
use crate::fec::CodingMatrixInfo;
use crate::word_size::WordSize;

/// Build header [`CodingMatrixInfo`] and max packets per frame.
pub(crate) fn header_matrix_config(delay: u16) -> FecResult<(CodingMatrixInfo, u16)> {
    let num_frames = num_frames_for_delay(delay);
    let n_cols = num_frames;
    let mut n_rows = 256u16;
    while n_rows * n_cols > 255 || !n_rows.is_multiple_of(num_frames) {
        n_rows -= 1;
    }
    let info = CodingMatrixInfo::new(n_rows, n_cols, WordSize::W8)?;
    Ok((info, n_rows / num_frames))
}

/// FEC-encodes frame sizes into `sizes_of_frames_encoding` (`MultiFECHeaderCode`).
pub(crate) struct MultiFecHeaderCode {
    delay: u16,
    num_frames: u16,
    max_pkts_per_frame: u16,
    coding_matrix_info: CodingMatrixInfo,
    block_code: BlockCode,
    timeslot: i32,
}

impl MultiFecHeaderCode {
    pub(crate) fn new(delay: u16, max_pkts_per_frame: u16) -> FecResult<Self> {
        let (info, matrix_pkts) = header_matrix_config(delay)?;
        if max_pkts_per_frame > matrix_pkts {
            return Err(FecError::InvalidStreamingCodeConfig);
        }
        let block = BlockCode::new(info, delay, (1, 0), 8)?;
        Ok(Self {
            delay,
            num_frames: num_frames_for_delay(delay),
            max_pkts_per_frame,
            coding_matrix_info: info,
            block_code: block,
            timeslot: -1,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_block_code(
        block_code: BlockCode,
        delay: u16,
        max_pkts_per_frame: u16,
    ) -> FecResult<Self> {
        let info = block_code.coding_matrix_info();
        let num_frames = num_frames_for_delay(delay);
        // C++ `MultiFECHeaderCode` requires `n_rows == num_frames *
        // max_pkts_per_frame` so that row indexing in `encode_sizes_of_frames` /
        // `decode_sizes_of_frames` (`time * max_pkts_per_frame + ...`) cannot
        // overflow the matrix. We mirror that exact equality here.
        let expected_rows = num_frames.checked_mul(max_pkts_per_frame);
        if info.w != WordSize::W8
            || block_code.packet_size() != 8
            || block_code.delay() != delay
            || info.n_cols != num_frames
            || expected_rows != Some(info.n_rows)
        {
            return Err(FecError::InvalidStreamingCodeConfig);
        }
        Ok(Self {
            delay,
            num_frames,
            max_pkts_per_frame,
            coding_matrix_info: info,
            block_code,
            timeslot: -1,
        })
    }

    pub(crate) fn encode_sizes_of_frames(
        &mut self,
        frame_size: u64,
        pkt_pos_in_frame: u16,
    ) -> FecResult<u64> {
        if pkt_pos_in_frame == 0 {
            self.timeslot += 1;
            self.block_code.update_timeslot(self.timeslot as u16)?;
            let val = put_u64(frame_size);
            self.block_code
                .place_payload((self.timeslot as u16) % self.num_frames, false, &val)?;
            self.block_code.encode()?;
            return Ok(frame_size);
        }
        let time = (self.timeslot as u16) % self.num_frames;
        let row =
            time * self.max_pkts_per_frame + ((pkt_pos_in_frame - 1) % self.max_pkts_per_frame);
        let encoding = self.block_code.row_slice(row, true)?;
        Ok(get_u64(encoding).unwrap_or(0))
    }

    pub(crate) fn decode_sizes_of_frames(
        &mut self,
        frame_num: u16,
        pos_in_frame: u16,
        size: u64,
    ) -> FecResult<Vec<(u16, u64)>> {
        self.advance_to_wire_frame(frame_num)?;
        let payload = put_u64(size);
        if pos_in_frame == 0 {
            self.block_code
                .place_payload(frame_num % self.num_frames, false, &payload)?;
        } else {
            let row = (frame_num % self.num_frames) * self.max_pkts_per_frame
                + ((pos_in_frame - 1) % self.max_pkts_per_frame);
            self.block_code.place_payload(row, true, &payload)?;
        }
        if pos_in_frame == 0 {
            if self.block_code.can_recover() {
                let mut recovered = self.recover()?;
                // recover() only returns erased-then-recovered columns; our own
                // frame was placed (not erased) so it is not included.
                if !recovered.iter().any(|&(w, _)| w == frame_num) {
                    recovered.push((frame_num, size));
                }
                return Ok(recovered);
            }
            return Ok(alloc::vec![(frame_num, size)]);
        }
        if self.block_code.can_recover() {
            return self.recover();
        }
        Ok(Vec::new())
    }

    fn advance_to_wire_frame(&mut self, frame_num: u16) -> FecResult<()> {
        if self.timeslot < 0 {
            let mut t = 0u16;
            while t != frame_num.wrapping_add(1) {
                self.block_code.update_timeslot(t)?;
                t = t.wrapping_add(1);
            }
            self.timeslot = i32::from(frame_num);
            return Ok(());
        }
        let current = self.timeslot as u16;
        if frame_num == current {
            return Ok(());
        }
        if frame_num > current
            || current.wrapping_sub(frame_num) > crate::datagram::MAX_FRAME_NUM / 2
        {
            let mut t = current.wrapping_add(1);
            while t != frame_num.wrapping_add(1) {
                self.block_code.update_timeslot(t)?;
                t = t.wrapping_add(1);
            }
            self.timeslot = i32::from(frame_num);
        }
        Ok(())
    }

    fn recover(&mut self) -> FecResult<Vec<(u16, u64)>> {
        let recovered = self.block_code.decode()?;
        let mut out = Vec::new();
        for (pos, &rec) in recovered.iter().enumerate() {
            if !rec {
                continue;
            }
            let row = self.block_code.row_slice(pos as u16, false)?;
            let size = get_u64(row).unwrap_or(0);
            let rel = self
                .coding_matrix_info
                .frame_of_col(pos as u16, self.delay)?;
            let frame_num = absolute_frame_num(self.timeslot as u16, rel, self.num_frames);
            out.push((frame_num, size));
        }
        Ok(out)
    }
}

fn absolute_frame_num(timeslot: u16, rel_frame: u16, num_frames: u16) -> u16 {
    let base = timeslot % num_frames;
    let offset = (rel_frame + num_frames - base) % num_frames;
    if offset == 0 {
        timeslot
    } else {
        timeslot.wrapping_add(offset).wrapping_sub(num_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::{MultiFecHeaderCode, header_matrix_config};
    use crate::fec::BlockCode;
    use crate::fec::num_frames_for_delay;

    #[test]
    fn encode_sizes_pos_zero_returns_frame_size() {
        for delay in 1..5u16 {
            let (info, max_pkts) = header_matrix_config(delay).unwrap();
            let block = BlockCode::new(info, delay, (1, 0), 8).unwrap();
            let mut header = MultiFecHeaderCode::from_block_code(block, delay, max_pkts).unwrap();
            let frame_size = u64::from(delay);
            let enc = header.encode_sizes_of_frames(frame_size, 0).unwrap();
            assert_eq!(enc, frame_size);
            let parity = header.encode_sizes_of_frames(frame_size, 1).unwrap();
            assert_ne!(parity, frame_size);
        }
    }

    #[test]
    fn decode_single_frame_roundtrip() {
        for num_pkts in 1..5u16 {
            for start_pos in 0..num_pkts.saturating_sub(1) {
                let delay = 0u16;
                let (info, max_pkts) = header_matrix_config(delay).unwrap();
                let enc_block = BlockCode::new(info, delay, (1, 0), 8).unwrap();
                let mut enc =
                    MultiFecHeaderCode::from_block_code(enc_block, delay, max_pkts).unwrap();
                let dec_block = BlockCode::new(info, delay, (1, 0), 8).unwrap();
                let mut dec =
                    MultiFecHeaderCode::from_block_code(dec_block, delay, max_pkts).unwrap();

                for frame_num in 0..5u16 {
                    let size = u64::from(frame_num + 4);
                    for pos in 0..=start_pos + 1 {
                        let val = enc.encode_sizes_of_frames(size, pos).unwrap();
                        if pos == start_pos + 1 {
                            assert_ne!(val, size);
                            let recovered =
                                dec.decode_sizes_of_frames(frame_num, pos, val).unwrap();
                            assert_eq!(recovered.len(), 1);
                            assert_eq!(recovered[0].0, frame_num);
                            assert_eq!(recovered[0].1, size);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decode_recovers_after_lost_frame() {
        for delay in 1..4u16 {
            for lost_frame in 0..4u16 {
                let (info, max_pkts) = header_matrix_config(delay).unwrap();
                let enc_block = BlockCode::new(info, delay, (1, 0), 8).unwrap();
                let mut enc =
                    MultiFecHeaderCode::from_block_code(enc_block, delay, max_pkts).unwrap();
                let dec_block = BlockCode::new(info, delay, (1, 0), 8).unwrap();
                let mut dec =
                    MultiFecHeaderCode::from_block_code(dec_block, delay, max_pkts).unwrap();

                for frame_num in 0..lost_frame + 2 {
                    let size = u64::from(frame_num + 4);
                    for start_pos in 0..2u16 {
                        if frame_num == lost_frame {
                            let _ = enc.encode_sizes_of_frames(size, start_pos);
                        } else {
                            let val = enc.encode_sizes_of_frames(size, start_pos).unwrap();
                            let recovered = dec
                                .decode_sizes_of_frames(frame_num, start_pos, val)
                                .unwrap();
                            if start_pos == 0 {
                                assert_eq!(recovered.len(), 1);
                                assert_eq!(recovered[0].0, frame_num);
                                assert_eq!(recovered[0].1, size);
                            } else if frame_num == lost_frame + 1 {
                                assert_eq!(recovered.len(), 1);
                                assert_eq!(recovered[0].0, frame_num - 1);
                                assert_eq!(recovered[0].1, size - 1);
                            } else {
                                assert!(recovered.is_empty());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decode_intersperse_lost_frames() {
        for delay in 4..8u16 {
            for start in 0..8u16 {
                for pkts_per_frame in 1..4u16 {
                    let (info, max_pkts) = header_matrix_config(delay).unwrap();
                    let mut enc = MultiFecHeaderCode::from_block_code(
                        BlockCode::new(info, delay, (1, 0), 8).unwrap(),
                        delay,
                        max_pkts,
                    )
                    .unwrap();
                    let mut dec = MultiFecHeaderCode::from_block_code(
                        BlockCode::new(info, delay, (1, 0), 8).unwrap(),
                        delay,
                        max_pkts,
                    )
                    .unwrap();
                    let mut num_missing = 2u16;
                    for frame_num in 0..start + 6 {
                        let size = u64::from(20 - frame_num);
                        let max_pos = pkts_per_frame;
                        for frame_pos in 0..max_pos {
                            if frame_num == start || frame_num == start + 2 {
                                let _ = enc.encode_sizes_of_frames(size, frame_pos);
                            } else {
                                let val = enc.encode_sizes_of_frames(size, frame_pos).unwrap();
                                let recovered = dec
                                    .decode_sizes_of_frames(frame_num, frame_pos, val)
                                    .unwrap();
                                if frame_pos == 0 {
                                    assert_eq!(recovered.len(), 1);
                                    assert_eq!(recovered[0].0, frame_num);
                                    assert_eq!(recovered[0].1, size);
                                } else if (start + 1) % 2 != 0 || pkts_per_frame > 1 {
                                    if (frame_num == start + 1 || frame_num == start + 3)
                                        && frame_pos == 1
                                    {
                                        assert_eq!(recovered.len(), 1);
                                        assert_eq!(recovered[0].0, frame_num - 1);
                                        assert_eq!(recovered[0].1, u64::from(20 - frame_num + 1));
                                    }
                                } else if frame_num >= start + 3 && num_missing != 0 {
                                    num_missing -= 1;
                                    if num_missing == 0 {
                                        assert_eq!(recovered.len(), 2);
                                        for j in 0..2u16 {
                                            let frame = recovered[j as usize].0;
                                            assert!(frame == start || frame == start + 2);
                                            assert_ne!(recovered[1 - j as usize].0, frame);
                                            assert_eq!(
                                                recovered[j as usize].1,
                                                u64::from(20 - frame)
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn header_matrix_rows_divisible_by_num_frames() {
        for delay in 1..8u16 {
            let num_frames = num_frames_for_delay(delay);
            let (info, max_pkts) = header_matrix_config(delay).unwrap();
            assert_eq!(info.n_cols, num_frames);
            assert_eq!(info.n_rows % num_frames, 0);
            assert!(info.n_rows * info.n_cols <= 255);
            assert_eq!(max_pkts, info.n_rows / num_frames);
        }
    }
}
