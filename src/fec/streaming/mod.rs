//! High-level streaming FEC over one [`BlockCode`] window.
//!
//! [`StreamingCode`] drives a sliding window of [`BlockCode`]s; the supporting
//! submodules handle stripe placement ([`stripes`]), recoverability analysis
//! ([`recovery`], [`theoretical`], [`flow_decode`]) and matrix-position
//! reordering ([`reorder`]).

mod flow_decode;
pub(crate) mod recovery;
mod reorder;
pub(crate) mod stripes;
mod theoretical;

#[cfg(any(test, feature = "bench"))]
pub(super) mod test_common;

use alloc::vec::Vec;

use crate::datagram::FecDatagram;

use self::stripes::{
    StripePlacement, compute_num_data_stripes, expired_frame, final_pos_within_stripe,
    first_pos_of_frame_in_block_code, first_pos_of_frame_in_block_code_erased, get_frame,
    parity_payloads_into, place_frame, stripe_slice,
};
use crate::fec::error::FecResult;
use crate::fec::num_frames_for_delay;
use crate::fec::{BlockCode, CodingMatrixInfo};

pub(crate) struct StreamingCode {
    delay: u16,
    num_frames: u16,
    stripe_size: u16,
    frame_max_data_stripes: u16,
    frame_max_fec_stripes: u16,
    timeslot: Option<u16>,
    coding_matrix_info: CodingMatrixInfo,
    block_code: BlockCode,
}

impl StreamingCode {
    pub(crate) fn new(
        delay: u16,
        stripe_size: u16,
        block_code: BlockCode,
        w: u16,
        frame_max_data_stripes: u16,
        frame_max_fec_stripes: u16,
    ) -> Result<Self, crate::fec::error::FecError> {
        let num_frames = num_frames_for_delay(delay);
        let info = block_code.coding_matrix_info();
        if info.n_rows != num_frames * frame_max_fec_stripes
            || info.n_cols != num_frames * frame_max_data_stripes
            || stripe_size != w * block_code.packet_size()
            || block_code.delay() != delay
        {
            return Err(crate::fec::error::FecError::InvalidStreamingCodeConfig);
        }
        Ok(Self {
            delay,
            num_frames,
            stripe_size,
            frame_max_data_stripes,
            frame_max_fec_stripes,
            timeslot: None,
            coding_matrix_info: info,
            block_code,
        })
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn set_timeslot(&mut self, timeslot: u16) {
        self.timeslot = Some(timeslot);
    }

    pub(crate) fn stream_timeslot(&self) -> Option<u16> {
        self.timeslot
    }

    pub(crate) fn recovered_frame_pos_to_frame_num(&self, timeslot: u16, pos: u16) -> u16 {
        (timeslot + pos + 1) % self.num_frames
    }

    pub(crate) fn updated_ts(&mut self, frame_num: u16) -> FecResult<u16> {
        if let Some(ts) = self.timeslot {
            if frame_num == ts {
                return Ok(ts);
            }
            if frame_num > ts {
                let mut t = ts.wrapping_add(1);
                while t != frame_num.wrapping_add(1) {
                    self.block_code.update_timeslot(t)?;
                    t = t.wrapping_add(1);
                }
                self.timeslot = Some(frame_num);
                return Ok(frame_num);
            }
            if ts.wrapping_sub(frame_num) > crate::datagram::MAX_FRAME_NUM / 2 {
                let mut t = ts.wrapping_add(1);
                while t != frame_num.wrapping_add(1) {
                    self.block_code.update_timeslot(t)?;
                    t = t.wrapping_add(1);
                }
                self.timeslot = Some(frame_num);
                return Ok(frame_num);
            }
            return Ok(ts);
        }
        let mut t = 0u16;
        while t != frame_num.wrapping_add(1) {
            self.block_code.update_timeslot(t)?;
            t = t.wrapping_add(1);
        }
        self.timeslot = Some(frame_num);
        Ok(frame_num)
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn into_block_code(self) -> BlockCode {
        self.block_code
    }

    pub(crate) fn encode(
        &mut self,
        data_pkts: &[FecDatagram],
        parity_pkt_sizes: &[u16],
        frame_size: u32,
        parity_out: &mut Vec<u8>,
    ) -> FecResult<()> {
        let ts = *self.timeslot.get_or_insert(0);
        self.block_code.update_timeslot(ts)?;
        let frame = get_frame(data_pkts, frame_size);
        let placement = StripePlacement {
            stripe_size: self.stripe_size,
            frame_max_data_stripes: self.frame_max_data_stripes,
            frame_max_fec_stripes: self.frame_max_fec_stripes,
            num_frames: self.num_frames,
        };
        place_frame(&frame, ts, &placement, &mut self.block_code, frame_size)?;
        self.block_code.pad(
            ts,
            compute_num_data_stripes(frame.len() as u32, self.stripe_size),
        )?;
        self.block_code.encode()?;
        parity_payloads_into(
            self.coding_matrix_info,
            ts,
            parity_pkt_sizes,
            &placement,
            &mut self.block_code,
            self.delay,
            parity_out,
        )?;
        self.timeslot = Some(ts.wrapping_add(1));
        Ok(())
    }

    pub(crate) fn add_packet(&mut self, pkt: &FecDatagram, frame_num: u16) -> FecResult<()> {
        self.timeslot = Some(self.updated_ts(frame_num)?);
        debug_assert_eq!(pkt.payload.len() % self.stripe_size as usize, 0);
        let start = first_pos_of_frame_in_block_code(
            pkt.frame_num,
            self.num_frames,
            self.frame_max_data_stripes,
            self.frame_max_fec_stripes,
            pkt.is_parity,
        );
        let start_pos = start + pkt.stripe_pos_in_frame;
        let stripes_per_pkt = pkt.payload.len() / self.stripe_size as usize;
        for rel in 0..stripes_per_pkt {
            let stripe = stripe_slice(&pkt.payload, rel as u16, self.stripe_size, self.stripe_size);
            self.block_code
                .place_payload(start_pos + rel as u16, pkt.is_parity, stripe)?;
        }
        Ok(())
    }

    pub(crate) fn is_stripe_recovered(
        &self,
        frame_num: u16,
        stripe_num: u16,
        frame_size: u32,
    ) -> bool {
        debug_assert!(stripe_num < compute_num_data_stripes(frame_size, self.stripe_size));
        let first = first_pos_of_frame_in_block_code_erased(
            frame_num,
            self.num_frames,
            self.frame_max_data_stripes,
            self.frame_max_fec_stripes,
            false,
        );
        let pos = first + stripe_num;
        !self.block_code.erased()[pos as usize]
    }

    #[cfg(test)]
    pub(crate) fn recovered_stripe(
        &self,
        frame_size: u32,
        frame_num: u16,
        stripe_num: u16,
    ) -> FecResult<Vec<u8>> {
        let start = first_pos_of_frame_in_block_code_erased(
            frame_num,
            self.num_frames,
            self.frame_max_data_stripes,
            self.frame_max_fec_stripes,
            false,
        );
        let absolute = start + stripe_num;
        let row = self.block_code.row_slice(absolute, false)?;
        let sz = final_pos_within_stripe(frame_size, self.stripe_size, stripe_num) + 1;
        Ok(row[..sz as usize].to_vec())
    }

    pub(crate) fn is_frame_recovered(&self, frame_num: u16, frame_size: u32) -> bool {
        let ts = match self.timeslot {
            Some(t) => t,
            None => return false,
        };
        if expired_frame(frame_num, ts, self.num_frames) {
            return false;
        }
        let num_data = compute_num_data_stripes(frame_size, self.stripe_size);
        (0..num_data).all(|stripe| self.is_stripe_recovered(frame_num, stripe, frame_size))
    }

    pub(crate) fn recovered_frame(&self, frame_num: u16, frame_size: u32) -> FecResult<Vec<u8>> {
        if !self.is_frame_recovered(frame_num, frame_size) {
            return Err(crate::fec::error::FecError::FrameNotRecovered { frame_num });
        }
        let num_data = compute_num_data_stripes(frame_size, self.stripe_size);
        let start = first_pos_of_frame_in_block_code_erased(
            frame_num,
            self.num_frames,
            self.frame_max_data_stripes,
            self.frame_max_fec_stripes,
            false,
        );
        let mut payload = Vec::with_capacity(frame_size as usize);
        for stripe in 0..num_data {
            let row = self.block_code.row_slice(start + stripe, false)?;
            let sz = final_pos_within_stripe(frame_size, self.stripe_size, stripe) + 1;
            payload.extend_from_slice(&row[..sz as usize]);
        }
        Ok(payload)
    }

    pub(crate) fn pad_frames(
        &mut self,
        frame_sizes: &[Option<u64>],
        timeslot: u16,
    ) -> FecResult<()> {
        for t in 0..self.num_frames {
            if let Some(size) = frame_sizes.get(t as usize).copied().flatten() {
                let num_data = compute_num_data_stripes(size as u32, self.stripe_size);
                let fnum = self.recovered_frame_pos_to_frame_num(timeslot, t);
                self.block_code.pad(fnum, num_data)?;
            }
        }
        Ok(())
    }

    pub(crate) fn decode(&mut self, frame_sizes: &[Option<u64>]) -> FecResult<()> {
        let ts = self
            .timeslot
            .ok_or(crate::fec::error::FecError::TimeslotNotSet)?;
        if frame_sizes.len() != self.num_frames as usize {
            return Err(crate::fec::error::FecError::InvalidFrameSizeWindow);
        }
        self.pad_frames(frame_sizes, ts)?;
        self.decode_until_fixed_point()?;
        Ok(())
    }

    fn decode_until_fixed_point(&mut self) -> FecResult<()> {
        if !self.block_code.can_recover() {
            return Ok(());
        }
        let recovered = self.block_code.decode()?;
        if recovered.iter().any(|&r| r) {
            self.block_code.decode()?;
        }
        Ok(())
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn parity_row(&self, row: u16) -> FecResult<Vec<u8>> {
        self.block_code.get_row(row, true)
    }
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn place_parity_row(&mut self, row: u16, vals: &[u8]) -> FecResult<()> {
        self.block_code.place_payload(row, true, vals)
    }

    #[cfg(test)]
    pub(crate) fn block_code_mut(&mut self) -> &mut BlockCode {
        &mut self.block_code
    }

    #[cfg(test)]
    pub(crate) fn erased_slice(&self) -> &[bool] {
        self.block_code.erased()
    }

    #[cfg(test)]
    pub(crate) fn block_code(&self) -> &BlockCode {
        &self.block_code
    }
}

// C++ `test_streaming_code.cc` tests (non-burst).

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::StreamingCode;
    use super::stripes::parity_packet_slices;
    use super::stripes::{
        compute_num_data_stripes, final_pos_within_stripe, first_pos_of_frame_in_block_code,
    };
    use super::test_common::{frame_sizes_window, make_pair, stripe_payload};
    use crate::datagram::FecDatagram;
    use crate::fec::BlockCode;
    use crate::fec::CodingMatrixInfo;
    use crate::fec::num_frames_for_delay;
    use crate::fec::test_util::{GlibcRand, get_missing};

    const MAX_DATA_STRIPES: u16 = 50;
    const MAX_FEC_STRIPES: u16 = 25;

    fn large_pair(tau: u16) -> (StreamingCode, u16, u16, u16) {
        let w = crate::word_size::WordSize::W32;
        let packet_size = 8u16;
        let stripe_size = packet_size * u16::from(w);
        let num_frames = num_frames_for_delay(tau);
        let info = CodingMatrixInfo::new(
            num_frames * MAX_FEC_STRIPES,
            num_frames * MAX_DATA_STRIPES,
            w,
        )
        .unwrap();
        let block = BlockCode::new(info, tau, (1, 1), packet_size).unwrap();
        let code = StreamingCode::new(
            tau,
            stripe_size,
            block,
            u16::from(w),
            MAX_DATA_STRIPES,
            MAX_FEC_STRIPES,
        )
        .unwrap();
        (code, stripe_size, packet_size, num_frames)
    }

    #[test]
    fn updated_ts() {
        let mut rng = GlibcRand::new(0);
        for tau in 0..4u16 {
            for _trial in 0..20 {
                let (mut code, _stripe_size, _packet_size, num_frames) = large_pair(tau);
                let mut ts = 0u16;
                let num_init = rng.rand_usize(num_frames as usize * 5) as u16;
                while ts < num_init {
                    assert_eq!(ts, code.updated_ts(ts).unwrap());
                    code.set_timeslot(ts);
                    ts += 1;
                }
                let extra = rng.rand_usize(2 * num_frames as usize) as u16;
                ts += extra;

                for s in 0..(MAX_DATA_STRIPES * num_frames) {
                    code.block_code_mut().place_payload(s, false, &[1]).unwrap();
                }
                assert_eq!(ts, code.updated_ts(ts).unwrap());

                let mut start = num_init;
                let mut end = ts;
                let mut end_check = start + num_frames;
                if start + num_frames - 1 <= ts {
                    end = start + num_frames - 1;
                    end_check = end.saturating_sub(1);
                }
                if tau == 0 {
                    start = ts;
                    end = ts;
                    end_check = 0;
                }
                for j in start..=end {
                    let row = first_pos_of_frame_in_block_code(
                        j,
                        num_frames,
                        MAX_DATA_STRIPES,
                        MAX_FEC_STRIPES,
                        false,
                    );
                    assert_eq!(code.block_code().get_row(row, false).unwrap()[0], 0);
                }
                for j in end + 1..end_check {
                    let row = first_pos_of_frame_in_block_code(
                        j,
                        num_frames,
                        MAX_DATA_STRIPES,
                        MAX_FEC_STRIPES,
                        false,
                    );
                    assert_eq!(code.block_code().get_row(row, false).unwrap()[0], 1);
                }
            }
        }
    }

    #[test]
    fn add_pkt_is_stripe_recovered_recovered_stripe() {
        let mut rng = GlibcRand::new(0);
        for tau in 0..4u16 {
            for _trial in 0..20 {
                for stripes_per_pkt in 1..4u16 {
                    let (mut code, stripe_size, _packet_size, num_frames) = large_pair(tau);
                    let mut frame_num = 0u16;
                    let num_init = rng.rand_usize(num_frames as usize * 5) as u16 + num_frames;
                    while frame_num < num_init {
                        code.updated_ts(frame_num).unwrap();
                        code.set_timeslot(frame_num);
                        frame_num += 1;
                    }
                    frame_num += rng.rand_usize(2 * num_frames as usize) as u16 + 1;

                    let mut frame_size =
                        rng.rand_usize(MAX_DATA_STRIPES as usize * stripe_size as usize) as u16;
                    if frame_size == 0 {
                        frame_size = 1;
                    }
                    while !frame_size.is_multiple_of(stripe_size) {
                        frame_size += 1;
                    }
                    while !frame_size.is_multiple_of(stripes_per_pkt * stripe_size) {
                        frame_size += 1;
                    }
                    let num_pkts = frame_size / (stripes_per_pkt * stripe_size);
                    let stripe_pos =
                        (rng.rand_usize(num_pkts as usize) * stripes_per_pkt as usize) as u16;
                    let pkt = FecDatagram {
                        seq_num: 0,
                        is_parity: false,
                        frame_num,
                        sizes_of_frames_encoding: 0,
                        pos_in_frame: stripe_pos,
                        stripe_pos_in_frame: stripe_pos,
                        payload: Bytes::from(vec![1u8; (stripes_per_pkt * stripe_size) as usize]),
                    };
                    code.add_packet(&pkt, frame_num).unwrap();

                    for check_frame in 0..num_frames {
                        if check_frame % num_frames == frame_num % num_frames {
                            let n_stripes =
                                compute_num_data_stripes(u32::from(frame_size), stripe_size);
                            for stripe in 0..n_stripes {
                                let is_recovered = stripe >= pkt.stripe_pos_in_frame
                                    && stripe < pkt.stripe_pos_in_frame + stripes_per_pkt;
                                assert_eq!(
                                    code.is_stripe_recovered(
                                        check_frame,
                                        stripe,
                                        u32::from(frame_size)
                                    ),
                                    is_recovered
                                );
                                let expected_len = final_pos_within_stripe(
                                    u32::from(frame_size),
                                    stripe_size,
                                    stripe,
                                ) + 1;
                                let recovered = code
                                    .recovered_stripe(u32::from(frame_size), check_frame, stripe)
                                    .unwrap();
                                assert_eq!(recovered.len(), expected_len as usize);
                                assert!(
                                    recovered
                                        .iter()
                                        .all(|&b| b == if is_recovered { 1 } else { 0 })
                                );
                            }
                        } else {
                            let size = rng
                                .rand_usize(MAX_DATA_STRIPES as usize * stripe_size as usize)
                                as u16;
                            for stripe in 0..compute_num_data_stripes(u32::from(size), stripe_size)
                            {
                                assert!(!code.is_stripe_recovered(
                                    check_frame,
                                    stripe,
                                    u32::from(size)
                                ));
                                let expected_len =
                                    final_pos_within_stripe(u32::from(size), stripe_size, stripe)
                                        + 1;
                                let recovered = code
                                    .recovered_stripe(u32::from(size), check_frame, stripe)
                                    .unwrap();
                                assert_eq!(recovered.len(), expected_len as usize);
                                assert!(recovered.iter().all(|&b| b == 0));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn is_frame_recovered_expired() {
        let mut rng = GlibcRand::new(0);
        for tau in 0..4u16 {
            for _trial in 0..2 {
                let num_frames = num_frames_for_delay(tau);
                for start in num_frames..(4 * num_frames) {
                    let (mut code, _stripe_size, _packet_size, _) = large_pair(tau);
                    let mut frame_num = 0u16;
                    while frame_num <= start {
                        code.updated_ts(frame_num).unwrap();
                        code.set_timeslot(frame_num);
                        frame_num += 1;
                    }
                    for t in 0..=(start - num_frames) {
                        let size = rng.rand_usize(10_000) as u16 + 1;
                        assert!(!code.is_frame_recovered(t, u32::from(size)));
                    }
                }
            }
        }
    }

    #[test]
    fn is_frame_recovered_and_recovered_frame() {
        let mut rng = GlibcRand::new(0);
        for tau in 0..4u16 {
            for trial in 0..2u16 {
                let num_frames = num_frames_for_delay(tau);
                for start in num_frames..(4 * num_frames) {
                    let (mut code, stripe_size, _packet_size, _) = large_pair(tau);
                    let mut frame_num = 0u16;
                    let mut sizes = Vec::new();
                    let mut missings = Vec::new();
                    let mut num_missings = Vec::new();

                    while frame_num <= start {
                        code.updated_ts(frame_num).unwrap();
                        code.set_timeslot(frame_num);
                        if frame_num > start - num_frames {
                            let max_size = if trial == 0 {
                                2 * stripe_size
                            } else {
                                MAX_DATA_STRIPES * stripe_size
                            };
                            let size = rng.rand_usize(max_size as usize) as u16 + 1;
                            let num_data_stripes =
                                compute_num_data_stripes(u32::from(size), stripe_size);
                            code.block_code_mut()
                                .pad(frame_num, num_data_stripes)
                                .unwrap();
                            let num_missing = rng.rand_usize(num_data_stripes as usize + 1) as u8;
                            let missing =
                                get_missing(num_data_stripes as u8, num_missing, &mut rng);
                            sizes.push(size);
                            missings.push(missing.clone());
                            num_missings.push(num_missing);
                            for pos in 0..num_data_stripes {
                                let pkt = FecDatagram {
                                    seq_num: 0,
                                    is_parity: false,
                                    frame_num,
                                    sizes_of_frames_encoding: 0,
                                    pos_in_frame: pos,
                                    stripe_pos_in_frame: pos,
                                    payload: Bytes::from(vec![
                                        (frame_num + 1) as u8;
                                        stripe_size as usize
                                    ]),
                                };
                                if !missing[pos as usize] {
                                    code.add_packet(&pkt, frame_num).unwrap();
                                }
                            }
                        }
                        frame_num += 1;
                    }

                    let window_start = start - num_frames + 1;
                    for t in window_start..=start {
                        let idx = (t - window_start) as usize;
                        let size = sizes[idx];
                        let num_missing = num_missings[idx];
                        assert_eq!(
                            code.is_frame_recovered(t, u32::from(size)),
                            num_missing == 0
                        );
                        if num_missing == 0 {
                            let recovered = code.recovered_frame(t, u32::from(size)).unwrap();
                            assert_eq!(recovered.len(), size as usize);
                            assert!(recovered.iter().all(|&b| b == (t + 1) as u8));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pad_frames() {
        let mut rng = GlibcRand::new(0);
        for tau in 0..4u16 {
            for _trial in 0..2 {
                for _stripes_per_pkt in 1..4u16 {
                    let num_frames = num_frames_for_delay(tau);
                    for timeslot in 0..(3 * num_frames) {
                        let (mut code, stripe_size, _packet_size, _) = large_pair(tau);
                        let mut frame_num = 0u16;
                        while frame_num < num_frames {
                            code.updated_ts(frame_num).unwrap();
                            code.set_timeslot(frame_num);
                            frame_num += 1;
                        }
                        let mut frame_sizes: Vec<Option<u64>> =
                            (0..num_frames).map(|_| Some(0)).collect();
                        for pos in 0..num_frames {
                            frame_sizes[pos as usize] = Some(1 + rng.rand_usize(10_000) as u64);
                        }
                        code.pad_frames(&frame_sizes, timeslot).unwrap();
                        for j in 0..num_frames {
                            for stripe in 0..MAX_DATA_STRIPES {
                                let fnum = code.recovered_frame_pos_to_frame_num(timeslot, j);
                                let pos = MAX_DATA_STRIPES * fnum + stripe;
                                let expected = stripe
                                    < compute_num_data_stripes(
                                        frame_sizes[j as usize].unwrap() as u32,
                                        stripe_size,
                                    );
                                assert_eq!(code.erased_slice()[pos as usize], expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn encode_decode_within_frame() {
        let mut rng = GlibcRand::new(0);
        let w = crate::word_size::WordSize::W8;
        let packet_size = 8u16;
        let stripe_size = packet_size * u16::from(w);

        for n_data in 1..5u16 {
            for n_par in 1..5u16 {
                for delay in 0..4u16 {
                    let num_frames = num_frames_for_delay(delay);
                    for timeslot in 0..(num_frames * 3) {
                        for u in 0..2u16 {
                            let vu = (1u16, u);
                            let size = n_data * stripe_size;
                            let parity_sizes = vec![stripe_size; n_par as usize];
                            let (mut encode, mut decode, info) =
                                make_pair(delay, stripe_size, packet_size, w, n_data, n_par, vu);
                            let _ = info;
                            let mut sqn = 0u32;

                            for time in 0..timeslot {
                                let mut pkts = Vec::new();
                                for pos in 0..n_data {
                                    let col = (time % num_frames) * n_data + pos;
                                    pkts.push(FecDatagram {
                                        seq_num: sqn,
                                        is_parity: false,
                                        frame_num: time,
                                        sizes_of_frames_encoding: 0,
                                        pos_in_frame: pos,
                                        stripe_pos_in_frame: pos,
                                        payload: Bytes::from(stripe_payload(col, stripe_size)),
                                    });
                                    sqn += 1;
                                }
                                let mut parity_scratch = Vec::new();
                                encode
                                    .encode(
                                        &pkts,
                                        &parity_sizes,
                                        u32::from(size),
                                        &mut parity_scratch,
                                    )
                                    .unwrap();
                                decode
                                    .encode(
                                        &pkts,
                                        &parity_sizes,
                                        u32::from(size),
                                        &mut parity_scratch,
                                    )
                                    .unwrap();
                            }

                            if timeslot > 0 {
                                decode.set_timeslot(timeslot - 1);
                            }

                            let max_missing = n_data.min(n_par) + 1;
                            let missing = get_missing(
                                n_data as u8,
                                rng.rand_usize(max_missing as usize) as u8,
                                &mut rng,
                            );
                            let mut num_missing = 0u8;
                            let mut pkts = Vec::new();
                            for pos in 0..n_data {
                                let col = (timeslot % num_frames) * n_data + pos;
                                let pkt = FecDatagram {
                                    seq_num: sqn,
                                    is_parity: false,
                                    frame_num: timeslot,
                                    sizes_of_frames_encoding: 0,
                                    pos_in_frame: pos,
                                    stripe_pos_in_frame: pos,
                                    payload: Bytes::from(stripe_payload(col, stripe_size)),
                                };
                                sqn += 1;
                                if !missing[pos as usize] {
                                    decode.add_packet(&pkt, timeslot).unwrap();
                                } else {
                                    num_missing += 1;
                                }
                                pkts.push(pkt);
                            }

                            let mut parity_out = Vec::new();
                            encode
                                .encode(&pkts, &parity_sizes, u32::from(size), &mut parity_out)
                                .unwrap();
                            let missing_par = get_missing(
                                n_par as u8,
                                rng.rand_usize((n_par - num_missing as u16 + 1) as usize) as u8,
                                &mut rng,
                            );
                            for pos in 0..n_par {
                                if !missing_par[pos as usize] {
                                    let payload = parity_packet_slices(&parity_out, &parity_sizes)
                                        .nth(pos as usize)
                                        .expect("parity slice");
                                    let parity_pkt = FecDatagram {
                                        seq_num: sqn,
                                        is_parity: true,
                                        frame_num: timeslot,
                                        sizes_of_frames_encoding: 0,
                                        pos_in_frame: pos,
                                        stripe_pos_in_frame: pos,
                                        payload: Bytes::copy_from_slice(payload),
                                    };
                                    sqn += 1;
                                    decode.add_packet(&parity_pkt, timeslot).unwrap();
                                }
                            }

                            if num_missing == 0 {
                                continue;
                            }

                            decode
                                .decode(&frame_sizes_window(timeslot, num_frames, size as u64))
                                .unwrap();
                            assert!(decode.is_frame_recovered(timeslot, u32::from(size)));
                        }
                    }
                }
            }
        }
    }
}

// Burst loss scenarios from C++ `test_streaming_code.cc`.

#[cfg(test)]
mod burst_tests {
    use super::test_common::{BurstScenario, assert_burst_stripes, run_burst};
    use crate::fec::num_frames_for_delay;
    use crate::word_size::WordSize;

    const DELAY: u16 = 3;
    const W: WordSize = WordSize::W8;
    const PACKET_SIZE: u16 = 8;

    fn timeslots(num_frames: u16) -> impl Iterator<Item = u16> + Clone {
        (num_frames - 1)..(4 * num_frames)
    }

    fn assert_burst(scenario: BurstScenario, packet_size: u16, n_data: u16, timeslot: u16) {
        let (decode, info, size, helper) =
            run_burst(scenario, DELAY, W, packet_size, n_data, timeslot);
        assert_burst_stripes(
            scenario, &decode, info, DELAY, size, n_data, timeslot, &helper,
        );
    }

    #[test]
    fn burst_scenarios() {
        let num_frames = num_frames_for_delay(DELAY);
        let ts = timeslots(num_frames);

        for exp_loss in 0..4u16 {
            for n_data in 1..5u16 {
                for timeslot in ts.clone() {
                    assert_burst(
                        BurstScenario::FullRecover { exp_loss },
                        PACKET_SIZE,
                        n_data,
                        timeslot,
                    );
                }
            }
        }

        for n_data in (2..10).step_by(2) {
            for timeslot in ts.clone() {
                assert_burst(BurstScenario::PartialRecover, PACKET_SIZE, n_data, timeslot);
            }
        }

        for packet_size in [8u16, 16] {
            for n_data in (2..10).step_by(2) {
                for timeslot in ts.clone() {
                    assert_burst(
                        BurstScenario::PartialRecoverExpPlus,
                        packet_size,
                        n_data,
                        timeslot,
                    );
                }
            }
        }

        for n_data in (2..10).step_by(2) {
            for timeslot in ts.clone() {
                assert_burst(
                    BurstScenario::FullRecoverExpPlus,
                    PACKET_SIZE,
                    n_data,
                    timeslot,
                );
            }
        }

        for n_data in (2..9).step_by(2) {
            for timeslot in ts.clone() {
                assert_burst(BurstScenario::VRecover, PACKET_SIZE, n_data, timeslot);
            }
        }

        for n_data in (2..9).step_by(2) {
            for timeslot in ts.clone() {
                assert_burst(BurstScenario::NotRecovered, PACKET_SIZE, n_data, timeslot);
            }
        }

        for n_data in 1..5u16 {
            for shift_start in (1..=3).rev() {
                for timeslot in ts.clone() {
                    assert_burst(
                        BurstScenario::FullRecoverExp { shift_start },
                        PACKET_SIZE,
                        n_data,
                        timeslot,
                    );
                }
            }
        }

        for n_data in (2..9).step_by(2) {
            for shift_start in (1..=3).rev() {
                for timeslot in ts.clone() {
                    assert_burst(
                        BurstScenario::NotRecoveredExp { shift_start },
                        PACKET_SIZE,
                        n_data,
                        timeslot,
                    );
                }
            }
        }
    }
}

// Regression: τ=3 burst at timeslot=6 with exp_loss=1 (flow + decode path).

#[cfg(test)]
mod burst_regression {
    use bytes::Bytes;

    use super::StreamingCode;
    use super::stripes::parity_packet_slices;
    use crate::datagram::FecDatagram;
    use crate::fec::BlockCode;
    use crate::fec::CodingMatrixInfo;
    use crate::fec::num_frames_for_delay;

    fn stripe_payload(col: u16, stripe_size: u16) -> Vec<u8> {
        vec![(3 * col + 1) as u8; stripe_size as usize]
    }

    #[test]
    fn burst_frame3_recovers_at_timeslot6_exp_loss1() {
        let delay = 3u16;
        let w = crate::word_size::WordSize::W32;
        let packet_size = 8u16;
        let stripe_size = packet_size * u16::from(w);
        let num_frames = num_frames_for_delay(delay);
        let shift = num_frames - 1;
        let n_data = 2u16;
        let n_par = 2u16;
        let size = n_data * stripe_size;
        let timeslot = 6u16;
        let exp_loss = 1u16;
        let parity_sizes = vec![stripe_size; n_par as usize];

        let info = CodingMatrixInfo::new(n_par * num_frames, n_data * num_frames, w).unwrap();
        let encode_block = BlockCode::new(info, delay, (1, 0), packet_size).unwrap();
        let decode_block = BlockCode::new(info, delay, (1, 0), packet_size).unwrap();
        let w_bits = u16::from(w);
        let mut encode =
            StreamingCode::new(delay, stripe_size, encode_block, w_bits, n_data, n_par).unwrap();
        let mut decode =
            StreamingCode::new(delay, stripe_size, decode_block, w_bits, n_data, n_par).unwrap();

        let mut sqn = 0u32;
        for time in 0..=timeslot {
            let mut data_pkts = Vec::new();
            for pos in 0..n_data {
                let col = (time % num_frames) * n_data + pos;
                data_pkts.push(FecDatagram {
                    seq_num: sqn,
                    is_parity: false,
                    frame_num: time,
                    sizes_of_frames_encoding: 0,
                    pos_in_frame: pos,
                    stripe_pos_in_frame: pos,
                    payload: Bytes::from(stripe_payload(col, stripe_size)),
                });
                sqn += 1;
            }
            let mut parity_out = Vec::new();
            encode
                .encode(&data_pkts, &parity_sizes, u32::from(size), &mut parity_out)
                .unwrap();

            let fm = time % num_frames;
            let parity_missing = fm == (timeslot - shift + 3) % num_frames
                || fm == (timeslot - shift + 4) % num_frames;
            let skip_0 = fm == (timeslot - shift) % num_frames && !exp_loss.is_multiple_of(2);
            let skip_1 = fm == (timeslot - shift + 1) % num_frames && exp_loss / 2 != 0;

            if !parity_missing && !skip_0 && !skip_1 {
                for pkt in &data_pkts {
                    decode.add_packet(pkt, time).unwrap();
                }
                for (pos, payload) in parity_packet_slices(&parity_out, &parity_sizes).enumerate() {
                    let parity_pkt = FecDatagram {
                        seq_num: sqn,
                        is_parity: true,
                        frame_num: time,
                        sizes_of_frames_encoding: 0,
                        pos_in_frame: pos as u16,
                        stripe_pos_in_frame: pos as u16,
                        payload: Bytes::copy_from_slice(payload),
                    };
                    sqn += 1;
                    decode.add_packet(&parity_pkt, time).unwrap();
                }
            }
        }

        decode.set_timeslot(timeslot - 1);
        let mut frame_sizes = Vec::new();
        if timeslot < num_frames - 1 {
            for _ in 0..(num_frames - timeslot - 1) as usize {
                frame_sizes.push(None);
            }
        }
        while frame_sizes.len() < num_frames as usize {
            frame_sizes.push(Some(size as u64));
        }
        decode.decode(&frame_sizes).unwrap();

        assert!(
            decode.is_stripe_recovered(3, 0, u32::from(size)),
            "frame 3 stripe 0 should recover via parities from frames 5/6"
        );
    }
}
