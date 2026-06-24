//! Theoretical streaming-code recovery (`streaming_code_theoretical.*`).

use alloc::vec::Vec;

use crate::fec::num_frames_for_delay;

#[derive(Debug, Clone)]
pub(super) struct StreamingCodeTheoretical {
    delay: u16,
    num_frames: u16,
    num_missing_us: Vec<u16>,
    num_missing_vs: Vec<u16>,
    num_received_parities: Vec<u16>,
    unusable_parities: Vec<bool>,
    decode_scratch: Vec<bool>,
}

impl StreamingCodeTheoretical {
    pub(super) fn new(delay: u16) -> Self {
        Self {
            delay,
            num_frames: 0,
            num_missing_us: Vec::new(),
            num_missing_vs: Vec::new(),
            num_received_parities: Vec::new(),
            unusable_parities: Vec::new(),
            decode_scratch: Vec::new(),
        }
    }

    pub(super) fn add_frames(
        &mut self,
        num_missing_us: &[u16],
        num_missing_vs: &[u16],
        num_received_parities: &[u16],
        timeslot: u16,
    ) {
        let num_frames = num_frames_for_delay(self.delay);
        self.num_missing_us.clear();
        self.num_missing_vs.clear();
        self.num_received_parities.clear();
        self.unusable_parities.clear();
        self.num_frames = 0;
        for j in timeslot + 1..timeslot + 1 + num_frames {
            self.add_frame(
                num_missing_us[(j % num_frames) as usize],
                num_missing_vs[(j % num_frames) as usize],
                num_received_parities[(j % num_frames) as usize],
            );
        }
    }

    fn add_frame(&mut self, num_missing_u: u16, num_missing_v: u16, num_received_parity: u16) {
        self.num_missing_us.push(num_missing_u);
        self.num_missing_vs.push(num_missing_v);
        self.num_received_parities.push(num_received_parity);
        self.num_frames += 1;
        debug_assert!(self.num_frames <= num_frames_for_delay(self.delay));
        self.unusable_parities
            .push(self.num_frames <= self.delay || num_received_parity == 0);
        self.decode_streaming_code();
        if self.num_frames == num_frames_for_delay(self.delay) {
            self.set_unusable();
        }
    }

    pub(super) fn frame_u_recovered(&self) -> Vec<bool> {
        self.part_of_frame_recovered(false)
    }

    pub(super) fn frame_v_recovered(&self) -> Vec<bool> {
        self.part_of_frame_recovered(true)
    }

    pub(super) fn frame_unusable_parities(&self) -> &[bool] {
        &self.unusable_parities
    }

    fn part_of_frame_recovered(&self, is_v: bool) -> Vec<bool> {
        let counts = if is_v {
            &self.num_missing_vs
        } else {
            &self.num_missing_us
        };
        counts.iter().map(|&n| n == 0).collect()
    }

    fn decode_streaming_code(&mut self) {
        if self.num_received_parities.last().copied().unwrap_or(0) == 0 {
            return;
        }
        if self.decode_final() {
            let last = (self.num_frames - 1) as usize;
            self.decode_frame(last);
        } else {
            let n = self.num_frames as usize;
            let mut scratch = core::mem::take(&mut self.decode_scratch);
            if scratch.len() < n {
                scratch.resize(n, false);
            }
            scratch[..n].copy_from_slice(&self.unusable_parities);
            for start_pos in 0..n {
                self.decode_from(start_pos, &mut scratch[..n]);
            }
            self.decode_scratch = scratch;
        }
    }

    fn decode_frame(&mut self, position: usize) {
        if self.num_missing_us[position] + self.num_missing_vs[position]
            <= self.num_received_parities[position]
        {
            self.num_missing_us[position] = 0;
            self.num_missing_vs[position] = 0;
        }
    }

    fn frame_decoded(&self, position: usize) -> bool {
        self.num_missing_us[position] == 0 && self.num_missing_vs[position] == 0
    }

    fn decode_final(&self) -> bool {
        (0..self.num_frames as usize - 1).all(|j| self.frame_decoded(j))
    }

    fn decode_from(&mut self, start_pos: usize, scratch: &mut [bool]) {
        let (decode_u, decode_v) = self.decoding_availabilities(start_pos, scratch);
        if decode_u {
            self.num_missing_us[start_pos] = 0;
        }
        if decode_v {
            self.num_missing_vs[start_pos] = 0;
        }
        self.set_unusable_from_pos(start_pos, scratch);
    }

    fn set_unusable(&mut self) {
        let n = self.num_frames as usize;
        for pos in 0..n {
            Self::mark_unusable_slots(
                pos,
                self.delay,
                n,
                &self.num_missing_us,
                &self.num_missing_vs,
                &mut self.unusable_parities,
            );
        }
    }

    fn mark_unusable_slots(
        pos: usize,
        delay: u16,
        num_frames: usize,
        num_missing_us: &[u16],
        num_missing_vs: &[u16],
        unusable: &mut [bool],
    ) {
        if num_missing_us[pos] != 0 || num_missing_vs[pos] != 0 {
            unusable[pos] = true;
        }
        let delay_pos = pos + delay as usize;
        if num_missing_us[pos] > 0 && delay_pos < num_frames {
            unusable[delay_pos] = true;
        }
        if num_missing_vs[pos] > 0 {
            let end = core::cmp::min(num_frames, delay_pos + 1);
            for slot in unusable.iter_mut().take(end).skip(pos + 1) {
                *slot = true;
            }
        }
    }

    fn set_unusable_from_pos(&self, pos: usize, unusable: &mut [bool]) {
        Self::mark_unusable_slots(
            pos,
            self.delay,
            self.num_frames as usize,
            &self.num_missing_us,
            &self.num_missing_vs,
            unusable,
        );
    }

    fn net_v_parities(&self, position: usize, curr_unusable: &[bool]) -> u16 {
        if curr_unusable[position] {
            return 0;
        }
        if self.num_missing_us[position] >= self.num_received_parities[position] {
            return 0;
        }
        self.num_received_parities[position] - self.num_missing_us[position]
    }

    fn decode_v_deficit(&self, start: usize, parities_out: &[bool]) -> i32 {
        let mut num_lost_v = 0i32;
        let mut available_v_parity = 0i32;
        let end = core::cmp::min(self.num_frames as usize, start + self.delay as usize);
        for pos in start..end {
            num_lost_v += self.num_missing_vs[pos] as i32;
            available_v_parity += self.net_v_parities(pos, parities_out) as i32;
            if num_lost_v <= available_v_parity {
                return 0;
            }
        }
        num_lost_v - available_v_parity
    }

    fn decode_only_u(&self, start: usize, parities_out: &[bool]) -> bool {
        let mut lost = self.num_missing_us[start] as i32
            - if parities_out[start] {
                0
            } else {
                self.num_received_parities[start] as i32
            };
        if lost <= 0 {
            return true;
        }
        if start + self.delay as usize >= self.num_frames as usize {
            return false;
        }
        let d = start + self.delay as usize;
        lost += self.num_missing_us[d] as i32 + self.num_missing_vs[d] as i32
            - if parities_out[d] {
                0
            } else {
                self.num_received_parities[d] as i32
            };
        lost <= 0
    }

    fn decode_success(
        &self,
        start: usize,
        parities_out: &[bool],
        mut deficit: i32,
    ) -> (bool, bool) {
        if start + self.delay as usize >= self.num_frames as usize {
            return (false, false);
        }
        let d = start + self.delay as usize;
        deficit += self.num_missing_us[start] as i32
            + self.num_missing_us[d] as i32
            + self.num_missing_vs[d] as i32
            - if parities_out[d] {
                0
            } else {
                self.num_received_parities[d] as i32
            };
        let ok = deficit <= 0;
        (ok, ok)
    }

    fn decoding_availabilities(&self, start: usize, parities_out: &[bool]) -> (bool, bool) {
        let deficit = self.decode_v_deficit(start, parities_out);
        if deficit == 0 {
            (self.decode_only_u(start, parities_out), true)
        } else {
            self.decode_success(start, parities_out, deficit)
        }
    }
}
