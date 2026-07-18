//! Stripe → datagram layout.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::config::Config;
use crate::feedback::Feedback;

/// Default MTU for Tambur packetization (`Packetization::MTU`).
pub(crate) const DEFAULT_MTU: u16 = 1500;

/// Parity ratio `(numerator, denominator)` for a [`Feedback`] level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParityRatio {
    num: u32,
    den: u32,
}

impl ParityRatio {
    fn num_parity_stripes(self, num_data_stripes: u16) -> u16 {
        if self.num == 0 {
            return 0;
        }
        let n = u32::from(num_data_stripes) * self.num;
        n.div_ceil(self.den) as u16
    }
}

fn parity_ratio(feedback: Feedback) -> ParityRatio {
    match feedback {
        Feedback::None => ParityRatio { num: 0, den: 1 },
        Feedback::High => ParityRatio { num: 1, den: 2 },
        Feedback::Low => ParityRatio { num: 1, den: 4 },
    }
}

/// One planned on-wire packet: payload length and whether it is parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PacketSpec {
    pub payload_len: u16,
    pub is_parity: bool,
}

/// Tambur streaming packetization with optional parity delay and window parity top-up.
///
/// The sliding window tracks `(data_stripes, parity_stripes)` for each frame and
/// enforces a credit/deficit model: total parity in the window must be at least
/// `ratio_baseline(total_data) + K_min`.  This prevents a single large frame's
/// parity surplus from starving subsequent small frames.
pub(crate) struct StreamingPacketization {
    stripe_size: u16,
    max_stripes_per_pkt: u16,
    delayed_parity: VecDeque<u16>,
    // sliding-window parity top-up (K_min)
    min_window_parity: u16,
    window_len: usize,
    window_data_sum: u16,
    window_parity_sum: u16,
    window_history: VecDeque<(u16, u16)>,
    /// Upper bound on parity stripes per frame; the adjusted count is
    /// clamped to this value so the FEC code never receives more stripes
    /// than its generator matrix has rows.
    max_fec_stripes: u16,
}

impl StreamingPacketization {
    pub(crate) fn new(
        stripe_size: u16,
        max_pkt_size: u16,
        parity_delay: u16,
        window_len: u16,
        min_window_parity: u16,
        max_fec_stripes: u16,
    ) -> Self {
        let max_stripes_per_pkt = max_pkt_size / stripe_size.max(1);
        let mut delayed_parity = VecDeque::new();
        for _ in 0..parity_delay {
            delayed_parity.push_back(0);
        }
        Self {
            stripe_size,
            max_stripes_per_pkt: max_stripes_per_pkt.max(1),
            delayed_parity,
            min_window_parity,
            window_len: window_len as usize,
            window_data_sum: 0,
            window_parity_sum: 0,
            window_history: VecDeque::with_capacity(window_len as usize),
            max_fec_stripes,
        }
    }

    pub(crate) fn from_config(config: &Config) -> Self {
        Self::new(
            config.stripe_size() as u16,
            config.max_pkt_size.get(),
            config.parity_delay,
            config.window_frames(),
            config.min_window_parity,
            config.max_fec_stripes,
        )
    }

    /// Testing helper: expose the window_len used for window tracking.
    #[cfg(test)]
    pub(crate) fn window_len_for_test(&self) -> usize {
        self.window_len
    }

    pub(crate) fn plan_frame(&mut self, frame_size: u32, feedback: Feedback) -> Vec<PacketSpec> {
        let num_data = compute_num_data_stripes(frame_size, self.stripe_size);
        let base_parity = parity_ratio(feedback).num_parity_stripes(num_data);

        let produced = if self.min_window_parity > 0 {
            // slide window: remove oldest entry
            if self.window_history.len() == self.window_len {
                let (old_data, old_parity) = self.window_history.pop_front().unwrap();
                self.window_data_sum -= old_data;
                self.window_parity_sum -= old_parity;
            }
            // credit/deficit: window must have ratio(total_data) + K_min parity.
            let total_data = self.window_data_sum + num_data;
            let target =
                parity_ratio(feedback).num_parity_stripes(total_data) + self.min_window_parity;
            let gap = target.saturating_sub(self.window_parity_sum + base_parity);
            // Per-frame minimum floor: every frame contributes at least
            // ceil(K_min / window_len) surplus parity, ensuring uniform
            // protection even for small frames.
            let per_frame_extra = self.min_window_parity.div_ceil(self.window_len as u16);
            let adjusted = (base_parity + gap)
                .max(base_parity + per_frame_extra)
                .min(self.max_fec_stripes);
            self.window_history.push_back((num_data, adjusted));
            self.window_data_sum += num_data;
            self.window_parity_sum += adjusted;
            adjusted
        } else {
            base_parity
        };

        self.delayed_parity.push_back(produced);
        let emitted = self.delayed_parity.pop_front().unwrap_or(0);

        Self::specs_for_stripes(
            emitted,
            num_data,
            self.stripe_size,
            self.max_stripes_per_pkt,
        )
    }

    fn specs_for_stripes(
        num_parity_stripes: u16,
        num_data_stripes: u16,
        stripe_size: u16,
        mut stripes_per_pkt: u16,
    ) -> Vec<PacketSpec> {
        while stripes_per_pkt > 1
            && (!num_parity_stripes.is_multiple_of(stripes_per_pkt)
                || !num_data_stripes.is_multiple_of(stripes_per_pkt))
        {
            stripes_per_pkt -= 1;
        }
        // When the frame has very few packets, force spp=1 so each stripe
        // becomes its own packet.  This prevents a single packet loss from
        // wiping out multiple stripes (which would devastate small frames).
        let num_data_pkts = num_data_stripes / stripes_per_pkt;
        let num_parity_pkts = num_parity_stripes / stripes_per_pkt;
        if num_data_pkts + num_parity_pkts < 6 && stripes_per_pkt > 1 {
            stripes_per_pkt = 1;
        }
        let pkt_payload = stripe_size * stripes_per_pkt;
        let num_data_pkts = num_data_stripes / stripes_per_pkt;
        let num_parity_pkts = num_parity_stripes / stripes_per_pkt;

        let mut specs = Vec::with_capacity(num_data_pkts as usize + num_parity_pkts as usize);
        for _ in 0..num_data_pkts {
            specs.push(PacketSpec {
                payload_len: pkt_payload,
                is_parity: false,
            });
        }
        for _ in 0..num_parity_pkts {
            specs.push(PacketSpec {
                payload_len: pkt_payload,
                is_parity: true,
            });
        }
        specs
    }
}

fn compute_num_data_stripes(frame_size: u32, stripe_size: u16) -> u16 {
    let n = num_data_stripes_for_frame_size(frame_size, u32::from(stripe_size));
    n.try_into().expect("data stripes fits in u16")
}

/// Data stripes required for a frame of `frame_size` bytes (ceil division).
pub(crate) fn num_data_stripes_for_frame_size(frame_size: u32, stripe_size: u32) -> u32 {
    if frame_size == 0 || stripe_size == 0 {
        0
    } else {
        frame_size.div_ceil(stripe_size)
    }
}

/// Parity stripes at [`Feedback::High`] (50%) for a given data stripe count.
pub(crate) fn num_parity_stripes_high(num_data_stripes: u32) -> u32 {
    num_data_stripes.div_ceil(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: disable window tracking with `min_window_parity = 0`.
    fn pktn(stripe_size: u16, max_pkt_size: u16, parity_delay: u16) -> StreamingPacketization {
        StreamingPacketization::new(stripe_size, max_pkt_size, parity_delay, 0, 0, 1)
    }

    #[test]
    fn parity_stripes_high_is_half() {
        let mut p = pktn(256, 1500, 0);
        let specs = p.plan_frame(1024, Feedback::High);
        let data: u16 = specs
            .iter()
            .filter(|s| !s.is_parity)
            .map(|s| s.payload_len)
            .sum();
        let par: u16 = specs
            .iter()
            .filter(|s| s.is_parity)
            .map(|s| s.payload_len)
            .sum();
        assert_eq!(data, 1024);
        assert_eq!(par, 512);
    }

    #[test]
    fn none_has_no_parity_pkts() {
        let mut p = pktn(256, 1500, 0);
        let specs = p.plan_frame(512, Feedback::None);
        assert!(specs.iter().all(|s| !s.is_parity));
    }

    #[test]
    fn parity_delay_defers_parity() {
        let stripe = 256;
        let frame = 1024;
        let mut immediate = pktn(stripe, 1500, 0);
        assert!(
            immediate
                .plan_frame(frame, Feedback::High)
                .iter()
                .any(|s| s.is_parity)
        );

        let mut delayed = pktn(stripe, 1500, 2);
        assert!(
            !delayed
                .plan_frame(frame, Feedback::High)
                .iter()
                .any(|s| s.is_parity)
        );
        assert!(
            !delayed
                .plan_frame(frame, Feedback::High)
                .iter()
                .any(|s| s.is_parity)
        );
        assert!(
            delayed
                .plan_frame(frame, Feedback::High)
                .iter()
                .any(|s| s.is_parity)
        );
    }

    #[test]
    fn from_config_uses_parity_delay() {
        let config = Config::builder().parity_delay(1).build().unwrap();
        let mut p = StreamingPacketization::from_config(&config);
        assert!(
            !p.plan_frame(512, Feedback::High)
                .iter()
                .any(|s| s.is_parity)
        );
        assert!(
            p.plan_frame(512, Feedback::High)
                .iter()
                .any(|s| s.is_parity)
        );
    }

    // ---- window parity top-up tests ----

    #[test]
    fn min_window_parity_disabled_no_tracking() {
        // min_window_parity = 0 → same behavior as before
        let mut p = pktn(256, 1500, 0);
        let specs = p.plan_frame(256, Feedback::High);
        // 256 bytes → 1 data stripe → ceil(1/2) = 1 parity stripe → 1+1 = 2 pkts
        assert_eq!(specs.len(), 2);
        assert!(!specs[0].is_parity);
        assert!(specs[1].is_parity);
    }

    #[test]
    fn min_window_parity_tops_up_tiny_frames() {
        // stripe=256, 256-byte frames → 1 data stripe → base parity = 1 (ceil(1/2))
        // window_len = 7, min_window_parity = 7 → target = ceil(1/2) + 7 = 8
        // gap = 8 - (0 + 1) = 7 → adjusted = 8 parity stripes
        // specs_for_stripes(8, 1, 256, 5): stripes_per_pkt=1 → 8 parity packets × 256B
        let mut p = StreamingPacketization::new(256, 1500, 0, 7, 7, 128);
        let specs = p.plan_frame(256, Feedback::High);
        let par_bytes: u16 = specs
            .iter()
            .filter(|s| s.is_parity)
            .map(|s| s.payload_len)
            .sum();
        assert_eq!(
            par_bytes,
            8 * 256,
            "expected 8 parity stripes totalling {}B",
            8 * 256
        );
    }

    #[test]
    fn min_window_parity_reuses_window_accumulation() {
        // stripe=256, window_len=3, min=6, max=128
        // per_frame_extra = ceil(6/3) = 2
        // Frame 1 (512B): 2 data → 1 base, total_data=2, target=ceil(2/2)+6=7
        //   gap=7-(0+1)=6 → adjusted = max(1+6, 1+2) = 7
        // Frame 2 (512B): total_data=4, target=8, gap=8-(7+1)=0
        //   adjusted = max(1+0, 1+2) = 3
        // Frame 3 (512B): total_data=6, target=9, gap=9-(10+1)=0
        //   adjusted = max(1, 3) = 3
        // Frame 4 (512B): slide (2,7) → data_sum=4, par_sum=6
        //   total_data=6, target=9, gap=9-(6+1)=2 → adjusted = max(3, 3) = 3
        //   specs(3,2,256,5): stripes_per_pkt = 2 → 1+1=2 parity pkt × 512B = 1536
        //   Wait: 3 parity stripes at spp=2 → num_parity_pkts=1 (3/2=1 integer... 3%2≠0)
        //   Actually: specs_for_stripes(3,2,256,5): spp=5→4 (2%4≠0)→3→2 (2%2=0, 3%2≠0)→1
        //   spp=1 → 3 parity packets × 256B = 768
        let mut p = StreamingPacketization::new(256, 1500, 0, 3, 6, 128);
        for _ in 0..3 {
            p.plan_frame(512, Feedback::High);
        }
        let specs = p.plan_frame(512, Feedback::High);
        let par_bytes: u16 = specs
            .iter()
            .filter(|s| s.is_parity)
            .map(|s| s.payload_len)
            .sum();
        assert_eq!(
            par_bytes,
            3 * 256,
            "expected 3 parity stripes totalling {}B",
            3 * 256
        );
    }

    #[test]
    fn min_window_parity_tops_up_to_meet_data_dependent_target() {
        // 1024B → 4 data → 2 base parity.  K_min=2 means target = ceil(4/2)+2 = 4.
        // gap = 4 − (0 + 2) = 2 → adjusted = 4 parity stripes = 1024B.
        let mut p = StreamingPacketization::new(256, 1500, 0, 3, 2, 128);
        let specs = p.plan_frame(1024, Feedback::High);
        let par_bytes: u16 = specs
            .iter()
            .filter(|s| s.is_parity)
            .map(|s| s.payload_len)
            .sum();
        assert_eq!(
            par_bytes,
            4 * 256,
            "expected 4 parity stripes (ratio baseline + K_min) = {}B",
            4 * 256
        );
    }
}
