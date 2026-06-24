//! End-to-end session harness for Criterion (`benches/e2e.rs`).
//!
//! Simulates UDP-style I/O: each datagram is serialized to a contiguous buffer
//! ([`FecDatagram::to_bytes`]) and parsed back ([`FecDatagram::from_bytes`]) before
//! ingest. Feedback bytes round-trip through [`FeedbackCodec`] the same way.

#![cfg_attr(not(feature = "bench"), allow(unreachable_pub))]

use alloc::vec::Vec;
use core::num::NonZeroU16;
use core::time::Duration;

use bytes::Bytes;

use crate::codec::{Decoder, DecoderEvent, Encoder};
use crate::config::{Config, Tau};
use crate::datagram::FecDatagram;
use crate::feedback::{Feedback, FeedbackCodec};
use crate::predictor::{FeedbackManager, HighBandwidthPredictor};
use crate::word_size::WordSize;

/// Session parameters for [`E2eBenchSession`].
#[derive(Debug, Clone, Copy)]
pub struct E2eBenchParams {
    /// Source frames in the main loop (before tail flush).
    pub num_main_frames: u16,
    /// Extra frames after the main loop to drain the sliding window (usually `2τ+1`).
    pub tail_frames: u16,
    /// Fixed application payload size per frame.
    pub frame_size: usize,
    /// Per-packet drop probability in `[0.0, 1.0]` (Bernoulli, seeded RNG).
    pub loss_probability: f32,
    /// Monotonic clock step after each source frame.
    pub frame_interval: Duration,
}

impl E2eBenchParams {
    /// One source frame — Criterion `time` ≈ CPU cost per frame (encode + wire + decode).
    pub fn single_frame(frame_size: usize, loss_probability: f32) -> Self {
        Self {
            num_main_frames: 1,
            tail_frames: 0,
            frame_size,
            loss_probability,
            frame_interval: Duration::from_millis(33),
        }
    }

    /// Short multi-frame session; fewer frames for large payloads and loss (faster benches).
    pub fn short_session(frame_size: usize, loss_probability: f32) -> Self {
        let tail_frames = 7; // 2×τ+1 for τ=3
        let num_main_frames = if loss_probability > 0.0 {
            if frame_size >= 16 * 1024 { 6 } else { 8 }
        } else if frame_size >= 48 * 1024 {
            8
        } else if frame_size >= 16 * 1024 {
            10
        } else {
            12
        };
        Self {
            num_main_frames,
            tail_frames,
            frame_size,
            loss_probability,
            frame_interval: Duration::from_millis(33),
        }
    }

    /// Total source frames in [`E2eBenchSession::run`].
    pub fn total_frames(self) -> u16 {
        self.num_main_frames + self.tail_frames
    }
}

/// Counters returned by [`E2eBenchSession::run`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct E2eBenchStats {
    /// Source frames passed to [`Encoder::encode_payload`].
    pub frames_encoded: u16,
    /// [`DecoderEvent::FrameRecovered`] events observed.
    pub frames_recovered: u32,
    /// FEC datagrams emitted by the encoder (before loss).
    pub packets_sent: u32,
    /// Datagrams delivered over the simulated wire.
    pub packets_delivered: u32,
    /// Datagrams dropped by the loss model.
    pub packets_dropped: u32,
    /// Total bytes on the simulated wire (FEC packets only).
    pub wire_bytes: u64,
    /// Closed feedback windows handled via [`FeedbackManager`].
    pub feedback_rounds: u32,
}

/// Build a production-like [`Config`] for e2e benches.
///
/// Stripe caps are derived from `frame_size` so large frames (e.g. 48 KiB) do not
/// exceed [`Config::max_data_stripes`] / [`Config::max_fec_stripes`].
pub fn e2e_bench_config(tau: u8, frame_size: u32) -> Config {
    let stripe_size = u32::from(WordSize::W32) * 8;
    let num_data_stripes = frame_size.div_ceil(stripe_size).max(1) as u16;
    // [`Feedback::High`] emits 50% parity stripes.
    let max_fec_stripes = ((u32::from(num_data_stripes) + 1) / 2).max(1) as u16;

    Config::builder()
        .tau(Tau::new(tau).expect("tau in range"))
        .w(WordSize::W32)
        .packet_size(NonZeroU16::new(8).expect("non-zero"))
        .max_data_stripes(NonZeroU16::new(num_data_stripes).expect("non-zero data stripes"))
        .max_fec_stripes(max_fec_stripes)
        .max_frame_size(frame_size)
        .max_pkt_size(NonZeroU16::new(1500).expect("mtu"))
        .feedback_interval(Duration::from_millis(500))
        .high_redundancy_byte(1)
        .build()
        .expect("e2e bench config valid")
}

/// glibc-compatible `rand()` after `srand(0)`.
struct BenchRng {
    state: u32,
}

impl BenchRng {
    const fn seeded() -> Self {
        Self { state: 0 }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.state / 65536) % 32768
    }

    fn bernoulli(&mut self, p: f32) -> bool {
        if p <= 0.0 {
            return false;
        }
        if p >= 1.0 {
            return true;
        }
        let threshold = (p * 32768.0) as u32;
        self.next_u32() < threshold
    }
}

/// Full sender ↔ wire ↔ receiver session (one encode/decode pair per run).
pub struct E2eBenchSession {
    encoder: Encoder,
    decoder: Decoder,
    feedback_mgr: FeedbackManager<HighBandwidthPredictor>,
    feedback_codec: FeedbackCodec,
    clock: Duration,
    rng: BenchRng,
    params: E2eBenchParams,
    payload_byte: u8,
}

impl E2eBenchSession {
    /// New session; encoder starts with [`Feedback::High`] until the first report.
    pub fn new(config: Config, params: E2eBenchParams, payload_byte: u8) -> Self {
        let feedback_codec = FeedbackCodec::new(&config);
        let mut encoder = Encoder::new(config.clone()).expect("encoder");
        encoder.apply_feedback(Feedback::High);
        Self {
            encoder,
            decoder: Decoder::new(config.clone()).expect("decoder"),
            feedback_mgr: FeedbackManager::with_current_feedback(
                HighBandwidthPredictor,
                config.clone(),
                Feedback::High,
            ),
            feedback_codec,
            clock: Duration::ZERO,
            rng: BenchRng::seeded(),
            params,
            payload_byte,
        }
    }

    /// Run main + tail frames; returns aggregate counters.
    pub fn run(&mut self) -> E2eBenchStats {
        let mut stats = E2eBenchStats::default();
        let total = self.params.total_frames();
        for frame_idx in 0..total {
            self.run_one_frame(frame_idx, &mut stats);
        }
        stats
    }

    /// Encode → wire → decode for **one** frame (see `e2e/one_frame` benches).
    pub fn run_single_frame(&mut self) -> E2eBenchStats {
        let mut stats = E2eBenchStats::default();
        self.run_one_frame(0, &mut stats);
        stats
    }

    fn run_one_frame(&mut self, frame_idx: u16, stats: &mut E2eBenchStats) {
        let payload = Bytes::from(vec![self.payload_byte; self.params.frame_size]);
        let pkts: Vec<FecDatagram> = self
            .encoder
            .encode_payload(payload)
            .expect("encode_payload")
            .to_vec();
        stats.frames_encoded = frame_idx + 1;

        for pkt in &pkts {
            stats.packets_sent += 1;
            if self.rng.bernoulli(self.params.loss_probability) {
                stats.packets_dropped += 1;
                continue;
            }
            self.deliver_datagram(pkt, stats);
            stats.packets_delivered += 1;
        }

        self.clock += self.params.frame_interval;
        let events = self.decoder.poll(self.clock);
        self.drain_decoder_events(events, stats);
    }

    /// Serialize like `sendto` / parse like `recvfrom` on a UDP socket.
    fn deliver_datagram(&mut self, pkt: &FecDatagram, stats: &mut E2eBenchStats) {
        let wire = pkt.to_bytes().expect("to_bytes");
        stats.wire_bytes += wire.len() as u64;
        let parsed = FecDatagram::from_bytes(wire).expect("from_bytes");
        let outcome = self.decoder.receive_datagram(parsed, self.clock);
        self.drain_decoder_events(outcome.events, stats);
    }

    fn drain_decoder_events(&mut self, events: Vec<DecoderEvent>, stats: &mut E2eBenchStats) {
        for event in events {
            match event {
                DecoderEvent::FrameRecovered(rec) => {
                    stats.frames_recovered += 1;
                    core::hint::black_box(rec.payload.as_ref());
                }
                DecoderEvent::LossReportReady(report) => {
                    let fb = self.feedback_mgr.handle_report(report);
                    self.apply_feedback_wire(fb);
                    stats.feedback_rounds += 1;
                }
            }
        }
    }

    fn apply_feedback_wire(&mut self, fb: Feedback) {
        let wire = self.feedback_codec.encode_bytes(fb);
        let decoded = self
            .feedback_codec
            .decode_bytes(&wire)
            .expect("feedback wire roundtrip");
        self.encoder.apply_feedback(decoded);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2e_bench_single_frame() {
        let frame_size = 4 * 1024;
        let params = E2eBenchParams::single_frame(frame_size, 0.0);
        let config = e2e_bench_config(3, frame_size as u32);
        let mut session = E2eBenchSession::new(config, params, 0x42);
        let stats = session.run_single_frame();
        assert_eq!(stats.frames_encoded, 1);
        assert_eq!(stats.packets_dropped, 0);
        assert!(stats.wire_bytes > 0);
    }

    #[test]
    fn e2e_bench_no_loss_recovers_all_frames() {
        let params = E2eBenchParams {
            num_main_frames: 10,
            tail_frames: 7,
            frame_size: 2_048,
            loss_probability: 0.0,
            frame_interval: Duration::from_millis(33),
        };
        let config = e2e_bench_config(3, params.frame_size as u32);
        let mut session = E2eBenchSession::new(config, params, 0x42);
        let stats = session.run();
        assert_eq!(stats.frames_encoded, 17);
        assert_eq!(stats.packets_dropped, 0);
        assert!(stats.frames_recovered >= u32::from(params.num_main_frames));
        assert!(stats.wire_bytes > 0);
    }

    #[test]
    fn e2e_bench_large_frame_48k() {
        let frame_size = 48 * 1024;
        let params = E2eBenchParams {
            num_main_frames: 5,
            tail_frames: 7,
            frame_size,
            loss_probability: 0.0,
            frame_interval: Duration::from_millis(33),
        };
        let config = e2e_bench_config(3, frame_size as u32);
        assert_eq!(config.max_data_stripes().get(), 192);
        let mut session = E2eBenchSession::new(config, params, 0x42);
        let stats = session.run();
        assert_eq!(stats.packets_dropped, 0);
        assert!(stats.wire_bytes > 0);
    }
}
