//! FEC receiver session: datagrams into recovered frames and periodic feedback.

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::time::Duration;

use bytes::Bytes;

use crate::config::Config;
use crate::datagram::FecDatagram;
use crate::error::Result;
use crate::fec::{MultiFecHeaderCode, StreamingCode};
use crate::frame::RecoveredFrame;
use crate::loss::{LossComputer, LossMetricComputer, LossReport};

use super::receive_frame::ReceiveFrame;
use super::seq_dedup::SeqDedup;
use super::session::{self, SessionIndexTracker, wire_frame_num};
use super::{ReceiveStatus, frame_expired};

fn frame_size_as_u32(size: u64) -> u32 {
    size.min(u64::from(u32::MAX)) as u32
}

/// Side effect produced while advancing the decoder (recovery or closed window).
#[derive(Debug, Clone, PartialEq)]
pub enum DecoderEvent {
    /// A source frame became recoverable.
    FrameRecovered(RecoveredFrame),
    /// A feedback observation window closed; call [`FeedbackManager::handle_report`](crate::FeedbackManager::handle_report)
    /// (or your own predictor) and send feedback to the sender.
    LossReportReady(LossReport),
}

/// Result of [`Decoder::receive_datagram`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiveOutcome {
    /// Whether the datagram was ingested into decoder state.
    pub status: ReceiveStatus,
    /// Recovery and feedback events produced by this call (and any elapsed time).
    pub events: Vec<DecoderEvent>,
}

/// FEC receiver: datagrams → recovered frames + periodic feedback.
pub struct Decoder {
    config: Config,
    code: StreamingCode,
    header_code: MultiFecHeaderCode,
    window_size: u16,
    frames_ring: Vec<ReceiveFrame>,
    feedback_frames: VecDeque<ReceiveFrame>,
    seen_seq: SeqDedup,
    session: SessionIndexTracker,
    received_first: bool,
    last_session_index: u64,
    last_loss_report: Option<LossReport>,
    /// Monotonic time when feedback was last emitted ([`Duration::ZERO`] before the first tick).
    last_feedback_at: Duration,
    frame_sizes_scratch: Vec<Option<u64>>,
}

impl Decoder {
    /// Create a decoder for the given validated session configuration.
    pub fn new(config: Config) -> Result<Self> {
        let window_size = config.window_frames();
        let mut frames_ring = Vec::with_capacity(window_size as usize);
        for _ in 0..window_size {
            frames_ring.push(ReceiveFrame::new(0, 0));
        }
        Ok(Self {
            config: config.clone(),
            code: config.streaming_code()?,
            header_code: config.header_code()?,
            window_size,
            frames_ring,
            feedback_frames: VecDeque::new(),
            seen_seq: SeqDedup::default(),
            session: SessionIndexTracker::default(),
            received_first: false,
            last_session_index: 0,
            last_loss_report: None,
            last_feedback_at: Duration::ZERO,
            frame_sizes_scratch: Vec::new(),
        })
    }

    /// Session configuration (read-only).
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Ingest one parsed datagram and collect recovery / feedback events.
    ///
    /// `now` is a monotonic timestamp from the embedder (e.g. OS uptime). The decoder
    /// compares it against [`Config::feedback_interval`] to emit [`DecoderEvent::LossReportReady`].
    pub fn receive_datagram(&mut self, datagram: FecDatagram, now: Duration) -> ReceiveOutcome {
        let status = self.ingest_datagram(datagram);
        let events = self.emit_pending(now);
        ReceiveOutcome { status, events }
    }

    /// Advance time without ingesting a datagram (e.g. feedback timer while idle).
    pub fn poll(&mut self, now: Duration) -> Vec<DecoderEvent> {
        self.emit_pending(now)
    }

    /// Metrics for the most recently closed window (debug / custom predictors).
    pub fn last_loss_report(&self) -> Option<&LossReport> {
        self.last_loss_report.as_ref()
    }

    fn slot_index(&self, session_index: u64) -> usize {
        (session_index % u64::from(self.window_size)) as usize
    }

    fn frame_occupied(&self, session_index: u64) -> bool {
        let slot = &self.frames_ring[self.slot_index(session_index)];
        slot.is_occupied() && slot.session_index == session_index
    }

    fn min_session_in_window(&self) -> Option<u64> {
        self.frames_ring
            .iter()
            .filter(|f| f.is_occupied())
            .map(|f| f.session_index)
            .min()
    }

    fn frame_mut(&mut self, session_index: u64, wire: u16) -> &mut ReceiveFrame {
        let idx = self.slot_index(session_index);
        let slot = &mut self.frames_ring[idx];
        if !slot.is_occupied() || slot.session_index != session_index {
            slot.reset_for(session_index, wire, false);
        }
        slot
    }

    fn frame_get_mut(&mut self, session_index: u64) -> Option<&mut ReceiveFrame> {
        if !self.frame_occupied(session_index) {
            return None;
        }
        let idx = self.slot_index(session_index);
        Some(&mut self.frames_ring[idx])
    }

    fn ingest_datagram(&mut self, datagram: FecDatagram) -> ReceiveStatus {
        if !self.seen_seq.insert(datagram.seq_num) {
            return ReceiveStatus::Duplicate;
        }

        let session_index = self.session.resolve_wire(datagram.frame_num);
        let wire = datagram.frame_num;

        let window = self.window_size;
        if self.received_first
            && session::is_out_of_window_session(session_index, self.last_session_index, window)
        {
            self.seen_seq.remove(datagram.seq_num);
            return ReceiveStatus::OutOfWindow;
        }

        if self.received_first {
            let idx = self.slot_index(session_index);
            if self.frames_ring[idx].is_occupied()
                && self.frames_ring[idx].session_index != session_index
                && session_index < self.frames_ring[idx].session_index
            {
                self.seen_seq.remove(datagram.seq_num);
                return ReceiveStatus::OutOfWindow;
            }
        }

        self.purge_expired(session_index);
        self.fill_missing_frames(session_index);

        if !self.frame_occupied(session_index) {
            self.frame_mut(session_index, wire);
        }

        self.decode_headers(&datagram, session_index);

        if self.code.add_packet(&datagram, wire).is_err() {
            self.seen_seq.remove(datagram.seq_num);
            return ReceiveStatus::OutOfWindow;
        }

        self.frame_mut(session_index, wire).add_pkt(datagram);

        let _ = self.try_decode();

        self.received_first = true;
        if session_index > self.last_session_index {
            self.last_session_index = session_index;
        }
        ReceiveStatus::Accepted
    }

    fn emit_pending(&mut self, now: Duration) -> Vec<DecoderEvent> {
        let mut events = self.drain_recovered();
        if let Some(report) = self.maybe_emit_loss_report(now) {
            events.push(DecoderEvent::LossReportReady(report));
        }
        events
    }

    fn drain_recovered(&mut self) -> Vec<DecoderEvent> {
        let mut events = Vec::new();
        while let Some(frame) = self.take_recovered_frame() {
            events.push(DecoderEvent::FrameRecovered(frame));
        }
        events
    }

    fn take_recovered_frame(&mut self) -> Option<RecoveredFrame> {
        let candidate = self.frames_ring.iter().find_map(|frame| {
            if !frame.is_occupied() || frame.is_delivered() {
                return None;
            }
            let session_index = frame.session_index;
            let wire = frame.frame_num;
            let size = frame.frame_size?;
            let size_u32 = frame_size_as_u32(size);
            if !self.code.is_frame_recovered(wire, size_u32) {
                return None;
            }
            Some((session_index, wire, size_u32, frame.all_data_received()))
        })?;

        let (session_index, wire, size_u32, direct_reception) = candidate;
        let payload = self.code.recovered_frame(wire, size_u32).ok()?;
        let idx = self.slot_index(session_index);
        let frame = &mut self.frames_ring[idx];
        frame.mark_delivered(!direct_reception);
        Some(RecoveredFrame {
            frame_num: session_index,
            payload: Bytes::from(payload),
            direct_reception,
        })
    }

    fn maybe_emit_loss_report(&mut self, now: Duration) -> Option<LossReport> {
        if !self.feedback_due(now) {
            return None;
        }
        self.last_feedback_at = now;

        let report = self.build_loss_report();
        self.last_loss_report = Some(report.clone());
        self.feedback_frames.clear();
        Some(report)
    }

    fn feedback_due(&self, now: Duration) -> bool {
        now.saturating_sub(self.last_feedback_at) >= self.config.feedback_interval()
    }

    fn fill_missing_frames(&mut self, session_index: u64) {
        if !self.received_first {
            return;
        }
        if session_index <= self.last_session_index + 1 {
            return;
        }
        let mut start = self.last_session_index + 1;
        while start < session_index {
            if !self.frame_occupied(start) {
                let idx = self.slot_index(start);
                self.frames_ring[idx].reset_for(start, wire_frame_num(start), true);
            }
            start += 1;
        }
    }

    fn purge_expired(&mut self, session_index: u64) {
        while let Some(expired) = self.min_session_in_window() {
            if !frame_expired(session_index, expired, self.window_size) {
                break;
            }
            self.purge_frame(expired);
        }
    }

    fn purge_frame(&mut self, session_index: u64) {
        let idx = self.slot_index(session_index);
        if self.frames_ring[idx].is_occupied()
            && self.frames_ring[idx].session_index == session_index
        {
            let seqs: Vec<u32> = self.frames_ring[idx].seq_nums().collect();
            let frame = self.frames_ring[idx].take_into_feedback();
            for seq in seqs {
                self.seen_seq.remove(seq);
            }
            self.feedback_frames.push_back(frame);
        }
    }

    fn decode_headers(&mut self, pkt: &FecDatagram, current_session: u64) {
        let Ok(recovered) = self.header_code.decode_sizes_of_frames(
            pkt.frame_num,
            pkt.pos_in_frame,
            pkt.sizes_of_frames_encoding,
        ) else {
            return;
        };
        let epoch = session::session_epoch(current_session);
        for (wire, size) in recovered {
            let mut session = epoch * session::FRAME_SPACE + u64::from(wire);
            if session > current_session
                && session - current_session > u64::from(session::HALF_FRAME_SPACE)
            {
                session -= session::FRAME_SPACE;
            }
            if let Some(frame) = self.frame_get_mut(session) {
                frame.update_size(size);
            }
        }
    }

    fn try_decode(&mut self) -> Result<()> {
        let window = self.window_size as usize;
        if self.frame_sizes_scratch.len() != window {
            self.frame_sizes_scratch.resize(window, None);
        }
        self.frame_sizes_scratch.fill(None);

        let ts = match self.code.stream_timeslot() {
            Some(ts) => ts,
            None => return Ok(()),
        };
        let num_frames = window as u16;
        let ts_mod = ts % num_frames;

        for frame in &self.frames_ring {
            if frame.is_occupied() {
                let w_mod = frame.frame_num % num_frames;
                let pos = ((w_mod + num_frames - ts_mod - 1) % num_frames) as usize;
                self.frame_sizes_scratch[pos] = frame.frame_size;
            }
        }

        self.code.decode(&self.frame_sizes_scratch)?;

        for frame in &mut self.frames_ring {
            if frame.is_occupied()
                && let Some(size) = frame.frame_size
                && self
                    .code
                    .is_frame_recovered(frame.frame_num, frame_size_as_u32(size))
            {
                frame.set_decoded();
            }
        }
        Ok(())
    }

    fn build_loss_report(&self) -> LossReport {
        // Match C++ `QualityReporter::generate_quality_report`: only frames that
        // were already purged (and thus had a final recovery decision) feed the
        // loss computer. In-flight frames may still be recovered by future
        // parities, so reporting them as lost would skew the predictor.
        let loss_info = LossComputer::new(&self.feedback_frames, true).loss_info();
        let mut metrics =
            LossMetricComputer::new(loss_info.clone(), 0, 1, 3, 2, self.config.tau.get())
                .compute_loss_metrics();
        metrics.redundancy_wire_byte = 0.0;

        LossReport {
            metrics,
            packet_losses: loss_info.packet_losses,
            frame_losses: loss_info.frame_losses,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use super::*;
    use crate::codec::Encoder;
    use crate::feedback::Feedback;
    use bytes::Bytes;

    fn test_config() -> Config {
        Config::builder()
            .tau(crate::config::Tau::new(3).unwrap())
            .w(crate::WordSize::W32)
            .packet_size(core::num::NonZeroU16::new(8).unwrap())
            .max_data_stripes(core::num::NonZeroU16::new(16).unwrap())
            .max_fec_stripes(16)
            .max_frame_size(4096)
            .feedback_interval(Duration::from_millis(0))
            .build()
            .unwrap()
    }

    fn recv(dec: &mut Decoder, pkt: FecDatagram, now: Duration) -> ReceiveOutcome {
        dec.receive_datagram(pkt, now)
    }

    #[test]
    fn session_wrap_wire_frame_zero_at_boundary() {
        let config = test_config();
        let mut enc = Encoder::new(config).unwrap();
        enc.set_next_session_index(session::FRAME_SPACE);
        let pkts = enc.encode_payload(Bytes::from(vec![1u8; 128])).unwrap();
        assert_eq!(pkts[0].frame_num, 0);
    }

    #[test]
    fn slot_aliasing_across_wrap_with_loss() {
        let config = test_config();
        let mut enc = Encoder::new(config.clone()).unwrap();
        enc.apply_feedback(Feedback::High);
        let mut dec = Decoder::new(config).unwrap();

        // Start 4 frames before the wire-frame wrap (FRAME_SPACE = 32768).
        // Frames: 32764..32773. With wire-based slot indexing:
        //   32767 (wire 32767) → slot 0  32768 (wire 0) → slot 0 → COLLISION
        // With session-based indexing:
        //   32767 → slot 0     32768 → slot 1 → no collision
        let start = session::FRAME_SPACE - 4;
        enc.set_next_session_index(start);

        let total_frames: u64 = 10;

        // Track all recovered sessions
        let mut recovered: BTreeSet<u64> = BTreeSet::new();

        for i in 0..total_frames {
            let session = start + i;
            let payload = Bytes::from(vec![(session % 256) as u8; 128]);
            let pkts = enc.encode_payload(payload).unwrap();

            for (j, pkt) in pkts.iter().enumerate() {
                // Drop the last packet of the two frames right before the wrap
                // (32766 and 32767) so they depend on post-wrap FEC parity.
                if (session == start + 2 || session == start + 3) && j == pkts.len() - 1 {
                    continue;
                }
                let outcome = recv(&mut dec, pkt.clone(), Duration::ZERO);
                for event in outcome.events {
                    if let DecoderEvent::FrameRecovered(frame) = event {
                        recovered.insert(frame.frame_num);
                    }
                }
            }
        }

        // Final poll to drain any pending recovered frames
        let events = dec.poll(Duration::from_millis(100));
        for event in events {
            if let DecoderEvent::FrameRecovered(frame) = event {
                recovered.insert(frame.frame_num);
            }
        }

        // All 10 frames must be recovered — with the slot-aliasing bug,
        // the pre-wrap frames get their slot overwritten by post-wrap frames
        // and become unrecoverable despite sufficient FEC capacity.
        assert_eq!(
            recovered.len(),
            total_frames as usize,
            "all {} frames should be recovered despite wire wrap; missing: {:?}",
            total_frames,
            (start..start + total_frames)
                .filter(|s| !recovered.contains(s))
                .collect::<Vec<_>>()
        );
        // Verify monotonicity
        let mut prev: Option<u64> = None;
        for &s in &recovered {
            if let Some(p) = prev {
                assert!(s > p, "sessions must be recovered in order");
            }
            prev = Some(s);
        }
    }

    #[test]
    fn many_frames_monotonic_session_index() {
        let config = test_config();
        let mut enc = Encoder::new(config.clone()).unwrap();
        enc.apply_feedback(Feedback::None);
        let mut dec = Decoder::new(config).unwrap();

        let total = session::FRAME_SPACE + 50;
        let mut last_session = None;
        for i in 0..total {
            let payload = Bytes::from(vec![(i % 256) as u8; 128]);
            let pkts = enc.encode_payload(payload.clone()).unwrap();
            for pkt in pkts {
                let outcome = recv(&mut dec, pkt.clone(), Duration::ZERO);
                for event in outcome.events {
                    if let DecoderEvent::FrameRecovered(frame) = event {
                        if let Some(prev) = last_session {
                            assert!(frame.frame_num > prev);
                        }
                        last_session = Some(frame.frame_num);
                        assert_eq!(frame.frame_num, i);
                    }
                }
            }
        }
        assert_eq!(last_session, Some(total - 1));
    }
}
