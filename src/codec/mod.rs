//! Public encoder and decoder session types.

mod decoder;
mod encoder;
mod receive_frame;
mod seq_dedup;
mod session;
mod window;

use crate::config::Config;
use crate::error::{CodecError, Error, InternalError, Result};
use crate::fec::{
    BlockCode, CodingMatrixInfo, FecError, MultiFecHeaderCode, StreamingCode, header_matrix_config,
    num_frames_for_delay,
};

pub use decoder::{Decoder, DecoderEvent, ReceiveOutcome};
pub use encoder::Encoder;
pub(crate) use receive_frame::ReceiveFrame;
pub(crate) use window::frame_expired;

impl From<FecError> for Error {
    fn from(err: FecError) -> Self {
        match err {
            FecError::FrameNotRecovered { frame_num } => {
                Self::Codec(CodecError::FrameNotRecovered {
                    frame_num: u64::from(frame_num),
                })
            }
            // FEC internals are not part of the stable public error surface.
            _other => Self::Internal(InternalError::InvariantViolated),
        }
    }
}

impl Config {
    pub(crate) fn streaming_code(&self) -> Result<StreamingCode> {
        let delay = u16::from(self.tau);
        let stripe_size = self.stripe_size() as u16;
        let num_frames = num_frames_for_delay(delay);
        let info = CodingMatrixInfo::new(
            self.max_fec_stripes * num_frames,
            self.max_data_stripes.get() * num_frames,
            self.w,
        )?;
        let block = BlockCode::new(info, delay, (1, 0), self.packet_size.get())?;
        StreamingCode::new(
            delay,
            stripe_size,
            block,
            u16::from(self.w),
            self.max_data_stripes.get(),
            self.max_fec_stripes,
        )
        .map_err(Into::into)
    }

    pub(crate) fn header_code(&self) -> Result<MultiFecHeaderCode> {
        let delay = u16::from(self.tau);
        let (_, max_pkts) = header_matrix_config(delay)?;
        MultiFecHeaderCode::new(delay, max_pkts).map_err(Into::into)
    }
}

/// Result of [`Decoder::receive_datagram`].
///
/// These are not errors — duplicates and late packets are normal on unreliable
/// transports and do not corrupt decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReceiveStatus {
    /// Datagram was accepted and may contribute to recovery.
    Accepted,
    /// Same sequence number already seen; ignored.
    Duplicate,
    /// Source frame is outside the current coding window; ignored.
    OutOfWindow,
}

impl ReceiveStatus {
    /// `true` if the datagram was ingested into decoder state.
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted)
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use super::*;
    use crate::datagram::FecDatagram;
    use crate::feedback::Feedback;
    use crate::frame::RecoveredFrame;
    use crate::predictor::{BandwidthPredictor, RecommendContext};
    use bytes::Bytes;

    struct OnLossHigh;

    impl BandwidthPredictor for OnLossHigh {
        fn recommend(&mut self, ctx: &RecommendContext<'_>) -> Feedback {
            if ctx.report.metrics.loss_fraction > 0.0 {
                Feedback::High
            } else {
                Feedback::None
            }
        }
    }

    fn recv(dec: &mut Decoder, pkt: FecDatagram, now: Duration) -> ReceiveOutcome {
        dec.receive_datagram(pkt, now)
    }

    fn recovered_from(events: &[DecoderEvent]) -> Option<RecoveredFrame> {
        events.iter().find_map(|e| match e {
            DecoderEvent::FrameRecovered(f) => Some(f.clone()),
            DecoderEvent::LossReportReady(_) => None,
        })
    }

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

    #[test]
    fn encode_decode_no_loss() {
        let config = test_config();
        let mut enc = Encoder::new(config.clone()).unwrap();
        enc.apply_feedback(Feedback::High);

        let mut dec = Decoder::new(config).unwrap();
        let payload = Bytes::from(vec![0xABu8; 512]);

        let pkts = enc.encode_payload(payload.clone()).unwrap();

        let mut recovered = None;
        for pkt in pkts {
            let outcome = recv(&mut dec, pkt.clone(), Duration::ZERO);
            assert_eq!(outcome.status, ReceiveStatus::Accepted);
            if let Some(frame) = recovered_from(&outcome.events) {
                recovered = Some(frame);
            }
        }

        let recovered = recovered.expect("frame recovered");
        assert_eq!(recovered.frame_num, 0);
        assert_eq!(recovered.payload, payload);
        assert!(recovered.direct_reception);
    }

    #[test]
    fn encode_decode_with_one_data_loss() {
        let config = test_config();
        let mut enc = Encoder::new(config.clone()).unwrap();
        enc.apply_feedback(Feedback::High);

        let mut dec = Decoder::new(config).unwrap();
        let payload = Bytes::from(vec![0xCDu8; 1536]);

        let pkts = enc.encode_payload(payload.clone()).unwrap();

        let skip = pkts
            .iter()
            .position(|p| !p.is_parity && p.pos_in_frame > 0)
            .expect("at least one non-first data pkt");
        let mut recovered = None;
        for (i, pkt) in pkts.iter().enumerate() {
            if i == skip {
                continue;
            }
            let outcome = recv(&mut dec, pkt.clone(), Duration::ZERO);
            assert_eq!(outcome.status, ReceiveStatus::Accepted);
            if let Some(frame) = recovered_from(&outcome.events) {
                recovered = Some(frame);
            }
        }

        let recovered = recovered.expect("frame recovered");
        assert_eq!(recovered.payload, payload);
        assert!(!recovered.direct_reception);
    }

    #[test]
    fn duplicate_datagram_ignored() {
        let config = test_config();
        let mut enc = Encoder::new(config.clone()).unwrap();
        let mut dec = Decoder::new(config).unwrap();
        let pkts = enc.encode_payload(Bytes::from(vec![1u8; 256])).unwrap();
        assert_eq!(
            recv(&mut dec, pkts[0].clone(), Duration::ZERO).status,
            ReceiveStatus::Accepted
        );
        assert_eq!(
            recv(&mut dec, pkts[0].clone(), Duration::ZERO).status,
            ReceiveStatus::Duplicate
        );
    }

    #[test]
    fn feedback_on_loss() {
        // Match C++ QualityReporter semantics: loss is reported only for frames
        // that already left the coding window (purged). With `tau = 3` the
        // window is 7 frames; we send 10 frames with one dropped non-first data
        // packet in frame 0 — by frame 9 the lost frame has been purged and
        // contributes a packet loss to the feedback report.
        let mut config = test_config();
        config.feedback_interval = Duration::from_secs(3600);
        let mut enc = Encoder::new(config.clone()).unwrap();
        enc.apply_feedback(Feedback::None);
        let mut dec = Decoder::new(config.clone()).unwrap();
        let predictor = OnLossHigh;

        let total_frames = 10u16;
        for frame_idx in 0..total_frames {
            let pkts = enc.encode_payload(Bytes::from(vec![2u8; 1536])).unwrap();
            if frame_idx == 0 {
                let skip = pkts
                    .iter()
                    .position(|p| !p.is_parity && p.pos_in_frame == 0)
                    .expect("first data pkt");
                for (i, pkt) in pkts.iter().enumerate() {
                    if i != skip {
                        recv(&mut dec, pkt.clone(), Duration::ZERO);
                    }
                }
            } else {
                for pkt in pkts {
                    recv(&mut dec, pkt.clone(), Duration::ZERO);
                }
            }
        }

        let events = dec.poll(Duration::from_secs(3600));
        let report = events
            .into_iter()
            .find_map(|e| match e {
                DecoderEvent::LossReportReady(report) => Some(report),
                DecoderEvent::FrameRecovered(_) => None,
            })
            .expect("loss report");
        let mut feedback_mgr = crate::predictor::FeedbackManager::new(predictor, config.clone());
        let fb = feedback_mgr.handle_report(report);
        assert_eq!(fb, Feedback::High);
    }
}
