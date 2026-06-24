//! Bandwidth / redundancy prediction.

use alloc::collections::VecDeque;

use crate::config::Config;
use crate::feedback::{Feedback, FeedbackCodec};
use crate::loss::{LossMetrics, LossReport};

/// Input for [`BandwidthPredictor::recommend`].
///
/// The decoder emits [`LossReport`](crate::LossReport) via
/// [`DecoderEvent::LossReportReady`](crate::DecoderEvent::LossReportReady).
/// Prefer [`FeedbackManager`] to maintain `history` and `current_feedback` automatically.
pub struct RecommendContext<'a> {
    /// Metrics for the window that just closed.
    pub report: &'a LossReport,
    /// Prior windows, oldest first (length ≤ [`Config::feedback_history_len`]).
    pub history: &'a [LossReport],
    /// Redundancy level currently in use on the sender (last feedback sent).
    pub current_feedback: Feedback,
    /// Session parameters (read-only).
    pub config: &'a Config,
}

/// User-defined strategy for adaptive FEC bandwidth.
///
/// Shipped baselines: [`HighBandwidthPredictor`] (Tambur-full-BW) and
/// [`LowBandwidthPredictor`] (Tambur-low-BW heuristic without ML weights).
pub trait BandwidthPredictor {
    /// Choose the next redundancy level to advertise to the sender.
    fn recommend(&mut self, ctx: &RecommendContext<'_>) -> Feedback;

    /// Called when a feedback window closes, before `recommend` (optional).
    fn on_window_closed(&mut self, _report: &LossReport) {}
}

/// Adaptive-redundancy state: loss-report history, predictor, and last sent feedback.
///
/// Wire this to [`DecoderEvent::LossReportReady`](crate::DecoderEvent::LossReportReady):
/// call [`Self::handle_report`], encode the returned [`Feedback`] with
/// [`Self::encode_wire`], and send it to the encoder.
pub struct FeedbackManager<P> {
    predictor: P,
    history: VecDeque<LossReport>,
    current_feedback: Feedback,
    config: Config,
    codec: FeedbackCodec,
}

impl<P: BandwidthPredictor> FeedbackManager<P> {
    /// New manager with [`Feedback::None`] as the assumed sender redundancy.
    pub fn new(predictor: P, config: Config) -> Self {
        let codec = FeedbackCodec::new(&config);
        Self {
            predictor,
            history: VecDeque::new(),
            current_feedback: Feedback::default(),
            config,
            codec,
        }
    }

    /// Same as [`Self::new`], but seed sender redundancy (e.g. after session join).
    pub fn with_current_feedback(predictor: P, config: Config, current_feedback: Feedback) -> Self {
        let mut mgr = Self::new(predictor, config);
        mgr.current_feedback = current_feedback;
        mgr
    }

    /// Last redundancy level returned by [`Self::handle_report`] (or the seeded value).
    pub fn current_feedback(&self) -> Feedback {
        self.current_feedback
    }

    /// Session configuration (read-only).
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Borrow the wrapped predictor (custom policies, inspection).
    pub fn predictor(&self) -> &P {
        &self.predictor
    }

    /// Mutable access to the wrapped predictor.
    pub fn predictor_mut(&mut self) -> &mut P {
        &mut self.predictor
    }

    /// Encode a [`Feedback`] to its one-byte wire value using this session's codec.
    pub fn encode_wire(&self, feedback: Feedback) -> u8 {
        self.codec.encode(feedback)
    }

    /// Process a closed feedback window: update history, run the predictor, return next level.
    ///
    /// Injects [`LossMetrics::redundancy_wire_byte`](crate::LossMetrics::redundancy_wire_byte)
    /// from [`Self::current_feedback`] (the decoder leaves it at `0`).
    pub fn handle_report(&mut self, mut report: LossReport) -> Feedback {
        report.metrics.redundancy_wire_byte = f32::from(self.codec.encode(self.current_feedback));

        self.predictor.on_window_closed(&report);

        let history = self.history.make_contiguous();
        let ctx = RecommendContext {
            report: &report,
            history,
            current_feedback: self.current_feedback,
            config: &self.config,
        };

        let new_feedback = self.predictor.recommend(&ctx);

        self.history.push_back(report);
        while self.history.len() > self.config.feedback_history_len().get() {
            self.history.pop_front();
        }

        self.current_feedback = new_feedback;
        new_feedback
    }
}

/// Tambur-full-BW / `StreamingCode_3-high-BW`.
///
/// Any packet loss in the window → [`Feedback::High`] (50% parity).
/// No loss → [`Feedback::None`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HighBandwidthPredictor;

impl BandwidthPredictor for HighBandwidthPredictor {
    fn recommend(&mut self, ctx: &RecommendContext<'_>) -> Feedback {
        if ctx.report.metrics.loss_fraction > 0.0 {
            Feedback::High
        } else {
            Feedback::None
        }
    }
}

/// Tambur-low-BW heuristic (without `Tambur-low-BW.pt`).
///
/// Replaces the ML second stage:
/// after loss is detected, prefer [`Feedback::Low`] (25%) when multi-frame guard
/// space is sufficient, otherwise [`Feedback::High`] (50%).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LowBandwidthPredictor;

impl LowBandwidthPredictor {
    /// True when metrics indicate insufficient guard space for low bandwidth.
    pub fn needs_high_bandwidth(metrics: &LossMetrics) -> bool {
        metrics.multi_frame_insufficient_guardspace > 0.0
    }
}

impl BandwidthPredictor for LowBandwidthPredictor {
    fn recommend(&mut self, ctx: &RecommendContext<'_>) -> Feedback {
        let metrics = &ctx.report.metrics;
        if metrics.loss_fraction <= 0.0 {
            return Feedback::None;
        }
        if Self::needs_high_bandwidth(metrics) {
            Feedback::High
        } else {
            Feedback::Low
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::num::NonZeroUsize;

    use crate::loss::LossMetrics;

    fn ctx(metrics: LossMetrics) -> (Config, LossReport) {
        let config = Config::builder().build().unwrap();
        let report = LossReport {
            metrics,
            packet_losses: Vec::new(),
            frame_losses: Vec::new(),
        };
        (config, report)
    }

    fn recommend<P: BandwidthPredictor>(
        predictor: &mut P,
        report: &LossReport,
        config: &Config,
    ) -> Feedback {
        let history: &[LossReport] = &[];
        predictor.recommend(&RecommendContext {
            report,
            history,
            current_feedback: Feedback::None,
            config,
        })
    }

    #[test]
    fn high_bandwidth_no_loss() {
        let mut p = HighBandwidthPredictor;
        let (config, report) = ctx(LossMetrics::default());
        assert_eq!(recommend(&mut p, &report, &config), Feedback::None);
    }

    #[test]
    fn high_bandwidth_on_any_loss() {
        let mut p = HighBandwidthPredictor;
        let (config, mut report) = ctx(LossMetrics::default());
        report.metrics.loss_fraction = 0.01;
        assert_eq!(recommend(&mut p, &report, &config), Feedback::High);
    }

    #[test]
    fn low_bandwidth_no_loss() {
        let mut p = LowBandwidthPredictor;
        let (config, report) = ctx(LossMetrics::default());
        assert_eq!(recommend(&mut p, &report, &config), Feedback::None);
    }

    #[test]
    fn low_bandwidth_sufficient_guardspace() {
        let mut p = LowBandwidthPredictor;
        let (config, mut report) = ctx(LossMetrics::default());
        report.metrics.loss_fraction = 0.1;
        report.metrics.multi_frame_insufficient_guardspace = 0.0;
        assert_eq!(recommend(&mut p, &report, &config), Feedback::Low);
    }

    #[test]
    fn low_bandwidth_insufficient_guardspace() {
        let mut p = LowBandwidthPredictor;
        let (config, mut report) = ctx(LossMetrics::default());
        report.metrics.loss_fraction = 0.1;
        report.metrics.multi_frame_insufficient_guardspace = 0.5;
        assert_eq!(recommend(&mut p, &report, &config), Feedback::High);
    }

    #[test]
    fn feedback_manager_injects_redundancy_wire_byte() {
        struct WireByteCheck {
            seen: Option<f32>,
        }
        impl BandwidthPredictor for WireByteCheck {
            fn recommend(&mut self, ctx: &RecommendContext<'_>) -> Feedback {
                self.seen = Some(ctx.report.metrics.redundancy_wire_byte);
                Feedback::None
            }
        }

        let config = Config::builder().build().unwrap();
        let mut mgr = FeedbackManager::with_current_feedback(
            WireByteCheck { seen: None },
            config,
            Feedback::High,
        );
        let report = LossReport {
            metrics: LossMetrics::default(),
            packet_losses: Vec::new(),
            frame_losses: Vec::new(),
        };
        mgr.handle_report(report);
        assert_eq!(mgr.predictor().seen, Some(1.0));
    }

    #[test]
    fn feedback_manager_trims_history() {
        struct HistoryRecorder {
            lens: Vec<usize>,
        }
        impl BandwidthPredictor for HistoryRecorder {
            fn recommend(&mut self, ctx: &RecommendContext<'_>) -> Feedback {
                self.lens.push(ctx.history.len());
                Feedback::None
            }
        }

        let config = Config::builder()
            .feedback_history_len(NonZeroUsize::new(2).unwrap())
            .build()
            .unwrap();
        let mut mgr = FeedbackManager::new(HistoryRecorder { lens: Vec::new() }, config);
        for _ in 0..4 {
            mgr.handle_report(LossReport {
                metrics: LossMetrics::default(),
                packet_losses: Vec::new(),
                frame_losses: Vec::new(),
            });
        }
        assert_eq!(mgr.predictor().lens, [0, 1, 2, 2]);
    }

    #[test]
    fn feedback_manager_updates_current_feedback() {
        let config = Config::builder().build().unwrap();
        let mut mgr = FeedbackManager::new(HighBandwidthPredictor, config);
        assert_eq!(mgr.current_feedback(), Feedback::None);
        let mut report = LossReport {
            metrics: LossMetrics::default(),
            packet_losses: Vec::new(),
            frame_losses: Vec::new(),
        };
        report.metrics.loss_fraction = 0.1;
        assert_eq!(mgr.handle_report(report), Feedback::High);
        assert_eq!(mgr.current_feedback(), Feedback::High);
    }
}
