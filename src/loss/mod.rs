//! Loss statistics for quality reporting and predictors.

mod computer;
mod info;
mod metrics;

pub(crate) use computer::LossComputer;
pub(crate) use metrics::LossMetricComputer;

/// Aggregated loss metrics for one feedback window (~2s in Tambur default).
///
/// Field layout from Tambur `LossMetrics`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LossMetrics {
    /// RFC burst density over packet loss bitmap.
    pub burst_density: f32,
    /// RFC burst density over frame loss bitmap.
    pub frame_burst_density: f32,
    /// RFC gap density over packet loss bitmap.
    pub gap_density: f32,
    /// RFC gap density over frame loss bitmap.
    pub frame_gap_density: f32,
    /// Mean packet burst length in the window.
    pub mean_burst_length: f32,
    /// Mean frame burst length in the window.
    pub mean_frame_burst_length: f32,
    /// Fraction of packets lost in the window.
    pub loss_fraction: f32,
    /// Fraction of frames with at least one lost packet.
    pub frame_loss_fraction: f32,
    /// Fraction of multi-frame burst events.
    pub multi_frame_loss_fraction: f32,
    /// Mean guard-space length between packet bursts.
    pub mean_guardspace_length: f32,
    /// Mean guard-space length between frame bursts.
    pub mean_frame_guardspace_length: f32,
    /// Indicator of insufficient multi-frame guard space (Tambur low-BW heuristic input).
    pub multi_frame_insufficient_guardspace: f32,
    /// Wire byte of the active [`Feedback`](crate::Feedback) level as `f32` (Tambur ML feature #13).
    pub redundancy_wire_byte: f32,
}

/// One closed observation window produced by the decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct LossReport {
    /// Aggregated scalar metrics for predictors.
    pub metrics: LossMetrics,
    /// Per-packet loss bitmap: `true` = lost.
    pub packet_losses: Vec<bool>,
    /// Per-frame loss bitmap: `true` = at least one packet lost.
    pub frame_losses: Vec<bool>,
}

#[cfg(test)]
mod test_common;
