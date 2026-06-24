//! Session configuration and redundancy profiles.

use core::convert::TryFrom;
use core::num::{NonZeroU8, NonZeroU16, NonZeroUsize};
use core::time::Duration;

use crate::error::ConfigError;
use crate::packetization::{DEFAULT_MTU, num_data_stripes_for_frame_size, num_parity_stripes_high};
use crate::word_size::WordSize;

/// Latency deadline in source frames (`tau <= 8` in Tambur).
///
/// Controls the sliding coding window size (`2τ+1` frames), recovery latency,
/// and how long frames stay in the decoder before they can appear in a
/// [`LossReport`](crate::LossReport).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tau(u8);

impl Tau {
    /// Maximum supported latency deadline.
    pub const MAX: u8 = 8;

    /// Construct a validated τ, or `None` when out of range.
    pub const fn new(value: u8) -> Option<Self> {
        if value <= Self::MAX {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Raw τ value (always `<= [`Self::MAX`]).
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl From<Tau> for u8 {
    fn from(tau: Tau) -> Self {
        tau.get()
    }
}

impl From<Tau> for u16 {
    fn from(tau: Tau) -> Self {
        u16::from(tau.get())
    }
}

impl TryFrom<u8> for Tau {
    type Error = ConfigError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(ConfigError::TauOutOfRange {
            value,
            max: Self::MAX,
        })
    }
}

/// Parameters for streaming FEC (Tambur-style).
///
/// Both [`Encoder`](crate::Encoder) and [`Decoder`](crate::Decoder) must use the
/// same `Config` for a session. Build with [`Config::builder`].
///
/// See the crate-level **Configuration reference** for a detailed description of
/// each field and how it affects latency, memory, bandwidth, and wire behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub(crate) tau: Tau,
    pub(crate) w: WordSize,
    pub(crate) packet_size: NonZeroU16,
    pub(crate) max_data_stripes: NonZeroU16,
    pub(crate) max_fec_stripes: u16,
    pub(crate) max_frame_size: u32,
    pub(crate) max_pkt_size: NonZeroU16,
    pub(crate) feedback_interval: Duration,
    pub(crate) high_redundancy_byte: u8,
    pub(crate) redundancy_step_factor: NonZeroU8,
    pub(crate) feedback_history_len: NonZeroUsize,
    pub(crate) parity_delay: u16,
    pub(crate) min_window_parity: u16,
}

impl Config {
    /// Start building a validated configuration.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }

    /// Latency deadline τ in source frames.
    ///
    /// The decoder keeps a sliding window of [`window_frames`](Self::window_frames)
    /// (= `2τ+1`) source frames. Higher τ tolerates longer burst loss but adds
    /// end-to-end delay and memory; lower τ is tighter latency.
    ///
    /// Must match on encoder and decoder. Valid range: `1..=8` ([`Tau::MAX`]).
    pub fn tau(&self) -> Tau {
        self.tau
    }

    /// Galois-field word width for payload Jerasure (`W8` or `W32`).
    ///
    /// Together with [`packet_size`](Self::packet_size) sets [`stripe_size`](Self::stripe_size)
    /// (`w × packet_size`). Affects CPU (SIMD-friendly `W32` vs smaller `W8` stripes)
    /// and how payload bytes map to coding symbols.
    pub fn w(&self) -> WordSize {
        self.w
    }

    /// Jerasure symbol size in bytes (not IP MTU).
    ///
    /// Multiplied by [`w`](Self::w) to form [`stripe_size`](Self::stripe_size).
    /// Smaller values increase stripe count per frame; larger values reduce
    /// datagram header overhead but require `stripe_size ≤ max_pkt_size`.
    pub fn packet_size(&self) -> NonZeroU16 {
        self.packet_size
    }

    /// Upper bound on data stripes per source frame.
    ///
    /// Caps largest encodable frame (~`max_data_stripes × stripe_size`, also
    /// limited by [`max_frame_size`](Self::max_frame_size)). Drives Cauchy matrix
    /// width and per-frame datagram count.
    pub fn max_data_stripes(&self) -> NonZeroU16 {
        self.max_data_stripes
    }

    /// Upper bound on parity stripes per source frame.
    ///
    /// Caps redundancy at [`Feedback::High`](crate::Feedback::High) (50% parity).
    /// Too low a cap prevents the encoder from emitting enough parities for the
    /// requested feedback level on large frames.
    pub fn max_fec_stripes(&self) -> u16 {
        self.max_fec_stripes
    }

    /// Maximum application payload per `encode_payload` call (bytes).
    ///
    /// Payloads larger than this are rejected with
    /// [`CodecError::PayloadTooLarge`](crate::CodecError::PayloadTooLarge).
    pub fn max_frame_size(&self) -> u32 {
        self.max_frame_size
    }

    /// Maximum on-wire FEC datagram payload (bytes).
    ///
    /// Default 1500 (Tambur MTU). Building via [`Config::builder`] fails if
    /// [`stripe_size`](Self::stripe_size) exceeds this value. Set to your real
    /// UDP payload limit (path MTU minus headers in your stack).
    pub fn max_pkt_size(&self) -> NonZeroU16 {
        self.max_pkt_size
    }

    /// Minimum interval between [`DecoderEvent::LossReportReady`](crate::DecoderEvent::LossReportReady)
    /// emissions.
    ///
    /// Uses the monotonic `now` clock you pass to [`Decoder::receive_datagram`](crate::Decoder::receive_datagram)
    /// and [`Decoder::poll`](crate::Decoder::poll). Shorter intervals adapt
    /// redundancy faster but increase control traffic.
    pub fn feedback_interval(&self) -> Duration {
        self.feedback_interval
    }

    /// Wire byte for [`Feedback::High`](crate::Feedback::High) (default `1`).
    ///
    /// Used by [`FeedbackCodec`](crate::FeedbackCodec) for interoperability.
    /// Does not change parity ratios (still 50% at High).
    pub fn high_redundancy_byte(&self) -> u8 {
        self.high_redundancy_byte
    }

    /// Multiplier for the low-redundancy wire byte (default `2`).
    ///
    /// [`low_redundancy_byte`](Self::low_redundancy_byte) =
    /// `high_redundancy_byte × redundancy_step_factor` (saturated to `u8::MAX`).
    pub fn redundancy_step_factor(&self) -> NonZeroU8 {
        self.redundancy_step_factor
    }

    /// Recommended depth of prior [`LossReport`](crate::LossReport)s for predictors.
    ///
    /// [`FeedbackManager`](crate::FeedbackManager) enforces this limit on its history deque.
    /// When calling [`RecommendContext`](crate::RecommendContext) manually, trim yourself.
    pub fn feedback_history_len(&self) -> NonZeroUsize {
        self.feedback_history_len
    }

    /// Frames to defer parity stripe emission (default `0`).
    ///
    /// Non-zero shifts when parity packets appear on the wire relative to their
    /// source frame. Encoder and decoder must agree.
    pub fn parity_delay(&self) -> u16 {
        self.parity_delay
    }

    /// Minimum total parity stripes in the sliding coding window (`K_min`).
    ///
    /// When the sum of parity stripes over the last [`window_frames`](Self::window_frames)
    /// emitted frames is below this threshold, the next frame's parity is topped up
    /// (up to [`max_fec_stripes`](Self::max_fec_stripes)). `0` disables the check.
    pub fn min_window_parity(&self) -> u16 {
        self.min_window_parity
    }

    /// Wire byte for [`Feedback::Low`](crate::Feedback::Low).
    ///
    /// Derived: `high_redundancy_byte × redundancy_step_factor` (default `2`).
    pub fn low_redundancy_byte(&self) -> u8 {
        u16::from(self.high_redundancy_byte)
            .saturating_mul(u16::from(self.redundancy_step_factor.get()))
            .min(u16::from(u8::MAX)) as u8
    }

    /// Number of frames in the sliding coding window: `2 * tau + 1`.
    pub fn window_frames(&self) -> u16 {
        u16::from(self.tau) * 2 + 1
    }

    /// Stripe size in bytes: `w * packet_size`.
    pub fn stripe_size(&self) -> u32 {
        u32::from(self.w) * u32::from(self.packet_size.get())
    }
}

/// Fluent builder for [`Config`].
///
/// Obtain via [`Config::builder`]. Each setter mirrors a [`Config`] accessor;
/// validation runs only in [`Self::build`].
#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    inner: Config,
    /// When `Some`, [`Self::build`] computes `min_window_parity` from `tau`.
    target_burst_packets: Option<u16>,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self {
            inner: Config {
                tau: Tau::new(3).expect("default tau is valid"),
                w: WordSize::W32,
                packet_size: NonZeroU16::new(8).expect("default packet_size is non-zero"),
                max_data_stripes: NonZeroU16::new(64)
                    .expect("default max_data_stripes is non-zero"),
                max_fec_stripes: 32,
                max_frame_size: 16_383,
                max_pkt_size: NonZeroU16::new(DEFAULT_MTU).expect("default mtu is non-zero"),
                feedback_interval: Duration::from_millis(2000),
                high_redundancy_byte: 1,
                redundancy_step_factor: NonZeroU8::new(2)
                    .expect("default redundancy_step_factor is non-zero"),
                feedback_history_len: NonZeroUsize::new(3)
                    .expect("default feedback_history_len is non-zero"),
                parity_delay: 0,
                min_window_parity: 0,
            },
            target_burst_packets: None,
        }
    }
}

impl ConfigBuilder {
    /// See [`Config::tau`].
    pub fn tau(mut self, tau: Tau) -> Self {
        self.inner.tau = tau;
        self
    }

    /// See [`Config::w`].
    pub fn w(mut self, w: WordSize) -> Self {
        self.inner.w = w;
        self
    }

    /// See [`Config::packet_size`].
    pub fn packet_size(mut self, packet_size: NonZeroU16) -> Self {
        self.inner.packet_size = packet_size;
        self
    }

    /// See [`Config::max_data_stripes`].
    pub fn max_data_stripes(mut self, n: NonZeroU16) -> Self {
        self.inner.max_data_stripes = n;
        self
    }

    /// See [`Config::max_fec_stripes`].
    pub fn max_fec_stripes(mut self, n: u16) -> Self {
        self.inner.max_fec_stripes = n;
        self
    }

    /// See [`Config::max_frame_size`].
    pub fn max_frame_size(mut self, bytes: u32) -> Self {
        self.inner.max_frame_size = bytes;
        self
    }

    /// Maximum on-wire datagram payload in bytes (default 1500).
    ///
    /// Must be ≥ [`Config::stripe_size`] after [`Config::builder`] → `.build()`, or
    /// [`ConfigError::StripeLargerThanPacket`] is returned.
    pub fn max_pkt_size(mut self, mtu: NonZeroU16) -> Self {
        self.inner.max_pkt_size = mtu;
        self
    }

    /// See [`Config::feedback_interval`].
    pub fn feedback_interval(mut self, interval: Duration) -> Self {
        self.inner.feedback_interval = interval;
        self
    }

    /// See [`Config::high_redundancy_byte`].
    pub fn high_redundancy_byte(mut self, byte: u8) -> Self {
        self.inner.high_redundancy_byte = byte;
        self
    }

    /// See [`Config::redundancy_step_factor`].
    pub fn redundancy_step_factor(mut self, factor: NonZeroU8) -> Self {
        self.inner.redundancy_step_factor = factor;
        self
    }

    /// See [`Config::feedback_history_len`].
    pub fn feedback_history_len(mut self, len: NonZeroUsize) -> Self {
        self.inner.feedback_history_len = len;
        self
    }

    /// See [`Config::parity_delay`].
    pub fn parity_delay(mut self, delay: u16) -> Self {
        self.inner.parity_delay = delay;
        self
    }

    /// See [`Config::min_window_parity`].
    pub fn min_window_parity(mut self, n: u16) -> Self {
        self.inner.min_window_parity = n;
        self
    }

    /// Convenience helper: set [`min_window_parity`](Self::min_window_parity)
    /// so the sliding window is guaranteed to carry enough parity equations
    /// to recover a burst of `burst_packets` consecutive packet losses.
    ///
    /// The exact K_min is computed in [`Self::build`] from the *final* `τ` value:
    ///
    /// ```text
    /// K_min = ceil(B · (2τ + 1) / (τ + 1))
    /// ```
    ///
    /// where `B = burst_packets`. Capped at [`max_fec_stripes`](Self::max_fec_stripes)
    /// during encoding.
    ///
    /// Call order with [`tau`](Self::tau) does not matter — the formula
    /// always uses the final `τ` at build time.
    pub fn target_burst_protection(mut self, burst_packets: u16) -> Self {
        self.target_burst_packets = Some(burst_packets);
        self
    }

    /// Produce a validated [`Config`].
    ///
    /// Returns [`ConfigError::StripeLargerThanPacket`] if
    /// `w * packet_size > max_pkt_size`, [`ConfigError::ParityCapacityOverflow`] if
    /// `stripe_size × max_fec_stripes` exceeds `u16::MAX`,
    /// [`ConfigError::InsufficientDataStripes`] /
    /// [`ConfigError::InsufficientFecStripes`] if `max_frame_size` cannot be encoded
    /// within the stripe caps at [`Feedback::High`](crate::Feedback::High).
    pub fn build(mut self) -> Result<Config, ConfigError> {
        let stripe_size = self.inner.stripe_size();
        let max_pkt_size = self.inner.max_pkt_size.get();
        if stripe_size > u32::from(max_pkt_size) {
            return Err(ConfigError::StripeLargerThanPacket {
                stripe_size,
                max_pkt_size,
            });
        }

        let max_fec = u32::from(self.inner.max_fec_stripes);
        if stripe_size.checked_mul(max_fec).is_none() {
            return Err(ConfigError::ParityCapacityOverflow {
                stripe_size,
                max_fec_stripes: self.inner.max_fec_stripes,
                product: u32::MAX,
                max_parity_bytes: u32::MAX,
            });
        }

        let max_frame_size = self.inner.max_frame_size;
        if max_frame_size > 0 {
            let required_data = num_data_stripes_for_frame_size(max_frame_size, stripe_size);
            let max_data = u32::from(self.inner.max_data_stripes.get());
            if required_data > max_data {
                return Err(ConfigError::InsufficientDataStripes {
                    max_frame_size,
                    stripe_size,
                    required: required_data,
                    max_data_stripes: self.inner.max_data_stripes.get(),
                });
            }

            let required_fec = num_parity_stripes_high(required_data);
            let max_fec = u32::from(self.inner.max_fec_stripes);
            if required_fec > max_fec {
                return Err(ConfigError::InsufficientFecStripes {
                    max_frame_size,
                    required: required_fec,
                    max_fec_stripes: self.inner.max_fec_stripes,
                });
            }
        }

        // target_burst_protection: compute K_min from the final τ value
        if let Some(burst) = self.target_burst_packets {
            let tau = u32::from(self.inner.tau.get());
            let window = tau * 2 + 1; // 2τ + 1
            let tau_plus_1 = tau + 1; // τ + 1
            let k_min = (u32::from(burst) * window).div_ceil(tau_plus_1);
            self.inner.min_window_parity = k_min as u16;
        }

        Ok(self.inner)
    }
}

impl TryFrom<ConfigBuilder> for Config {
    type Error = ConfigError;

    fn try_from(builder: ConfigBuilder) -> Result<Self, Self::Error> {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_try_from() {
        assert_eq!(Tau::try_from(3).unwrap().get(), 3);
        assert!(matches!(
            Tau::try_from(9),
            Err(ConfigError::TauOutOfRange { value: 9, max: 8 })
        ));
    }

    #[test]
    fn default_build_is_ok() {
        Config::builder().build().expect("defaults are valid");
    }

    #[test]
    fn stripe_size_exceeds_packet_size_is_rejected() {
        let res = Config::builder()
            .w(WordSize::W32)
            .packet_size(NonZeroU16::new(64).unwrap())
            .max_pkt_size(NonZeroU16::new(1024).unwrap())
            .build();
        assert_eq!(
            res,
            Err(ConfigError::StripeLargerThanPacket {
                stripe_size: 32 * 64,
                max_pkt_size: 1024,
            })
        );
    }

    #[test]
    fn stripe_size_equal_to_packet_size_is_accepted() {
        let cfg = Config::builder()
            .w(WordSize::W8)
            .packet_size(NonZeroU16::new(128).unwrap())
            .max_pkt_size(NonZeroU16::new(8 * 128).unwrap())
            .build()
            .unwrap();
        assert_eq!(cfg.stripe_size(), u32::from(cfg.max_pkt_size().get()));
    }

    #[test]
    fn insufficient_data_stripes_is_rejected() {
        let res = Config::builder()
            .max_frame_size(48 * 1024)
            .max_data_stripes(NonZeroU16::new(100).unwrap())
            .build();
        assert_eq!(
            res,
            Err(ConfigError::InsufficientDataStripes {
                max_frame_size: 48 * 1024,
                stripe_size: 256,
                required: 192,
                max_data_stripes: 100,
            })
        );
    }

    #[test]
    fn insufficient_fec_stripes_is_rejected() {
        let res = Config::builder()
            .max_frame_size(16 * 1024)
            .max_data_stripes(NonZeroU16::new(64).unwrap())
            .max_fec_stripes(16)
            .build();
        assert_eq!(
            res,
            Err(ConfigError::InsufficientFecStripes {
                max_frame_size: 16 * 1024,
                required: 32,
                max_fec_stripes: 16,
            })
        );
    }

    #[test]
    fn parity_capacity_large_is_accepted() {
        // stripe_size = 32×32 = 1024; 1024×256 = 262k — now fits in u32 parity cap.
        Config::builder()
            .w(WordSize::W32)
            .packet_size(NonZeroU16::new(32).unwrap())
            .max_data_stripes(NonZeroU16::new(512).unwrap())
            .max_fec_stripes(256)
            .max_frame_size(262_144)
            .max_pkt_size(NonZeroU16::new(1400).unwrap())
            .build()
            .unwrap();
    }

    #[test]
    fn min_window_parity_default_is_zero() {
        let cfg = Config::builder().build().unwrap();
        assert_eq!(cfg.min_window_parity(), 0);
    }

    #[test]
    fn min_window_parity_set_and_get() {
        let cfg = Config::builder().min_window_parity(14).build().unwrap();
        assert_eq!(cfg.min_window_parity(), 14);
    }

    #[test]
    fn min_window_parity_propagates_to_packetization() {
        let cfg = Config::builder()
            .tau(Tau::new(3).unwrap())
            .min_window_parity(14)
            .build()
            .unwrap();
        let p = crate::packetization::StreamingPacketization::from_config(&cfg);
        // verify window_len from tau=3 → 7 frames
        assert_eq!(p.window_len_for_test(), 7);
    }

    #[test]
    fn target_burst_protection_formula_tau3() {
        // τ=3: K_min = ceil(B · (2·3+1) / (3+1)) = ceil(B · 7/4)
        // B=4: ceil(4·7/4) = ceil(7) = 7
        let cfg = Config::builder()
            .tau(Tau::new(3).unwrap())
            .target_burst_protection(4)
            .build()
            .unwrap();
        assert_eq!(cfg.min_window_parity(), 7);
    }

    #[test]
    fn target_burst_protection_formula_tau1() {
        // τ=1: K_min = ceil(B · (2·1+1) / (1+1)) = ceil(B · 3/2)
        // B=4: ceil(4·3/2) = ceil(6) = 6
        let cfg = Config::builder()
            .tau(Tau::new(1).unwrap())
            .target_burst_protection(4)
            .build()
            .unwrap();
        assert_eq!(cfg.min_window_parity(), 6);
    }

    #[test]
    fn target_burst_protection_formula_tau6() {
        // τ=6: K_min = ceil(B · (2·6+1) / (6+1)) = ceil(B · 13/7)
        // B=4: ceil(4·13/7) = ceil(7.428...) = 8
        let cfg = Config::builder()
            .tau(Tau::new(6).unwrap())
            .target_burst_protection(4)
            .build()
            .unwrap();
        assert_eq!(cfg.min_window_parity(), 8);
    }

    #[test]
    fn target_burst_protection_no_overflow() {
        // B=255 with τ=8: K_min = ceil(255·17/9) = ceil(481.66) = 482
        let cfg = Config::builder()
            .tau(Tau::new(8).unwrap())
            .target_burst_protection(255)
            .build()
            .unwrap();
        assert_eq!(cfg.min_window_parity(), 482);
    }
}
