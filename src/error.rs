//! Error types for the public API.

use thiserror::Error;

/// Errors returned by encoding, decoding, and configuration.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// Wire format could not be parsed or was inconsistent.
    #[error(transparent)]
    Datagram(#[from] DatagramError),

    /// [`Config::builder`](crate::Config::builder) failed validation (see [`ConfigError`]).
    #[error(transparent)]
    Config(#[from] ConfigError),

    /// [`Encoder`](crate::Encoder) / [`Decoder`](crate::Decoder) session errors.
    #[error(transparent)]
    Codec(#[from] CodecError),

    /// Internal invariant violated (bug); should not happen in normal use.
    #[error(transparent)]
    Internal(#[from] InternalError),
}

/// [`Encoder`](crate::Encoder) / [`Decoder`](crate::Decoder) session errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CodecError {
    /// Source frame is not yet recoverable or was abandoned.
    #[error("frame {frame_num} not recovered")]
    FrameNotRecovered {
        /// Monotonic session frame index that could not be recovered.
        frame_num: u64,
    },

    /// Payload exceeds configured [`Config::max_frame_size`](crate::Config::max_frame_size).
    #[error("frame payload {size} bytes exceeds max_frame_size ({max})")]
    PayloadTooLarge {
        /// Actual payload size in bytes.
        size: u32,
        /// Configured [`Config::max_frame_size`](crate::Config::max_frame_size).
        max: u32,
    },
}

/// [`Config::builder`](crate::Config::builder) rejected the configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ConfigError {
    /// τ exceeds [`Tau::MAX`](crate::Tau::MAX).
    #[error("tau {value} exceeds maximum ({max})")]
    TauOutOfRange {
        /// Supplied τ value.
        value: u8,
        /// Maximum allowed τ ([`Tau::MAX`](crate::Tau::MAX)).
        max: u8,
    },

    /// Computed `stripe_size = w * packet_size` does not fit in one network packet.
    ///
    /// Without `stripe_size <= max_pkt_size`, `StreamingPacketization` would emit
    /// over-MTU datagrams (or zero-stripe packets).
    #[error("stripe_size ({stripe_size}) exceeds max_pkt_size ({max_pkt_size})")]
    StripeLargerThanPacket {
        /// Computed stripe size in bytes.
        stripe_size: u32,
        /// Configured maximum datagram payload size.
        max_pkt_size: u16,
    },

    /// `max_frame_size` exceeds the internal u16 frame size limit (65535).
    #[error("max_frame_size ({max_frame_size}) exceeds the u16 frame size limit ({max_allowed})")]
    FrameSizeTooLarge {
        /// Configured maximum source payload.
        max_frame_size: u32,
        /// Maximum allowed frame size (`u16::MAX`).
        max_allowed: u32,
    },

    /// [`Config::max_frame_size`](crate::Config::max_frame_size) needs more data stripes than [`Config::max_data_stripes`](crate::Config::max_data_stripes).
    #[error(
        "max_frame_size ({max_frame_size}) needs {required} data stripes at stripe_size {stripe_size}, but max_data_stripes is {max_data_stripes}"
    )]
    InsufficientDataStripes {
        /// Configured maximum source payload.
        max_frame_size: u32,
        /// Stripe size in bytes (`w × packet_size`).
        stripe_size: u32,
        /// Data stripes required for `max_frame_size`.
        required: u32,
        /// Configured [`Config::max_data_stripes`](crate::Config::max_data_stripes).
        max_data_stripes: u16,
    },

    /// [`Feedback::High`](crate::Feedback::High) on a max-sized frame needs more parity stripes than [`Config::max_fec_stripes`](crate::Config::max_fec_stripes).
    #[error(
        "max_frame_size ({max_frame_size}) needs {required} parity stripes at Feedback::High, but max_fec_stripes is {max_fec_stripes}"
    )]
    InsufficientFecStripes {
        /// Configured maximum source payload.
        max_frame_size: u32,
        /// Parity stripes required at [`Feedback::High`](crate::Feedback::High).
        required: u32,
        /// Configured [`Config::max_fec_stripes`](crate::Config::max_fec_stripes).
        max_fec_stripes: u16,
    },

    /// `stripe_size × max_fec_stripes` exceeds the internal per-frame parity byte cap (`u16::MAX`).
    ///
    /// Encoder parity extraction compares against this product as a byte budget; if it
    /// overflows `u16`, [`Encoder::encode_payload`](crate::Encoder::encode_payload) fails
    /// at [`Feedback::High`](crate::Feedback::High) with [`InternalError::InvariantViolated`].
    #[error(
        "stripe_size ({stripe_size}) × max_fec_stripes ({max_fec_stripes}) = {product}, which exceeds the parity byte cap ({max_parity_bytes})"
    )]
    ParityCapacityOverflow {
        /// Stripe size in bytes (`w × packet_size`).
        stripe_size: u32,
        /// Configured [`Config::max_fec_stripes`](crate::Config::max_fec_stripes).
        max_fec_stripes: u16,
        /// `stripe_size × max_fec_stripes` (may wrap `u16` when over the cap).
        product: u32,
        /// Maximum supported product (`u16::MAX`).
        max_parity_bytes: u32,
    },
}

/// [`FecDatagram`](crate::FecDatagram) / [`Feedback`](crate::Feedback) wire errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DatagramError {
    /// Feedback payload was empty.
    #[error("empty feedback datagram")]
    EmptyFeedback,

    /// Wire byte does not match any [`Feedback`](crate::Feedback) level for this session.
    #[error("unknown redundancy wire byte: {byte}")]
    UnknownRedundancyByte {
        /// Received wire byte.
        byte: u8,
    },

    /// Buffer shorter than a [`FecDatagram`](crate::FecDatagram) header.
    #[error("truncated fec datagram header")]
    TruncatedHeader,

    /// `frame_num` does not fit in 15 bits (max {max:?}).
    #[error("frame_num {frame_num} exceeds 15-bit limit ({max})")]
    FrameNumOutOfRange {
        /// Supplied frame index.
        frame_num: u16,
        /// Maximum encodable value (`2^15 - 1`).
        max: u16,
    },
}

/// Unexpected internal state (implementation bug).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InternalError {
    /// A library invariant was violated.
    #[error("internal invariant violated")]
    InvariantViolated,
}

/// Convenience alias for [`core::result::Result`] with [`enum@Error`].
pub type Result<T> = core::result::Result<T, Error>;
