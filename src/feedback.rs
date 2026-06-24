//! Receiver → sender feedback (redundancy level).

use crate::config::Config;
use crate::error::{DatagramError, Result};

/// Redundancy level advertised by the receiver to the sender.
///
/// Tambur defines exactly three discrete modes (0% / 50% / 25% parity ratio).
/// Wire encoding uses one byte per [`FeedbackCodec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Feedback {
    /// No FEC parity (0% overhead).
    #[default]
    None,
    /// High bandwidth / max recovery (50% parity ratio).
    High,
    /// Low bandwidth mode (25% parity ratio).
    Low,
}

/// Wire codec for [`Feedback`] (one byte per mode).
///
/// Built once from [`Config`] and reused for encode/decode without passing the full
/// session configuration on every feedback datagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedbackCodec {
    high_redundancy_byte: u8,
    low_redundancy_byte: u8,
}

impl FeedbackCodec {
    /// Capture redundancy wire bytes from session configuration.
    pub fn new(config: &Config) -> Self {
        Self {
            high_redundancy_byte: config.high_redundancy_byte(),
            low_redundancy_byte: config.low_redundancy_byte(),
        }
    }

    /// Encode feedback to its wire byte.
    pub fn encode(&self, feedback: Feedback) -> u8 {
        match feedback {
            Feedback::None => 0,
            Feedback::High => self.high_redundancy_byte,
            Feedback::Low => self.low_redundancy_byte,
        }
    }

    /// Decode a wire byte into feedback.
    pub fn decode(&self, byte: u8) -> Result<Feedback> {
        match byte {
            0 => Ok(Feedback::None),
            b if b == self.high_redundancy_byte => Ok(Feedback::High),
            b if b == self.low_redundancy_byte => Ok(Feedback::Low),
            byte => Err(DatagramError::UnknownRedundancyByte { byte }.into()),
        }
    }

    /// Serialize feedback to a single-byte datagram payload.
    pub fn encode_bytes(&self, feedback: Feedback) -> [u8; 1] {
        [self.encode(feedback)]
    }

    /// Parse feedback from a single-byte datagram payload.
    pub fn decode_bytes(&self, bytes: &[u8]) -> Result<Feedback> {
        let byte = *bytes.first().ok_or(DatagramError::EmptyFeedback)?;
        self.decode(byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_roundtrip() {
        let config = Config::builder().build().unwrap();
        let codec = FeedbackCodec::new(&config);
        for fb in [Feedback::None, Feedback::High, Feedback::Low] {
            let wire = codec.encode_bytes(fb);
            assert_eq!(codec.decode_bytes(&wire).unwrap(), fb);
        }
    }
}
