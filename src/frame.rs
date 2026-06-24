//! Source data units after FEC recovery.

use bytes::Bytes;

/// Source unit recovered by [`Decoder`](crate::Decoder).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredFrame {
    /// Monotonic session index (never wraps; wire `frame_num` cycles every 32 768 frames).
    pub frame_num: u64,
    /// Reconstructed application payload.
    pub payload: Bytes,
    /// `true` if every data packet arrived without FEC reconstruction.
    pub direct_reception: bool,
}
