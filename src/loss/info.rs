//! Flattened loss bitmaps for one feedback window.

use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct LossInfo {
    pub indices: Vec<u16>,
    pub packet_losses: Vec<bool>,
    pub frame_losses: Vec<bool>,
}
