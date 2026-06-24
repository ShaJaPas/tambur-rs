//! Shared helpers for loss-computer integration tests.

use alloc::vec::Vec;

use crate::codec::ReceiveFrame;
use crate::datagram::FecDatagram;
use crate::feedback::Feedback;
use crate::packetization::{DEFAULT_MTU, StreamingPacketization};

/// MTU for packetization.
pub(super) const MTU: u64 = 1500;

/// Stripe size for tests (`8 * 32`).
pub(super) const STRIPE_SIZE: u16 = 256;

const SIZE_INCREMENT: [u64; 6] = [1, 1, 1, 1, 1, MTU - 2];

/// glibc-compatible `rand()` after `srand(0)`.
pub(super) struct TestRng {
    state: u32,
}

impl TestRng {
    pub(super) fn seeded() -> Self {
        Self { state: 0 }
    }

    pub(super) fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.state / 65536) % 32768
    }

    pub(super) fn next_bounded(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        self.next_u32() % bound
    }
}

pub(super) fn size_increment(index: usize) -> u64 {
    SIZE_INCREMENT[index % SIZE_INCREMENT.len()]
}

pub(super) fn get_missing(num_pkts: u8, num_missing: u8, rng: &mut TestRng) -> Vec<bool> {
    let mut is_missing = vec![false; num_pkts as usize];
    let mut dropped = 0u8;
    while dropped < num_missing {
        let index = rng.next_bounded(u32::from(num_pkts)) as u8;
        if !is_missing[index as usize] {
            is_missing[index as usize] = true;
            dropped += 1;
        }
    }
    is_missing
}

pub(super) fn highest_missing_pos(missing: &[bool]) -> u16 {
    missing
        .iter()
        .enumerate()
        .filter(|&(_, m)| !m)
        .map(|(i, _)| i as u16)
        .next_back()
        .unwrap_or(0)
}

pub(super) fn make_frame(frame_num: u16, is_dummy: bool) -> ReceiveFrame {
    let mut frame = ReceiveFrame::new(u64::from(frame_num), frame_num);
    frame.reset_for(u64::from(frame_num), frame_num, is_dummy);
    frame
}

pub(super) fn generate_frame_pkts(
    packetization: &mut StreamingPacketization,
    frame_num: u16,
    frame_size: u64,
) -> Vec<FecDatagram> {
    let size = frame_size.min(u64::from(u16::MAX)) as u16;
    let specs = packetization.plan_frame(u32::from(size), Feedback::High);
    let mut pkts = Vec::with_capacity(specs.len());
    for (pos, spec) in specs.iter().enumerate() {
        pkts.push(FecDatagram {
            seq_num: u32::from(frame_num) * 10_000 + u32::from(pos as u16),
            is_parity: spec.is_parity,
            frame_num,
            sizes_of_frames_encoding: 0,
            pos_in_frame: pos as u16,
            stripe_pos_in_frame: pos as u16,
            payload: bytes::Bytes::from(vec![0u8; spec.payload_len as usize]),
        });
    }
    pkts
}

pub(super) fn new_packetization() -> StreamingPacketization {
    StreamingPacketization::new(STRIPE_SIZE, DEFAULT_MTU, 0, 0, 0, 1)
}
