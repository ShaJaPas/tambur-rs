//! BlockCode setup for benchmarks.
#![cfg_attr(not(feature = "bench"), allow(unreachable_pub))]

use alloc::vec::Vec;

use super::block_code::BlockCode;
use super::matrix::info::CodingMatrixInfo;
use super::puncture::num_frames_for_delay;
use super::streaming::test_common::{BurstScenario, run_burst_before_decode};
use crate::word_size::WordSize;

/// Data stripes per frame in encode bench.
pub const N_DATA_PER_FRAME: u16 = 19;

/// Parity stripes per frame in encode bench.
pub const N_PAR_PER_FRAME: u16 = 9;

/// Timeslot used in encode bench.
pub const ENCODE_BENCH_TIMESLOT: u16 = 50;

/// Offset into the decode loop (`num_frames - 1 + offset`, runs offset `< 100`).
pub const DECODE_BENCH_TIMESLOT_OFFSET: u16 = 50;

/// `packet_size = 8 * (1000 / w)`.
pub fn packet_size_for_w(w: WordSize) -> u16 {
    8 * (1000 / u16::from(w))
}

/// Decode loop: `timeslot = num_frames - 1 .. num_frames - 1 + 100`.
pub fn decode_bench_timeslot(delay: u16) -> u16 {
    let num_frames = num_frames_for_delay(delay);
    num_frames - 1 + DECODE_BENCH_TIMESLOT_OFFSET
}

/// Data stripes per frame in decode bench (`run_burst` frame size must fit in `u16`).
pub const DECODE_BENCH_N_DATA: u16 = 4;

/// Build encoder state immediately before [`BlockCode::encode`].
pub fn prepare_encode_block(delay: u16, w: WordSize, timeslot: u16) -> BlockCode {
    let packet_size = packet_size_for_w(w);
    let num_frames = num_frames_for_delay(delay);
    let n_cols = N_DATA_PER_FRAME * num_frames;
    let n_rows = N_PAR_PER_FRAME * num_frames;
    let info = CodingMatrixInfo::new(n_rows, n_cols, w).expect("matrix info");
    let stripe_bytes = (packet_size * u16::from(w)) as usize;

    let mut block = BlockCode::new(info, delay, (1, 1), packet_size).expect("block");

    let total = (n_cols + n_rows) as usize;
    let mut payloads = Vec::with_capacity(total);
    for j in 0..total {
        payloads.push(vec![(j + 1) as u8; stripe_bytes]);
    }

    for pos in 0..n_cols {
        block
            .place_payload(pos, false, &payloads[pos as usize])
            .expect("place data");
    }
    for pos in 0..n_rows {
        block
            .place_payload(pos, true, &payloads[(n_cols + pos) as usize])
            .expect("place parity");
    }

    for time in 0..timeslot {
        block.update_timeslot(time).expect("update");
        let cols = info.cols_of_frame(time, delay);
        for pos in cols.0..=cols.1 {
            block
                .place_payload(pos, false, &payloads[pos as usize])
                .expect("place data slot");
        }
        let rows = info.rows_of_frame(time, delay);
        for pos in rows.0..=rows.1 {
            block
                .place_payload(pos, true, &payloads[(n_cols + pos) as usize])
                .expect("place parity slot");
        }
    }

    block.update_timeslot(timeslot).expect("update timeslot");
    let cols = info.cols_of_frame(timeslot, delay);
    for pos in cols.0..=cols.1 {
        block
            .place_payload(pos, false, &payloads[pos as usize])
            .expect("place final data");
    }

    block
}

/// Timed portion of the encode bench.
pub fn run_encode(block: &mut BlockCode) {
    block.encode().expect("encode");
}

/// Build decoder state immediately before [`BlockCode::decode`].
///
/// Uses [`BurstScenario::FullRecover`] (from `test_streaming_code.cc`) so decode runs
/// the full Jerasure path. Burst-scale stripes (`packet_size = 8`) — block-scale
/// `N_DATA_PER_FRAME = 19` overflows `u16` frame sizes in the burst harness.
pub fn prepare_decode_block(delay: u16, w: WordSize, timeslot: u16) -> BlockCode {
    let (streaming, _, _, _) = run_burst_before_decode(
        BurstScenario::FullRecover { exp_loss: 0 },
        delay,
        w,
        8,
        DECODE_BENCH_N_DATA,
        timeslot,
    );
    streaming.into_block_code()
}

/// Build decoder state for large-stripe parallel decode bench.
///
/// Uses larger `packet_size` and `n_data` to trigger the parallel decode path
/// (≥12 erasures and ≥4096 B stripe size).
pub fn prepare_decode_block_large(
    delay: u16,
    w: WordSize,
    timeslot: u16,
    packet_size: u16,
    n_data: u16,
) -> BlockCode {
    let (streaming, _, _, _) = run_burst_before_decode(
        BurstScenario::FullRecover { exp_loss: 0 },
        delay,
        w,
        packet_size,
        n_data,
        timeslot,
    );
    streaming.into_block_code()
}

/// Timed portion of the decode bench.
pub fn run_decode(block: &mut BlockCode) -> Vec<bool> {
    block.decode().expect("decode")
}

#[cfg(test)]
#[allow(clippy::std_instead_of_core)]
mod smoke {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn encode_bench_harness_smoke() {
        let block = prepare_encode_block(3, WordSize::W32, ENCODE_BENCH_TIMESLOT);
        let mut block = block;
        run_encode(&mut block);
    }

    #[test]
    fn decode_bench_harness_timing() {
        let delay = 3;
        let timeslot = decode_bench_timeslot(delay);
        for w in [WordSize::W8, WordSize::W32] {
            let t0 = Instant::now();
            let template = prepare_decode_block(delay, w, timeslot);
            let prep = t0.elapsed();
            assert!(
                prep < Duration::from_secs(60),
                "prepare w={w:?} took {prep:?}"
            );

            let t2 = Instant::now();
            let mut block = template.clone();
            let mask = run_decode(&mut block);
            let dec = t2.elapsed();
            assert!(
                mask.iter().any(|&r| r),
                "w={w:?} bench decode must recover at least one stripe"
            );
            eprintln!("w={w:?} full decode took {dec:?}");
        }
    }
}
