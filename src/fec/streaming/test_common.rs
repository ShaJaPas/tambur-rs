//! Shared helpers for streaming integration tests.
#![allow(clippy::too_many_arguments)]
#![cfg_attr(all(feature = "bench", not(test)), allow(dead_code))]

use bytes::Bytes;

use super::stripes::parity_packet_slices;
use crate::datagram::FecDatagram;
use crate::fec::num_frames_for_delay;
use crate::fec::{BlockCode, CodingMatrixInfo, StreamingCode, StreamingCodeHelper};

pub(crate) fn frame_sizes_window(timeslot: u16, num_frames: u16, size: u64) -> Vec<Option<u64>> {
    let mut frame_sizes = Vec::new();
    if timeslot < num_frames - 1 {
        for _ in 0..(num_frames - timeslot - 1) as usize {
            frame_sizes.push(None);
        }
    }
    while frame_sizes.len() < num_frames as usize {
        frame_sizes.push(Some(size));
    }
    frame_sizes
}

pub(crate) fn stripe_payload(col: u16, stripe_size: u16) -> Vec<u8> {
    vec![(3 * col + 1) as u8; stripe_size as usize]
}

pub(crate) fn make_pair(
    delay: u16,
    stripe_size: u16,
    packet_size: u16,
    w: crate::word_size::WordSize,
    n_data: u16,
    n_par: u16,
    vu: (u16, u16),
) -> (StreamingCode, StreamingCode, CodingMatrixInfo) {
    let num_frames = num_frames_for_delay(delay);
    let info = CodingMatrixInfo::new(n_par * num_frames, n_data * num_frames, w).unwrap();
    let encode_block = BlockCode::new(info, delay, vu, packet_size).unwrap();
    let decode_block = BlockCode::new(info, delay, vu, packet_size).unwrap();
    let w_bits = u16::from(w);
    let encode =
        StreamingCode::new(delay, stripe_size, encode_block, w_bits, n_data, n_par).unwrap();
    let decode =
        StreamingCode::new(delay, stripe_size, decode_block, w_bits, n_data, n_par).unwrap();
    (encode, decode, info)
}

pub(crate) fn frame_mod(time: u16, num_frames: u16) -> u16 {
    time % num_frames
}

pub(crate) fn copy_parity_rows(
    encode: &StreamingCode,
    decode: &mut StreamingCode,
    info: CodingMatrixInfo,
    time: u16,
    delay: u16,
) {
    let rows = info.rows_of_frame(time, delay);
    for row in rows.0..=rows.1 {
        let payload = encode.parity_row(row).unwrap();
        decode.place_parity_row(row, &payload).unwrap();
    }
}

fn parity_pkt(sqn: &mut u32, time: u16, pos: usize, payload: Vec<u8>) -> FecDatagram {
    FecDatagram {
        seq_num: *sqn,
        is_parity: true,
        frame_num: time,
        sizes_of_frames_encoding: 0,
        pos_in_frame: pos as u16,
        stripe_pos_in_frame: pos as u16,
        payload: Bytes::from(payload),
    }
}

/// Burst integration scenarios from `test_streaming_code.cc`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BurstScenario {
    FullRecover { exp_loss: u16 },
    PartialRecover,
    PartialRecoverExpPlus,
    FullRecoverExpPlus,
    VRecover,
    NotRecovered,
    FullRecoverExp { shift_start: u16 },
    NotRecoveredExp { shift_start: u16 },
}

impl BurstScenario {
    fn vu(self) -> (u16, u16) {
        match self {
            Self::FullRecover { .. } | Self::FullRecoverExp { .. } => (1, 0),
            _ => (1, 1),
        }
    }

    fn n_par(self, n_data: u16) -> u16 {
        match self {
            Self::FullRecover { .. } | Self::PartialRecover | Self::VRecover => n_data,
            Self::FullRecoverExpPlus => n_data * 2,
            Self::NotRecovered | Self::NotRecoveredExp { .. } => n_data / 2,
            Self::PartialRecoverExpPlus | Self::FullRecoverExp { .. } => n_data,
        }
    }

    fn drop_data_frame(
        self,
        fm: u16,
        timeslot: u16,
        shift: u16,
        num_frames: u16,
        _exp_loss: u16,
        _shift_start: u16,
    ) -> bool {
        match self {
            Self::FullRecover { exp_loss } => {
                fm == (timeslot - shift) % num_frames && exp_loss % 2 != 0
                    || fm == (timeslot - shift + 1) % num_frames && exp_loss / 2 != 0
            }
            Self::PartialRecover | Self::FullRecoverExp { .. } => false,
            Self::PartialRecoverExpPlus => {
                fm == (timeslot - shift + 3) % num_frames
                    || fm == (timeslot - shift) % num_frames
                    || fm == (timeslot - shift + 1) % num_frames
            }
            Self::FullRecoverExpPlus => {
                fm == (timeslot - shift + 2) % num_frames
                    || fm == (timeslot - shift + 3) % num_frames
                    || fm == (timeslot - shift + 4) % num_frames
            }
            Self::VRecover | Self::NotRecovered | Self::NotRecoveredExp { .. } => false,
        }
    }

    fn drop_parity_frame(
        self,
        fm: u16,
        timeslot: u16,
        shift: u16,
        num_frames: u16,
        _shift_start: u16,
    ) -> bool {
        match self {
            Self::FullRecover { .. } | Self::PartialRecover | Self::NotRecovered => {
                fm == (timeslot - shift + 3) % num_frames
                    || fm == (timeslot - shift + 4) % num_frames
            }
            Self::PartialRecoverExpPlus => fm == (timeslot - shift + 3) % num_frames,
            Self::FullRecoverExpPlus => {
                fm == (timeslot - shift + 2) % num_frames
                    || fm == (timeslot - shift + 3) % num_frames
                    || fm == (timeslot - shift + 4) % num_frames
            }
            Self::VRecover => {
                fm == (timeslot - shift + 4) % num_frames
                    || fm == (timeslot - shift + 5) % num_frames
            }
            Self::FullRecoverExp { shift_start } | Self::NotRecoveredExp { shift_start } => {
                fm == (timeslot - shift + 3 - shift_start) % num_frames
                    || fm == (timeslot - shift + 4 - shift_start) % num_frames
            }
        }
    }

    fn use_place_payload(self) -> bool {
        !matches!(self, Self::FullRecover { .. })
    }

    fn send_parity_pkts(self) -> bool {
        matches!(self, Self::FullRecover { .. } | Self::PartialRecover)
    }

    fn stripe_recovered(
        self,
        frame: u16,
        col: u16,
        timeslot: u16,
        shift: u16,
        num_frames: u16,
        n_data: u16,
        _exp_loss: u16,
        _shift_start: u16,
        helper: &StreamingCodeHelper,
    ) -> bool {
        let stripe = col - frame * n_data;
        let _ = stripe;
        let fm = frame % num_frames;
        match self {
            Self::FullRecover { exp_loss } => {
                !((fm == (timeslot - shift) % num_frames && exp_loss % 2 != 0)
                    || (fm == (timeslot - shift + 1) % num_frames && exp_loss / 2 != 0))
            }
            Self::PartialRecover | Self::FullRecoverExpPlus => {
                !(fm == (timeslot - shift + 4) % num_frames
                    && helper.positions_of_us()[col as usize])
            }
            Self::PartialRecoverExpPlus => fm != (timeslot - shift) % num_frames,
            Self::VRecover => {
                !((fm == (timeslot - shift + 4) % num_frames
                    || fm == (timeslot - shift + 5) % num_frames)
                    && helper.positions_of_us()[col as usize])
            }
            Self::NotRecovered => {
                fm != (timeslot - shift + 3) % num_frames
                    && fm != (timeslot - shift + 4) % num_frames
            }
            Self::NotRecoveredExp { shift_start } => {
                fm != (timeslot - shift + 3 - shift_start) % num_frames
                    && fm != (timeslot - shift + 4 - shift_start) % num_frames
            }
            Self::FullRecoverExp { .. } => true,
        }
    }
}

fn run_burst_inner(
    scenario: BurstScenario,
    delay: u16,
    w: crate::word_size::WordSize,
    packet_size: u16,
    n_data: u16,
    timeslot: u16,
) -> (StreamingCode, CodingMatrixInfo, u32, StreamingCodeHelper) {
    let stripe_size = packet_size * u16::from(w);
    let num_frames = num_frames_for_delay(delay);
    let shift = num_frames - 1;
    let n_par = scenario.n_par(n_data);
    let size = u32::from(n_data) * u32::from(stripe_size);
    let parity_sizes = vec![stripe_size; n_par as usize];
    let vu = scenario.vu();
    let exp_loss = match scenario {
        BurstScenario::FullRecover { exp_loss } => exp_loss,
        _ => 0,
    };
    let shift_start = match scenario {
        BurstScenario::FullRecoverExp { shift_start }
        | BurstScenario::NotRecoveredExp { shift_start } => shift_start,
        _ => 0,
    };

    let (mut encode, mut decode, info) =
        make_pair(delay, stripe_size, packet_size, w, n_data, n_par, vu);
    let helper = StreamingCodeHelper::new(info, delay, vu);
    let mut sqn = 0u32;

    for time in 0..=timeslot {
        let mut data_pkts = Vec::new();
        for pos in 0..n_data {
            let col = frame_mod(time, num_frames) * n_data + pos;
            data_pkts.push(FecDatagram {
                seq_num: sqn,
                is_parity: false,
                frame_num: time,
                sizes_of_frames_encoding: 0,
                pos_in_frame: pos,
                stripe_pos_in_frame: pos,
                payload: Bytes::from(stripe_payload(col, stripe_size)),
            });
            sqn += 1;
        }
        let mut parity_out = Vec::new();
        encode
            .encode(&data_pkts, &parity_sizes, size, &mut parity_out)
            .unwrap();
        let fm = frame_mod(time, num_frames);
        let drop_data =
            scenario.drop_data_frame(fm, timeslot, shift, num_frames, exp_loss, shift_start);
        let drop_parity = scenario.drop_parity_frame(fm, timeslot, shift, num_frames, shift_start);

        if !drop_parity {
            if !drop_data {
                for pkt in &data_pkts {
                    decode.add_packet(pkt, time).unwrap();
                }
            }
            if scenario.use_place_payload() {
                copy_parity_rows(&encode, &mut decode, info, time, delay);
            }
            if scenario.send_parity_pkts() && !drop_data {
                for (pos, payload) in parity_packet_slices(&parity_out, &parity_sizes).enumerate() {
                    let pkt = parity_pkt(&mut sqn, time, pos, payload.to_vec());
                    decode.add_packet(&pkt, time).unwrap();
                }
            }
        }
    }

    if timeslot > 0 {
        decode.set_timeslot(timeslot - 1);
    }
    (decode, info, size, helper)
}

/// Burst setup without calling [`StreamingCode::decode`] (for decode benchmarks).
pub(crate) fn run_burst_before_decode(
    scenario: BurstScenario,
    delay: u16,
    w: crate::word_size::WordSize,
    packet_size: u16,
    n_data: u16,
    timeslot: u16,
) -> (StreamingCode, CodingMatrixInfo, u32, StreamingCodeHelper) {
    run_burst_inner(scenario, delay, w, packet_size, n_data, timeslot)
}

pub(crate) fn run_burst(
    scenario: BurstScenario,
    delay: u16,
    w: crate::word_size::WordSize,
    packet_size: u16,
    n_data: u16,
    timeslot: u16,
) -> (StreamingCode, CodingMatrixInfo, u32, StreamingCodeHelper) {
    let (mut decode, info, size, helper) =
        run_burst_inner(scenario, delay, w, packet_size, n_data, timeslot);
    decode
        .decode(&frame_sizes_window(
            timeslot,
            num_frames_for_delay(delay),
            u64::from(size),
        ))
        .unwrap();
    (decode, info, size, helper)
}

pub(crate) fn assert_burst_stripes(
    scenario: BurstScenario,
    decode: &StreamingCode,
    info: CodingMatrixInfo,
    delay: u16,
    size: u32,
    n_data: u16,
    timeslot: u16,
    helper: &StreamingCodeHelper,
) {
    let num_frames = num_frames_for_delay(delay);
    let shift = num_frames - 1;
    let exp_loss = match scenario {
        BurstScenario::FullRecover { exp_loss } => exp_loss,
        _ => 0,
    };
    let shift_start = match scenario {
        BurstScenario::FullRecoverExp { shift_start }
        | BurstScenario::NotRecoveredExp { shift_start } => shift_start,
        _ => 0,
    };

    for col in 0..(num_frames * n_data) {
        let frame = info.frame_of_col(col, delay).unwrap();
        let stripe = col - frame * n_data;
        let expected = scenario.stripe_recovered(
            frame,
            col,
            timeslot,
            shift,
            num_frames,
            n_data,
            exp_loss,
            shift_start,
            helper,
        );
        assert_eq!(
            decode.is_stripe_recovered(frame, stripe, size),
            expected,
            "scenario={scenario:?} col={col} frame={frame} timeslot={timeslot}"
        );
    }
}
