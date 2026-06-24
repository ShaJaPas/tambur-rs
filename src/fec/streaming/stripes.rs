//! Stripe / frame placement helpers (`streaming_code_auxiliary_functions.*`).

use alloc::vec::Vec;

use crate::datagram::FecDatagram;

use crate::fec::{BlockCode, CodingMatrixInfo};

pub(crate) fn compute_num_data_stripes(frame_size: u32, stripe_size: u16) -> u16 {
    assert!(frame_size > 0);
    let ss = u32::from(stripe_size);
    (frame_size / ss + u32::from(!frame_size.is_multiple_of(ss))) as u16
}

pub(crate) fn final_data_stripe_of_frame_diff_size(
    frame_size: u32,
    stripe_size: u16,
    relative_stripe: u16,
    num_data_stripes: u16,
) -> bool {
    relative_stripe == num_data_stripes - 1 && !frame_size.is_multiple_of(u32::from(stripe_size))
}

pub(crate) fn final_pos_within_stripe(
    frame_size: u32,
    stripe_size: u16,
    relative_stripe: u16,
) -> u16 {
    let num_data = compute_num_data_stripes(frame_size, stripe_size);
    if final_data_stripe_of_frame_diff_size(frame_size, stripe_size, relative_stripe, num_data) {
        ((frame_size % u32::from(stripe_size)) - 1) as u16
    } else {
        stripe_size - 1
    }
}

pub(crate) fn first_pos_of_frame_in_block_code(
    frame: u16,
    num_frames: u16,
    max_data_stripes: u16,
    max_fec_stripes: u16,
    is_parity: bool,
) -> u16 {
    if is_parity {
        (frame % num_frames) * max_fec_stripes
    } else {
        (frame % num_frames) * max_data_stripes
    }
}

pub(crate) fn first_pos_of_frame_in_block_code_erased(
    frame: u16,
    num_frames: u16,
    max_data_stripes: u16,
    max_fec_stripes: u16,
    is_parity: bool,
) -> u16 {
    if is_parity {
        num_frames * max_data_stripes + (frame % num_frames) * max_fec_stripes
    } else {
        first_pos_of_frame_in_block_code(
            frame,
            num_frames,
            max_data_stripes,
            max_fec_stripes,
            false,
        )
    }
}

pub(crate) fn expired_frame(frame_num: u16, timeslot: u16, num_frames: u16) -> bool {
    let dist = timeslot.wrapping_sub(frame_num);
    if dist > crate::datagram::MAX_FRAME_NUM / 2 {
        return false;
    }
    dist >= num_frames
}

pub(crate) fn get_frame(data_pkts: &[FecDatagram], frame_size: u32) -> Vec<u8> {
    let mut payload = Vec::with_capacity(frame_size as usize);
    if data_pkts.is_empty() {
        return payload;
    }
    let mut remaining = frame_size as usize;
    for pkt in data_pkts {
        if pkt.is_parity || remaining == 0 {
            continue;
        }
        let take = remaining.min(pkt.payload.len());
        payload.extend_from_slice(&pkt.payload[..take]);
        remaining -= take;
    }
    debug_assert_eq!(payload.len(), frame_size as usize);
    payload
}

pub(crate) fn stripe_slice(
    frame: &[u8],
    pos: u16,
    relative_stripe_size: u16,
    stripe_size: u16,
) -> &[u8] {
    let start = (pos as usize) * (stripe_size as usize);
    debug_assert!(start < frame.len());
    debug_assert!(start + relative_stripe_size as usize <= frame.len());
    &frame[start..start + relative_stripe_size as usize]
}

/// Geometry shared by [`place_frame`] and parity extraction.
pub(crate) struct StripePlacement {
    pub stripe_size: u16,
    pub frame_max_data_stripes: u16,
    pub frame_max_fec_stripes: u16,
    pub num_frames: u16,
}

impl StripePlacement {
    pub(crate) fn first_data_pos(&self, frame_num: u16) -> u16 {
        first_pos_of_frame_in_block_code(
            frame_num,
            self.num_frames,
            self.frame_max_data_stripes,
            self.frame_max_fec_stripes,
            false,
        )
    }
}

pub(crate) fn place_frame(
    frame: &[u8],
    frame_num: u16,
    placement: &StripePlacement,
    block_code: &mut BlockCode,
    frame_size: u32,
) -> Result<(), crate::fec::error::FecError> {
    let num_data_stripes = compute_num_data_stripes(frame_size, placement.stripe_size);
    debug_assert!(num_data_stripes <= placement.frame_max_data_stripes);
    let first = placement.first_data_pos(frame_num);
    for rel in 0..num_data_stripes {
        let rel_size = final_pos_within_stripe(frame_size, placement.stripe_size, rel) + 1;
        let stripe = stripe_slice(frame, rel, rel_size, placement.stripe_size);
        debug_assert_eq!(stripe.len(), rel_size as usize);
        block_code.place_payload(first + rel, false, stripe)?;
    }
    Ok(())
}

pub(crate) fn parity_payloads_into(
    info: CodingMatrixInfo,
    frame_num: u16,
    parity_pkt_sizes: &[u16],
    placement: &StripePlacement,
    block_code: &mut BlockCode,
    delay: u16,
    out: &mut Vec<u8>,
) -> Result<(), crate::fec::error::FecError> {
    if parity_pkt_sizes.is_empty() {
        out.clear();
        return Ok(());
    }
    let packet_size = parity_pkt_sizes[0];
    let total = u32::from(packet_size) * parity_pkt_sizes.len() as u32;
    if total > u32::from(placement.stripe_size) * u32::from(placement.frame_max_fec_stripes) {
        return Err(crate::fec::error::FecError::InvalidMatrixDimensions);
    }
    out.clear();
    out.resize(total as usize, 0);
    set_payloads_flat(info, out, total, packet_size, frame_num, block_code, delay)?;
    Ok(())
}

fn set_payloads_flat(
    info: CodingMatrixInfo,
    out: &mut [u8],
    total: u32,
    packet_size: u16,
    frame_num: u16,
    block_code: &BlockCode,
    delay: u16,
) -> Result<(), crate::fec::error::FecError> {
    debug_assert_eq!(total as usize, out.len());
    debug_assert_eq!(u64::from(total) % u64::from(packet_size), 0);
    let endpoints = info.rows_of_frame(frame_num, delay);
    let mut row_iter = endpoints.0..=endpoints.1;
    let mut row_data: Option<&[u8]> = None;
    let mut row_off = 0usize;
    let mut write_off = 0usize;

    while write_off < out.len() {
        let mut needed = packet_size as usize;
        while needed > 0 {
            if row_data.is_none() || row_off >= row_data.unwrap().len() {
                let row_pos = row_iter
                    .next()
                    .ok_or(crate::fec::error::FecError::InvalidMatrixDimensions)?;
                row_data = Some(block_code.row_slice(row_pos, true)?);
                row_off = 0;
            }
            let row = row_data.unwrap();
            let take = needed.min(row.len() - row_off);
            out[write_off..write_off + take].copy_from_slice(&row[row_off..row_off + take]);
            row_off += take;
            write_off += take;
            needed -= take;
        }
    }
    Ok(())
}

/// Split a flat parity buffer into per-packet slices (all packets share `parity_pkt_sizes[0]`).
#[cfg(any(test, feature = "bench"))]
pub(crate) fn parity_packet_slices<'a>(
    out: &'a [u8],
    parity_pkt_sizes: &[u16],
) -> impl Iterator<Item = &'a [u8]> + 'a {
    let packet_size = parity_pkt_sizes.first().copied().unwrap_or(0) as usize;
    (0..parity_pkt_sizes.len()).map(move |i| {
        let start = i * packet_size;
        &out[start..start + packet_size]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::num_frames_for_delay;

    #[test]
    fn num_data_stripes_round_up() {
        assert_eq!(compute_num_data_stripes(100, 32), 4);
        assert_eq!(compute_num_data_stripes(64, 32), 2);
    }

    #[test]
    fn first_pos_mapping() {
        let nf = num_frames_for_delay(3);
        assert_eq!(nf, 7);
        assert_eq!(first_pos_of_frame_in_block_code(5, nf, 4, 2, false), 20);
        assert_eq!(first_pos_of_frame_in_block_code(1, nf, 4, 2, false), 4);
        assert_eq!(
            first_pos_of_frame_in_block_code_erased(1, nf, 4, 2, true),
            nf * 4 + 2
        );
    }
}
