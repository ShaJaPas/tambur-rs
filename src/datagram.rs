//! FEC packet wire format.
//!
//! Layout matches Tambur `FECDatagram` (`fec_datagram.cc`): big-endian scalars,
//! `is_parity` in the MSB of the second field, `frame_num` in the lower 15 bits.

use core::convert::TryFrom;

use bytes::{BufMut, Bytes, BytesMut};

use crate::error::{DatagramError, Result};

/// Maximum value storable in the 15-bit `frame_num` wire field.
pub(crate) const MAX_FRAME_NUM: u16 = (1 << 15) - 1;

/// Parsed FEC datagram header (no payload).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FecDatagramHeader {
    /// Datagram sequence number (big-endian on wire).
    pub seq_num: u32,
    /// `true` for parity stripes, `false` for source data.
    pub is_parity: bool,
    /// Source frame index (15-bit on wire).
    pub frame_num: u16,
    /// Systematic encoding of recent frame sizes (header field).
    pub sizes_of_frames_encoding: u64,
    /// Packet index within the source frame.
    pub pos_in_frame: u16,
    /// Stripe index within the frame's packetization plan.
    pub stripe_pos_in_frame: u16,
}

impl FecDatagramHeader {
    /// Serialized header size in bytes (matches Tambur `serialize_to_string` output).
    pub const LEN: usize = 4 + 2 + 8 + 2 + 2;
}

impl TryFrom<&[u8]> for FecDatagramHeader {
    type Error = crate::error::Error;

    fn try_from(bytes: &[u8]) -> Result<Self> {
        parse_header(bytes)
    }
}

/// One on-the-wire FEC packet (data or parity).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FecDatagram {
    /// Datagram sequence number (big-endian on wire).
    pub seq_num: u32,
    /// `true` for parity stripes, `false` for source data.
    pub is_parity: bool,
    /// Source frame index (15-bit on wire).
    pub frame_num: u16,
    /// Systematic encoding of recent frame sizes (header field).
    pub sizes_of_frames_encoding: u64,
    /// Packet index within the source frame.
    pub pos_in_frame: u16,
    /// Stripe index within the frame's packetization plan.
    pub stripe_pos_in_frame: u16,
    /// Stripe payload bytes.
    pub payload: Bytes,
}

impl FecDatagram {
    /// Serialized header size in bytes (matches Tambur `serialize_to_string` output).
    pub const HEADER_LEN: usize = FecDatagramHeader::LEN;

    /// Total on-wire size.
    pub fn wire_len(&self) -> usize {
        Self::HEADER_LEN + self.payload.len()
    }

    /// Zero-copy wire view: small header buffer + payload (refcount only).
    ///
    /// Prefer this over [`Self::to_bytes`] when sending with vectored I/O (`writev` /
    /// `sendmsg`): header and payload can be transmitted without copying the payload.
    pub fn wire_parts(&self) -> Result<(Bytes, Bytes)> {
        let mut header = BytesMut::with_capacity(Self::HEADER_LEN);
        write_header(&mut header, self)?;
        Ok((header.freeze(), self.payload.clone()))
    }

    /// Encode to one contiguous buffer for transport.
    ///
    /// Copies the payload into a new allocation. For zero-copy sends use
    /// [`Self::wire_parts`] and transmit header + payload separately.
    pub fn to_bytes(&self) -> Result<Bytes> {
        let (header, payload) = self.wire_parts()?;
        let mut buf = BytesMut::with_capacity(header.len() + payload.len());
        buf.put(header);
        buf.put(payload);
        Ok(buf.freeze())
    }

    /// Parse header and payload from a transport buffer (zero-copy on payload).
    pub fn from_bytes(bytes: Bytes) -> Result<Self> {
        let header = parse_header(&bytes)?;
        if bytes.len() < Self::HEADER_LEN {
            return Err(DatagramError::TruncatedHeader.into());
        }
        let payload = bytes.slice(Self::HEADER_LEN..);
        Ok(Self::from_header(header, payload))
    }

    /// Build a datagram from a parsed header and payload slice.
    pub fn from_header(header: FecDatagramHeader, payload: Bytes) -> Self {
        Self {
            seq_num: header.seq_num,
            is_parity: header.is_parity,
            frame_num: header.frame_num,
            sizes_of_frames_encoding: header.sizes_of_frames_encoding,
            pos_in_frame: header.pos_in_frame,
            stripe_pos_in_frame: header.stripe_pos_in_frame,
            payload,
        }
    }
}

impl TryFrom<Bytes> for FecDatagram {
    type Error = crate::error::Error;

    fn try_from(bytes: Bytes) -> Result<Self> {
        Self::from_bytes(bytes)
    }
}

impl TryFrom<&[u8]> for FecDatagram {
    type Error = crate::error::Error;

    fn try_from(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(Bytes::copy_from_slice(bytes))
    }
}

impl TryFrom<FecDatagram> for Bytes {
    type Error = crate::error::Error;

    fn try_from(datagram: FecDatagram) -> Result<Bytes> {
        datagram.to_bytes()
    }
}

fn write_header(buf: &mut BytesMut, datagram: &FecDatagram) -> Result<()> {
    buf.put_u32(datagram.seq_num);
    buf.put_u16(encode_parity_frame(datagram.is_parity, datagram.frame_num));
    buf.put_u64(datagram.sizes_of_frames_encoding);
    buf.put_u16(datagram.pos_in_frame);
    buf.put_u16(datagram.stripe_pos_in_frame);
    Ok(())
}

fn parse_header(bytes: &[u8]) -> Result<FecDatagramHeader> {
    if bytes.len() < FecDatagramHeader::LEN {
        return Err(DatagramError::TruncatedHeader.into());
    }

    let seq_num = u32::from_be_bytes(bytes[0..4].try_into().expect("length checked"));
    let (is_parity, frame_num) = decode_parity_frame(u16::from_be_bytes(
        bytes[4..6].try_into().expect("length checked"),
    ));
    let sizes_of_frames_encoding =
        u64::from_be_bytes(bytes[6..14].try_into().expect("length checked"));
    let pos_in_frame = u16::from_be_bytes(bytes[14..16].try_into().expect("length checked"));
    let stripe_pos_in_frame = u16::from_be_bytes(bytes[16..18].try_into().expect("length checked"));

    Ok(FecDatagramHeader {
        seq_num,
        is_parity,
        frame_num,
        sizes_of_frames_encoding,
        pos_in_frame,
        stripe_pos_in_frame,
    })
}

/// Unpack Tambur MSB-0 bit field: bit 0 = `is_parity`, bits 1..=15 = `frame_num`.
fn decode_parity_frame(word: u16) -> (bool, u16) {
    let is_parity = (word >> 15) != 0;
    let frame_num = word & MAX_FRAME_NUM;
    (is_parity, frame_num)
}

/// Pack `is_parity` (MSB) and 15-bit `frame_num` into one big-endian `u16`.
fn encode_parity_frame(is_parity: bool, frame_num: u16) -> u16 {
    (u16::from(is_parity) << 15) | (frame_num & MAX_FRAME_NUM)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_datagram() -> FecDatagram {
        FecDatagram {
            seq_num: 0x1234_5678,
            is_parity: true,
            frame_num: 42,
            sizes_of_frames_encoding: 0x0123_4567_89ab_cdef,
            pos_in_frame: 7,
            stripe_pos_in_frame: 3,
            payload: Bytes::from_static(b"stripe-payload"),
        }
    }

    #[test]
    fn roundtrip_to_from_bytes() {
        let original = sample_datagram();
        let wire = original.to_bytes().unwrap();
        assert_eq!(wire.len(), original.wire_len());

        let parsed = FecDatagram::from_bytes(wire).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn from_bytes_zero_copy_payload() {
        let original = sample_datagram();
        let wire = original.to_bytes().unwrap();
        let parsed = FecDatagram::from_bytes(wire.clone()).unwrap();
        assert_eq!(
            parsed.payload.as_ptr(),
            wire.as_ptr().wrapping_add(FecDatagram::HEADER_LEN)
        );
    }

    #[test]
    fn wire_parts_shares_payload() {
        let original = sample_datagram();
        let payload_ptr = original.payload.as_ptr();
        let (_header, payload) = original.wire_parts().unwrap();
        assert_eq!(payload.as_ptr(), payload_ptr);
    }

    #[test]
    fn try_from_slice_roundtrip() {
        let original = sample_datagram();
        let wire: Bytes = original.clone().try_into().unwrap();
        let parsed = FecDatagram::try_from(wire).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn roundtrip_empty_payload() {
        let original = FecDatagram {
            payload: Bytes::new(),
            ..sample_datagram()
        };
        let parsed = FecDatagram::from_bytes(original.to_bytes().unwrap()).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn header_only_parse() {
        let wire = sample_datagram().to_bytes().unwrap();
        let header = FecDatagramHeader::try_from(wire.as_ref()).unwrap();
        assert_eq!(header.seq_num, 0x1234_5678);
        assert!(header.is_parity);
        assert_eq!(header.frame_num, 42);
    }

    #[test]
    fn parity_frame_bit_packing_matches_cpp() {
        assert_eq!(encode_parity_frame(true, 42), 0x802a);
        let (is_parity, frame_num) = decode_parity_frame(0x802a);
        assert!(is_parity);
        assert_eq!(frame_num, 42);

        let (is_parity, frame_num) = decode_parity_frame(encode_parity_frame(false, MAX_FRAME_NUM));
        assert!(!is_parity);
        assert_eq!(frame_num, MAX_FRAME_NUM);
    }

    #[test]
    fn wire_layout_is_big_endian() {
        let datagram = FecDatagram {
            seq_num: 1,
            is_parity: false,
            frame_num: 2,
            sizes_of_frames_encoding: 3,
            pos_in_frame: 4,
            stripe_pos_in_frame: 5,
            payload: Bytes::from_static(&[0xaa]),
        };
        let wire = datagram.to_bytes().unwrap();
        assert_eq!(
            wire.as_ref(),
            &[
                0, 0, 0, 1, // seq
                0, 2, // parity+frame
                0, 0, 0, 0, 0, 0, 0, 3, // sizes
                0, 4, // pos
                0, 5, // stripe pos
                0xaa,
            ]
        );
    }

    #[test]
    fn truncated_header_is_error() {
        let err = FecDatagram::from_bytes(Bytes::from_static(&[0u8; FecDatagram::HEADER_LEN - 1]))
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::Datagram(DatagramError::TruncatedHeader)
        ));
    }

    #[test]
    fn wire_frame_num_masks_to_15_bits() {
        let datagram = FecDatagram {
            frame_num: MAX_FRAME_NUM + 1,
            ..sample_datagram()
        };
        let wire = datagram.to_bytes().unwrap();
        let parsed = FecDatagram::from_bytes(wire).unwrap();
        assert_eq!(parsed.frame_num, 0);
    }
}
