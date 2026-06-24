//! FEC sender session: source frames into data and parity datagrams.

use alloc::vec::Vec;

use bytes::{BufMut, Bytes, BytesMut};

use super::session::wire_frame_num;
use crate::config::Config;
use crate::datagram::FecDatagram;
use crate::error::{CodecError, Error, InternalError, Result};
use crate::fec::{MultiFecHeaderCode, StreamingCode};
use crate::feedback::Feedback;
use crate::packetization::{PacketSpec, StreamingPacketization};

/// FEC sender: frames → datagrams, driven by receiver [`Feedback`].
pub struct Encoder {
    config: Config,
    current_feedback: Feedback,
    code: StreamingCode,
    header_code: MultiFecHeaderCode,
    packetization: StreamingPacketization,
    sqn: u32,
    next_session_index: u64,
    datagram_scratch: Vec<FecDatagram>,
    data_pkts_scratch: Vec<FecDatagram>,
    parity_sizes_scratch: Vec<u16>,
    parity_scratch: Vec<u8>,
}

impl Encoder {
    /// Start with no parity until the first [`Feedback`] arrives.
    pub fn new(config: Config) -> Result<Self> {
        let max_parity_bytes =
            config.max_fec_stripes() as usize * config.max_pkt_size.get() as usize;
        Ok(Self {
            config: config.clone(),
            current_feedback: Feedback::default(),
            code: config.streaming_code()?,
            header_code: config.header_code()?,
            packetization: StreamingPacketization::from_config(&config),
            sqn: 0,
            next_session_index: 0,
            datagram_scratch: Vec::new(),
            data_pkts_scratch: Vec::new(),
            parity_sizes_scratch: Vec::new(),
            parity_scratch: Vec::with_capacity(max_parity_bytes),
        })
    }

    /// Current redundancy level (last applied feedback).
    pub fn redundancy(&self) -> Feedback {
        self.current_feedback
    }

    /// Session configuration (read-only).
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Apply receiver feedback; affects subsequent [`encode_payload`](Self::encode_payload) calls.
    pub fn apply_feedback(&mut self, feedback: Feedback) {
        self.current_feedback = feedback;
    }

    /// Encode one source payload into data + parity datagrams.
    ///
    /// Frame numbers are assigned monotonically by the encoder session.
    /// The returned slice is valid until the next call to [`encode_payload`](Self::encode_payload).
    pub fn encode_payload(&mut self, payload: impl Into<Bytes>) -> Result<&[FecDatagram]> {
        let payload = payload.into();
        if payload.is_empty() {
            return Err(Error::Codec(CodecError::PayloadTooLarge {
                size: 0,
                max: self.config.max_frame_size,
            }));
        }
        if payload.len() as u32 > self.config.max_frame_size {
            return Err(Error::Codec(CodecError::PayloadTooLarge {
                size: payload.len() as u32,
                max: self.config.max_frame_size,
            }));
        }

        let session_index = self.next_session_index;
        let frame_num = wire_frame_num(session_index);
        let frame_size = payload.len() as u32;
        let specs = self
            .packetization
            .plan_frame(frame_size, self.current_feedback);

        self.build_data_datagrams(&payload, &specs, frame_num, frame_size)?;

        self.parity_sizes_scratch.clear();
        self.parity_sizes_scratch
            .extend(specs.iter().filter(|s| s.is_parity).map(|s| s.payload_len));

        self.code.encode(
            &self.data_pkts_scratch,
            &self.parity_sizes_scratch,
            frame_size,
            &mut self.parity_scratch,
        )?;

        self.interleave_datagrams(&specs, frame_num, frame_size)?;

        self.next_session_index += 1;
        Ok(&self.datagram_scratch)
    }

    /// Advance the session counter without encoding (tests / harness only).
    #[cfg(test)]
    pub fn set_next_session_index(&mut self, session_index: u64) {
        self.next_session_index = session_index;
    }

    fn build_data_datagrams(
        &mut self,
        payload: &Bytes,
        specs: &[PacketSpec],
        frame_num: u16,
        frame_size: u32,
    ) -> Result<()> {
        let frame_size_u64 = u64::from(frame_size);
        let stripe_size = self.config.stripe_size() as u16;
        let datagram_size = specs
            .iter()
            .find(|s| !s.is_parity)
            .map(|s| s.payload_len)
            .ok_or(Error::Internal(InternalError::InvariantViolated))?;
        let data_count = specs.iter().filter(|s| !s.is_parity).count();
        self.data_pkts_scratch.clear();
        self.data_pkts_scratch.reserve(data_count);

        let mut stripe_pos = 0u16;
        let mut offset = 0usize;
        let mut data_index = 0usize;

        for spec in specs {
            if spec.is_parity {
                continue;
            }
            let is_last = data_index + 1 == data_count;
            let partial = is_last && !frame_size.is_multiple_of(u32::from(datagram_size));
            let content_len = if partial {
                (frame_size % u32::from(datagram_size)) as usize
            } else {
                datagram_size as usize
            };

            let end = (offset + content_len).min(payload.len());
            let chunk = if partial {
                let mut buf = BytesMut::with_capacity(datagram_size as usize);
                if offset < payload.len() {
                    buf.extend_from_slice(&payload[offset..end]);
                }
                let padding = datagram_size as usize - content_len;
                if padding > 0 {
                    buf.put_bytes(0, padding);
                }
                buf.freeze()
            } else {
                payload.slice(offset..offset + datagram_size as usize)
            };
            offset = end;

            let sizes_encoding = self
                .header_code
                .encode_sizes_of_frames(frame_size_u64, data_index as u16)?;

            self.data_pkts_scratch.push(FecDatagram {
                seq_num: self.sqn,
                is_parity: false,
                frame_num,
                sizes_of_frames_encoding: sizes_encoding,
                pos_in_frame: data_index as u16,
                stripe_pos_in_frame: stripe_pos,
                payload: chunk,
            });
            self.sqn += 1;
            stripe_pos += datagram_size / stripe_size;
            data_index += 1;
        }
        Ok(())
    }

    fn interleave_datagrams(
        &mut self,
        specs: &[PacketSpec],
        frame_num: u16,
        frame_size: u32,
    ) -> Result<()> {
        let frame_size_u64 = u64::from(frame_size);
        let stripe_size = self.config.stripe_size() as u16;
        self.datagram_scratch.clear();
        self.datagram_scratch.reserve(specs.len());

        let parity_cap = self.parity_scratch.capacity();
        let parity_bytes = Bytes::from(core::mem::take(&mut self.parity_scratch));
        self.parity_scratch = Vec::with_capacity(parity_cap.max(parity_bytes.len()));

        let mut data_iter = self.data_pkts_scratch.iter();
        let mut parity_offset = 0usize;
        let num_data = specs.iter().filter(|s| !s.is_parity).count();
        let mut par_count = 0usize;
        let mut parity_stripe_pos = 0u16;

        for spec in specs {
            if spec.is_parity {
                let size = spec.payload_len as usize;
                let payload = parity_bytes.slice(parity_offset..parity_offset + size);
                parity_offset += size;
                par_count += 1;
                let pos = (num_data + par_count - 1) as u16;
                let sizes_encoding = self
                    .header_code
                    .encode_sizes_of_frames(frame_size_u64, pos)?;
                self.datagram_scratch.push(FecDatagram {
                    seq_num: self.sqn,
                    is_parity: true,
                    frame_num,
                    sizes_of_frames_encoding: sizes_encoding,
                    pos_in_frame: pos,
                    stripe_pos_in_frame: parity_stripe_pos,
                    payload,
                });
                parity_stripe_pos += spec.payload_len / stripe_size;
                self.sqn += 1;
            } else {
                let pkt = data_iter
                    .next()
                    .ok_or(Error::Internal(InternalError::InvariantViolated))?;
                self.datagram_scratch.push(pkt.clone());
            }
        }
        Ok(())
    }
}
