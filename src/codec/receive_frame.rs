//! Per-frame receive buffer.

use alloc::vec::Vec;

use crate::datagram::FecDatagram;

#[derive(Debug, Clone)]
pub(crate) struct ReceiveFrame {
    pub session_index: u64,
    pub frame_num: u16,
    pkts: Vec<FecDatagram>,
    is_missing: Vec<bool>,
    pub frame_size: Option<u64>,
    is_dummy: bool,
    is_decoded: bool,
    delivered: bool,
    fec_recovered: bool,
    first_sqn: Option<u32>,
    occupied: bool,
}

impl ReceiveFrame {
    pub(crate) fn new(session_index: u64, frame_num: u16) -> Self {
        Self {
            session_index,
            frame_num,
            pkts: Vec::new(),
            is_missing: Vec::new(),
            frame_size: None,
            is_dummy: false,
            is_decoded: false,
            delivered: false,
            fec_recovered: false,
            first_sqn: None,
            occupied: false,
        }
    }

    pub(crate) fn is_occupied(&self) -> bool {
        self.occupied
    }

    pub(crate) fn is_delivered(&self) -> bool {
        self.delivered
    }

    pub(crate) fn mark_delivered(&mut self, fec_recovered: bool) {
        self.delivered = true;
        if fec_recovered {
            self.fec_recovered = true;
        }
    }

    /// Reuse this slot for a new frame number, keeping `pkts` / `is_missing` capacity.
    pub(crate) fn reset_for(&mut self, session_index: u64, frame_num: u16, placeholder: bool) {
        self.session_index = session_index;
        self.frame_num = frame_num;
        self.pkts.clear();
        self.is_missing.clear();
        self.frame_size = None;
        self.is_dummy = placeholder;
        self.is_decoded = false;
        self.delivered = false;
        self.fec_recovered = false;
        self.first_sqn = None;
        self.occupied = true;
    }

    /// Move frame state out for feedback; leave an empty shell that keeps vector capacity.
    pub(crate) fn take_into_feedback(&mut self) -> ReceiveFrame {
        let mut taken = Self::new(self.session_index, self.frame_num);
        taken.session_index = self.session_index;
        taken.frame_num = self.frame_num;
        core::mem::swap(&mut taken.pkts, &mut self.pkts);
        core::mem::swap(&mut taken.is_missing, &mut self.is_missing);
        taken.frame_size = self.frame_size;
        taken.is_dummy = self.is_dummy;
        taken.is_decoded = self.is_decoded;
        taken.delivered = self.delivered;
        taken.fec_recovered = self.fec_recovered;
        taken.first_sqn = self.first_sqn;
        taken.occupied = true;
        self.reset_preserve_shell();
        taken
    }

    fn reset_preserve_shell(&mut self) {
        self.pkts.clear();
        self.is_missing.clear();
        self.frame_size = None;
        self.is_dummy = false;
        self.is_decoded = false;
        self.delivered = false;
        self.fec_recovered = false;
        self.first_sqn = None;
        self.occupied = false;
    }

    pub(crate) fn has_pkts(&self) -> bool {
        !self.is_dummy && !self.pkts.is_empty()
    }

    pub(crate) fn is_decoded(&self) -> bool {
        self.is_decoded || self.all_data_received()
    }

    pub(crate) fn set_decoded(&mut self) {
        self.is_decoded = true;
    }

    pub(crate) fn update_size(&mut self, size: u64) {
        self.frame_size = Some(size);
    }

    pub(crate) fn seq_nums(&self) -> impl Iterator<Item = u32> + '_ {
        self.pkts.iter().map(|p| p.seq_num)
    }

    pub(crate) fn add_pkt(&mut self, pkt: FecDatagram) {
        self.is_dummy = false;
        self.first_sqn = Some(pkt.seq_num.saturating_sub(u32::from(pkt.pos_in_frame)));
        self.pad_for_pkt_pos(
            pkt.pos_in_frame,
            self.pkts.last().map(|p| p.is_parity).unwrap_or(false),
        );
        self.pkts.push(pkt);
        self.is_missing.push(false);
    }

    /// Per-packet loss flags for [`LossComputer`](crate::loss::LossComputer).
    pub(crate) fn packet_losses(&self, pre_fec: bool) -> Vec<bool> {
        let mut out = Vec::with_capacity(self.is_missing.len());
        for (i, &loss) in self.is_missing.iter().enumerate() {
            let mut l = loss;
            if !pre_fec
                && self.is_decoded()
                && let Some(pkt) = self.pkts.get(i)
            {
                l |= !pkt.is_parity;
            }
            out.push(l);
        }
        out
    }

    pub(crate) fn all_data_received(&self) -> bool {
        let Some(size) = self.frame_size else {
            return false;
        };
        let mut received = 0u64;
        for (i, pkt) in self.pkts.iter().enumerate() {
            if !self.is_missing[i] && !pkt.is_parity {
                received += pkt.payload.len() as u64;
            }
        }
        received >= size
    }

    fn pad_for_pkt_pos(&mut self, packet_pos: u16, is_parity: bool) {
        while packet_pos > self.pkts.len() as u16 {
            self.pkts.push(empty_pkt(self, is_parity));
            self.is_missing.push(true);
        }
    }
}

fn empty_pkt(frame: &ReceiveFrame, is_parity: bool) -> FecDatagram {
    FecDatagram {
        seq_num: frame.first_sqn.unwrap_or(0) + frame.pkts.len() as u32,
        is_parity,
        frame_num: frame.frame_num,
        sizes_of_frames_encoding: frame.frame_size.unwrap_or(0),
        pos_in_frame: frame.pkts.len() as u16,
        stripe_pos_in_frame: frame.pkts.len() as u16,
        payload: bytes::Bytes::new(),
    }
}
