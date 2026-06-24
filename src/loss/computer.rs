//! Build packet/frame loss bitmaps from received frames.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use super::info::LossInfo;
use crate::codec::ReceiveFrame;

pub(crate) struct LossComputer<'a> {
    loss_info: LossInfo,
    _pre_fec: bool,
    _frames: &'a VecDeque<ReceiveFrame>,
}

impl<'a> LossComputer<'a> {
    pub(crate) fn new(frames: &'a VecDeque<ReceiveFrame>, pre_fec: bool) -> Self {
        let mut computer = Self {
            loss_info: LossInfo {
                indices: Vec::new(),
                packet_losses: Vec::new(),
                frame_losses: Vec::new(),
            },
            _pre_fec: pre_fec,
            _frames: frames,
        };
        for frame in frames {
            computer
                .loss_info
                .indices
                .push(computer.loss_info.packet_losses.len() as u16);
            if !frame.has_pkts() {
                computer.empty_frame(frame, pre_fec);
            } else {
                computer.nonempty_frame(frame, pre_fec);
            }
        }
        computer
    }

    pub(crate) fn loss_info(self) -> LossInfo {
        self.loss_info
    }

    fn empty_frame(&mut self, frame: &ReceiveFrame, pre_fec: bool) {
        self.loss_info
            .frame_losses
            .push(pre_fec || !frame.is_decoded());
        self.loss_info.packet_losses.push(true);
    }

    fn nonempty_frame(&mut self, frame: &ReceiveFrame, pre_fec: bool) {
        self.loss_info.frame_losses.push(!frame.is_decoded());
        let pkt_losses = frame.packet_losses(pre_fec);
        for loss in &pkt_losses {
            self.loss_info.packet_losses.push(*loss);
            if pre_fec
                && *loss
                && let Some(last) = self.loss_info.frame_losses.last_mut()
            {
                *last = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use alloc::collections::VecDeque;

    use crate::codec::ReceiveFrame;
    use crate::loss::LossComputer;

    use crate::loss::test_common::{
        MTU, TestRng, generate_frame_pkts, get_missing, highest_missing_pos, make_frame,
        new_packetization, size_increment,
    };

    #[test]
    fn one_frame_pre_fec_synthetic() {
        use crate::datagram::FecDatagram;

        fn make_pkt(frame_num: u16, pos: u16, is_parity: bool, payload_len: usize) -> FecDatagram {
            FecDatagram {
                seq_num: u32::from(frame_num) * 100 + u32::from(pos),
                is_parity,
                frame_num,
                sizes_of_frames_encoding: 0,
                pos_in_frame: pos,
                stripe_pos_in_frame: pos,
                payload: bytes::Bytes::from(vec![0u8; payload_len]),
            }
        }

        let num_pkts = 5u16;
        for num_lost in 0..num_pkts {
            let mut missing = vec![false; num_pkts as usize];
            for i in 0..num_lost {
                missing[i as usize] = true;
            }
            let mut frame = ReceiveFrame::new(0, 0);
            for (i, &miss) in missing.iter().enumerate() {
                if !miss {
                    frame.add_pkt(make_pkt(0, i as u16, i >= 3, 100));
                }
            }
            let mut frames = VecDeque::new();
            frames.push_back(frame.clone());
            let info = LossComputer::new(&frames, true).loss_info();
            let highest = highest_missing_pos(&missing);
            assert_eq!(info.indices, vec![0]);
            assert_eq!(info.frame_losses, vec![true]);
            assert_eq!(info.packet_losses, missing[..=highest as usize].to_vec());

            frame.update_size(300);
            frames.clear();
            frames.push_back(frame.clone());
            let data_lost = missing.iter().enumerate().any(|(i, &m)| m && i < 3);
            let post = LossComputer::new(&frames, false).loss_info();
            assert_eq!(post.frame_losses, vec![data_lost]);
        }
    }

    #[test]
    fn one_frame_with_generator() {
        let mut rng = TestRng::seeded();
        let mut packetization = new_packetization();
        for state in 0..2u16 {
            let _ = state;
            let mut increment = 0usize;
            let mut size = MTU - 2;
            while size < 10 * MTU {
                let pkts = generate_frame_pkts(&mut packetization, 0, size);
                let num_pkts = pkts.len() as u8;
                for num_lost in 0..num_pkts {
                    let mut frame = make_frame(pkts[0].frame_num, false);
                    let missing = get_missing(num_pkts, num_lost, &mut rng);
                    for (i, pkt) in pkts.iter().enumerate() {
                        if !missing[i] {
                            frame.add_pkt(pkt.clone());
                        }
                    }
                    let mut frames0 = VecDeque::new();
                    frames0.push_back(frame.clone());
                    let loss0 = LossComputer::new(&frames0, true).loss_info();
                    let highest = highest_missing_pos(&missing);
                    assert_eq!(loss0.indices, vec![0]);
                    assert_eq!(loss0.frame_losses, vec![true]);
                    assert_eq!(loss0.packet_losses, missing[..=highest as usize].to_vec());

                    frame.update_size(size);
                    let data_lost = missing
                        .iter()
                        .zip(pkts.iter())
                        .any(|(&m, p)| m && !p.is_parity);
                    let mut frames1 = VecDeque::new();
                    frames1.push_back(frame.clone());
                    let post = LossComputer::new(&frames1, false).loss_info();
                    assert_eq!(post.frame_losses, vec![data_lost]);
                    let pre = LossComputer::new(&frames1, true).loss_info();
                    assert_eq!(pre.packet_losses.len(), highest as usize + 1);
                    for miss_pos in 0..=highest {
                        assert_eq!(
                            pre.packet_losses[miss_pos as usize],
                            missing[miss_pos as usize]
                        );
                    }
                    assert_eq!(pre.indices, vec![0]);
                }
                size += size_increment(increment);
                increment += 1;
            }
        }
    }

    #[test]
    fn multiple_frames() {
        let mut rng = TestRng::seeded();
        let mut packetization = new_packetization();
        for update_size in 0..3u16 {
            for state in 0..15u16 {
                for max_num_frames in 2..30u16 {
                    let mut packet_losses = Vec::new();
                    let mut frame_losses = Vec::new();
                    let mut indices = Vec::new();
                    let mut frames1 = VecDeque::new();
                    for frame_ind in 1..=max_num_frames {
                        let size = (rng.next_bounded(if state < 1 { MTU * 9 } else { MTU } as u32)
                            as u64)
                            + MTU
                            - 2;
                        let pkts = generate_frame_pkts(&mut packetization, frame_ind, size);
                        let num_pkts = pkts.len() as u8;
                        let num_lost = rng.next_bounded(u32::from(num_pkts)) as u8;
                        let missing = get_missing(num_pkts, num_lost, &mut rng);
                        let mut frame = make_frame(pkts[0].frame_num, false);
                        for (i, pkt) in pkts.iter().enumerate() {
                            if !missing[i] {
                                frame.add_pkt(pkt.clone());
                            }
                        }
                        if update_size > 0 {
                            frame.update_size(size);
                        }
                        let mut frames0 = VecDeque::new();
                        frames0.push_back(frame.clone());
                        let loss0 = LossComputer::new(&frames0, true).loss_info();
                        indices.push(packet_losses.len() as u16);
                        packet_losses.extend_from_slice(&loss0.packet_losses);

                        let mut last_seen = 0u16;
                        let mut first_loss = u16::MAX;
                        for (l, &m) in missing.iter().enumerate() {
                            if m {
                                first_loss = first_loss.min(l as u16);
                            } else {
                                last_seen = l as u16;
                            }
                        }
                        let any_data_lost = missing
                            .iter()
                            .zip(pkts.iter())
                            .any(|(&m, p)| m && !p.is_parity);
                        let loss = if update_size == 0 {
                            true
                        } else if update_size == 1 {
                            (num_lost > 0 && last_seen > first_loss) || any_data_lost
                        } else {
                            any_data_lost
                        };
                        frame_losses.push(loss);
                        frames1.push_back(frame);
                    }
                    let loss1 = LossComputer::new(&frames1, update_size < 2).loss_info();
                    if update_size < 2 {
                        assert_eq!(packet_losses, loss1.packet_losses);
                    }
                    assert_eq!(frame_losses, loss1.frame_losses);
                    assert_eq!(indices, loss1.indices);
                }
            }
        }
    }

    #[test]
    fn missing_frame_nums() {
        let mut rng = TestRng::seeded();
        let mut packetization = new_packetization();
        for state in 0..2u16 {
            let _ = state;
            for max_num_frames in 2..30u16 {
                let mut packet_losses = Vec::new();
                let mut frame_losses = Vec::new();
                let mut indices = Vec::new();
                let mut frames1 = VecDeque::new();
                for frame_ind in 1..=max_num_frames {
                    let size = (rng.next_bounded(if state < 1 { MTU * 9 } else { MTU } as u32)
                        as u64)
                        + MTU
                        - 2;
                    let pkts = generate_frame_pkts(&mut packetization, frame_ind, size);
                    let num_pkts = pkts.len() as u8;
                    let mut num_lost = rng.next_bounded(u32::from(num_pkts)) as u8;
                    let mut missing = get_missing(num_pkts, num_lost, &mut rng);
                    if missing.last().copied().unwrap_or(false) {
                        num_lost = num_lost.saturating_sub(1);
                    }
                    if let Some(last) = missing.last_mut() {
                        *last = false;
                    }
                    let mut frame = make_frame(pkts[0].frame_num, false);
                    for (i, pkt) in pkts.iter().enumerate() {
                        if !missing[i] {
                            frame.add_pkt(pkt.clone());
                        }
                    }
                    frame.update_size(size);
                    let next_index = packet_losses.len() as u16;
                    let (next_packet_losses, next_frame_loss) = if frame_ind == 1
                        || frame_ind == max_num_frames
                        || rng.next_bounded(2) == 0
                    {
                        let mut frames0 = VecDeque::new();
                        frames0.push_back(frame.clone());
                        let loss0 = LossComputer::new(&frames0, true).loss_info();
                        frames1.push_back(frame);
                        (loss0.packet_losses, num_lost > 0)
                    } else {
                        frames1.push_back(make_frame(pkts[0].frame_num, true));
                        (vec![true], true)
                    };
                    indices.push(next_index);
                    frame_losses.push(next_frame_loss);
                    packet_losses.extend(next_packet_losses);
                }
                let loss1 = LossComputer::new(&frames1, true).loss_info();
                assert_eq!(packet_losses, loss1.packet_losses);
                assert_eq!(frame_losses, loss1.frame_losses);
                assert_eq!(indices, loss1.indices);
            }
        }
    }

    #[test]
    fn post_fec() {
        let mut rng = TestRng::seeded();
        let mut packetization = new_packetization();
        for t in (0..4).step_by(1) {
            let mut increment = 0usize;
            let mut size = MTU - 2;
            while size < 3 * MTU {
                let mut num_frames = 0u16;
                while num_frames <= 8 * t {
                    let mut frames = VecDeque::new();
                    let mut expected_frame_losses = Vec::new();
                    for frame_num in 0..num_frames {
                        let frame_size = size + u64::from(rng.next_bounded(MTU as u32));
                        let pkts = generate_frame_pkts(&mut packetization, frame_num, frame_size);
                        let num_pkts = pkts.len() as u8;
                        let empty_frame = rng.next_bounded(2) == 0;
                        let mut frame = make_frame(pkts[0].frame_num, empty_frame);
                        let missing = get_missing(
                            num_pkts,
                            rng.next_bounded(u32::from(num_pkts / 4 + 1)) as u8,
                            &mut rng,
                        );
                        for (i, pkt) in pkts.iter().enumerate() {
                            if !missing[i] && !empty_frame {
                                frame.add_pkt(pkt.clone());
                            }
                        }
                        let loss = rng.next_bounded(2) == 0;
                        expected_frame_losses.push(loss);
                        if !loss {
                            frame.set_decoded();
                        }
                        frames.push_back(frame);
                    }
                    let loss_info = LossComputer::new(&frames, false).loss_info();
                    assert_eq!(loss_info.frame_losses, expected_frame_losses);
                    if t == 0 {
                        break;
                    }
                    num_frames += t;
                }
                size += size_increment(increment);
                increment += 1;
            }
        }
    }

    #[test]
    fn empty_frame() {
        let mut frames = VecDeque::new();
        frames.push_back(make_frame(3, true));
        let info = LossComputer::new(&frames, true).loss_info();
        assert_eq!(info.indices, vec![0]);
        assert_eq!(info.frame_losses, vec![true]);
        assert_eq!(info.packet_losses, vec![true]);
    }

    #[test]
    fn multiple_frames_indices() {
        let mut frames = VecDeque::new();
        let mut expected_indices = Vec::new();
        let mut expected_packet = Vec::new();
        let mut expected_frame = Vec::new();

        for frame_num in 0..4u16 {
            expected_indices.push(expected_packet.len() as u16);
            let mut frame = ReceiveFrame::new(u64::from(frame_num), frame_num);
            use crate::datagram::FecDatagram;
            frame.add_pkt(FecDatagram {
                seq_num: u32::from(frame_num) * 100,
                is_parity: false,
                frame_num,
                sizes_of_frames_encoding: 0,
                pos_in_frame: 0,
                stripe_pos_in_frame: 0,
                payload: bytes::Bytes::from(vec![0u8; 200]),
            });
            frame.update_size(200);
            frames.push_back(frame);
            expected_packet.push(false);
            expected_frame.push(false);
        }

        let info = LossComputer::new(&frames, true).loss_info();
        assert_eq!(info.indices, expected_indices);
        assert_eq!(info.packet_losses, expected_packet);
        assert_eq!(info.frame_losses, expected_frame);
    }

    #[test]
    fn dummy_frame() {
        use crate::datagram::FecDatagram;

        let mut frames = VecDeque::new();
        frames.push_back(ReceiveFrame::new(0, 0));
        frames.back_mut().unwrap().add_pkt(FecDatagram {
            seq_num: 0,
            is_parity: false,
            frame_num: 0,
            sizes_of_frames_encoding: 0,
            pos_in_frame: 0,
            stripe_pos_in_frame: 0,
            payload: bytes::Bytes::from(vec![0u8; 100]),
        });
        frames.push_back(make_frame(1, true));
        let info = LossComputer::new(&frames, true).loss_info();
        assert_eq!(info.indices, vec![0, 1]);
        assert_eq!(info.frame_losses, vec![true, true]);
        assert_eq!(info.packet_losses, vec![false, true]);
    }
}
