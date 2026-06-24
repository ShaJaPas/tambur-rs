//! RFC burst/gap metrics from loss bitmaps.

mod burst;
mod gap;
mod multi_frame;

use alloc::collections::BTreeMap;

use super::LossMetrics;
use super::info::LossInfo;

pub(crate) struct LossMetricComputer {
    loss_info: LossInfo,
    redundancy_wire_byte: f32,
    g_min_packet: u16,
    g_min_frame: u16,
    multi_frame_burst: u16,
    multi_frame_minimal_guardspace: u16,
}

impl LossMetricComputer {
    pub(crate) fn new(
        loss_info: LossInfo,
        redundancy_wire_byte: u8,
        g_min_packet: u8,
        g_min_frame: u8,
        multi_frame_burst: u8,
        multi_frame_minimal_guardspace: u8,
    ) -> Self {
        Self {
            loss_info,
            redundancy_wire_byte: f32::from(redundancy_wire_byte),
            g_min_packet: u16::from(g_min_packet),
            g_min_frame: u16::from(g_min_frame),
            multi_frame_burst: u16::from(multi_frame_burst),
            multi_frame_minimal_guardspace: u16::from(multi_frame_minimal_guardspace),
        }
    }

    pub(crate) fn compute_loss_metrics(&self) -> LossMetrics {
        let packet_rfc =
            gap::get_rfc_densities_inner(&self.loss_info.packet_losses, self.g_min_packet);
        let frame_rfc =
            gap::get_rfc_densities_inner(&self.loss_info.frame_losses, self.g_min_frame);
        let (bursts, guardspaces) =
            burst::get_bursts_and_guardspaces(&self.loss_info.packet_losses);
        let (frame_bursts, frame_guardspaces) =
            burst::get_bursts_and_guardspaces(&self.loss_info.frame_losses);

        LossMetrics {
            burst_density: packet_rfc.0,
            frame_burst_density: frame_rfc.0,
            gap_density: packet_rfc.1,
            frame_gap_density: frame_rfc.1,
            mean_burst_length: Self::compute_mean_length(&bursts),
            mean_frame_burst_length: Self::compute_mean_length(&frame_bursts),
            loss_fraction: Self::compute_loss_fraction(&self.loss_info.packet_losses),
            frame_loss_fraction: Self::compute_loss_fraction(&self.loss_info.frame_losses),
            multi_frame_loss_fraction: multi_frame::compute_multi_frame_loss_fraction_inner(
                &self.loss_info.packet_losses,
                &self.loss_info.indices,
                self.multi_frame_burst,
            ),
            mean_guardspace_length: Self::compute_mean_length(&guardspaces),
            mean_frame_guardspace_length: Self::compute_mean_length(&frame_guardspaces),
            multi_frame_insufficient_guardspace: Self::compute_multi_frame_insufficient_guardspace(
                &frame_guardspaces,
                self.multi_frame_minimal_guardspace,
            ),
            redundancy_wire_byte: self.redundancy_wire_byte,
        }
    }

    pub(crate) fn compute_mean_length(stats: &BTreeMap<u16, u16>) -> f32 {
        let mut total = 0f32;
        let mut count = 0f32;
        for (&len, &occ) in stats {
            total += f32::from(len) * f32::from(occ);
            count += f32::from(occ);
        }
        if count > 0.0 { total / count } else { 0.0 }
    }

    pub(crate) fn compute_loss_fraction(losses: &[bool]) -> f32 {
        if losses.is_empty() {
            return 0.0;
        }
        let num_lost = losses.iter().filter(|&&l| l).count() as f32;
        num_lost / losses.len() as f32
    }

    pub(crate) fn compute_multi_frame_insufficient_guardspace(
        guardspaces: &BTreeMap<u16, u16>,
        multi_frame_minimal_guardspace: u16,
    ) -> f32 {
        if guardspaces.is_empty() {
            return 0.0;
        }
        let mut smaller = 0f32;
        let mut total = 0f32;
        for (&len, &occ) in guardspaces {
            if len < multi_frame_minimal_guardspace {
                smaller += f32::from(occ);
            }
            total += f32::from(occ);
        }
        smaller / total
    }

    #[cfg(test)]
    pub(crate) fn get_bursts_and_guardspaces(
        losses: &[bool],
    ) -> (BTreeMap<u16, u16>, BTreeMap<u16, u16>) {
        burst::get_bursts_and_guardspaces(losses)
    }

    #[cfg(test)]
    pub(crate) fn get_rfc_densities(&self, losses: &[bool], g_min: u16) -> (f32, f32) {
        gap::get_rfc_densities_inner(losses, g_min)
    }

    #[cfg(test)]
    pub(crate) fn consecutive_burst_info(
        &self,
        losses: &[bool],
        frame_indices: &[u16],
    ) -> (u16, u16) {
        multi_frame::consecutive_burst_info_inner(losses, frame_indices)
    }

    #[cfg(test)]
    pub(crate) fn get_positions_consecutive_losses(
        &self,
        losses: &[bool],
        frame_indices: &[u16],
        min_frames: u16,
    ) -> (Vec<u16>, Vec<u16>) {
        multi_frame::get_positions_consecutive_losses_inner(losses, frame_indices, min_frames)
    }

    #[cfg(test)]
    pub(crate) fn compute_multi_frame_loss_fraction(
        &self,
        losses: &[bool],
        frame_indices: &[u16],
        min_frames: u16,
    ) -> f32 {
        multi_frame::compute_multi_frame_loss_fraction_inner(losses, frame_indices, min_frames)
    }

    #[cfg(test)]
    pub(crate) fn get_window_metrics(&self, loss_info: &LossInfo) -> (f32, f32, f32) {
        let num_pkts = loss_info.packet_losses.len();
        if num_pkts == 0 {
            return (0.0, 0.0, 0.0);
        }
        let num_missing = loss_info.packet_losses.iter().filter(|&&l| l).count() as f32;
        let (bursts, _) = burst::get_bursts_and_guardspaces(&loss_info.packet_losses);
        let max_burst = bursts.keys().next_back().copied().unwrap_or(0);
        (num_pkts as f32, num_missing, f32::from(max_burst))
    }
}

#[cfg(test)]
mod tests {

    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;

    use crate::loss::LossMetricComputer;
    use crate::loss::info::LossInfo;

    const ERR: f32 = 0.000_01;

    fn computer(qr: u8) -> LossMetricComputer {
        LossMetricComputer::new(LossInfo::default(), qr, 1, 3, 2, 3)
    }

    fn map(pairs: &[(u16, u16)]) -> BTreeMap<u16, u16> {
        pairs.iter().copied().collect()
    }

    fn assert_near(a: f32, b: f32) {
        assert!((a - b).abs() < ERR, "expected {b}, got {a}");
    }

    fn assert_bursts_guardspaces(
        losses: &[bool],
        bursts: &[(u16, u16)],
        guardspaces: &[(u16, u16)],
    ) {
        let (b, g) = LossMetricComputer::get_bursts_and_guardspaces(losses);
        assert_eq!(b, map(bursts));
        assert_eq!(g, map(guardspaces));
    }

    #[test]
    fn bursts_guardspaces_no_loss() {
        assert_bursts_guardspaces(&[false; 20], &[], &[(20, 1)]);
    }

    #[test]
    fn bursts_guardspaces_all_loss() {
        assert_bursts_guardspaces(&[true; 10], &[(10, 1)], &[]);
    }

    #[test]
    fn bursts_guardspaces_begin_burst_one_loss() {
        assert_bursts_guardspaces(
            &[true, false, false, false, false, false],
            &[(1, 1)],
            &[(5, 1)],
        );
    }

    #[test]
    fn bursts_guardspaces_begin_burst_multiple_losses() {
        assert_bursts_guardspaces(
            &[true, true, true, false, false, false, false, false],
            &[(3, 1)],
            &[(5, 1)],
        );
    }

    #[test]
    fn bursts_guardspaces_middle_burst_one_loss() {
        assert_bursts_guardspaces(
            &[false, false, false, true, false, false, false],
            &[(1, 1)],
            &[(3, 2)],
        );
    }

    #[test]
    fn bursts_guardspaces_middle_burst_multiple_losses() {
        assert_bursts_guardspaces(
            &[
                false, true, true, true, true, false, false, false, false, false,
            ],
            &[(4, 1)],
            &[(5, 1), (1, 1)],
        );
    }

    #[test]
    fn bursts_guardspaces_end_burst_one_loss() {
        assert_bursts_guardspaces(
            &[false, false, false, false, false, true],
            &[(1, 1)],
            &[(5, 1)],
        );
    }

    #[test]
    fn bursts_guardspaces_end_burst_multiple_losses() {
        assert_bursts_guardspaces(&[false, true, true], &[(2, 1)], &[(1, 1)]);
    }

    #[test]
    fn bursts_guardspaces_two_bursts_middle() {
        assert_bursts_guardspaces(
            &[
                false, false, false, true, true, true, true, false, true, true, true, true, false,
                false, false, false, false, false, false, false, false, false,
            ],
            &[(4, 2)],
            &[(3, 1), (1, 1), (10, 1)],
        );
    }

    #[test]
    fn bursts_guardspaces_two_bursts_begin_end() {
        assert_bursts_guardspaces(
            &[
                true, true, false, false, false, false, false, false, false, false, true,
            ],
            &[(2, 1), (1, 1)],
            &[(8, 1)],
        );
    }

    #[test]
    fn bursts_guardspaces_two_bursts_begin_middle() {
        assert_bursts_guardspaces(
            &[
                true, true, false, false, false, false, true, true, true, true, false, false,
                false, false, false,
            ],
            &[(2, 1), (4, 1)],
            &[(5, 1), (4, 1)],
        );
    }

    #[test]
    fn bursts_guardspaces_two_bursts_middle_end() {
        assert_bursts_guardspaces(
            &[
                false, false, false, false, false, true, true, true, false, false, false, false,
                false, false, true, true, true, true,
            ],
            &[(3, 1), (4, 1)],
            &[(6, 1), (5, 1)],
        );
    }

    #[test]
    fn rfc_only_loss() {
        let g_min = 16u16;
        let losses = vec![true; 100];
        let vals = computer(0).get_rfc_densities(&losses, g_min);
        assert_eq!(vals, (1.0, 0.0));
    }

    #[test]
    fn rfc_no_loss() {
        let g_min = 16u16;
        let losses = vec![false; 100];
        let vals = computer(0).get_rfc_densities(&losses, g_min);
        assert_eq!(vals, (0.0, 0.0));
    }

    #[test]
    fn rfc_two_bursts_edges() {
        let g_min = 16u16;
        let losses = vec![
            true, true, true, true, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, true, true, true, true, true,
        ];
        let vals = computer(0).get_rfc_densities(&losses, g_min);
        assert_near(vals.0, 1.0);
        assert_near(vals.1, 0.0);
    }

    #[test]
    fn rfc_beginning_burst() {
        let g_min = 16u16;
        let c = computer(0);
        let losses = vec![
            false, false, false, false, false, false, false, false, true, true, true, true, true,
            true, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false,
        ];
        let vals = c.get_rfc_densities(&losses, g_min);
        assert_near(vals.0, 1.0);
        assert_near(vals.1, 0.0);
        let losses2 = vec![
            true, true, true, true, true, true, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
        ];
        let vals2 = c.get_rfc_densities(&losses2, g_min);
        assert_near(vals2.0, 1.0);
        assert_near(vals2.1, 0.0);
    }

    #[test]
    fn rfc_end_burst() {
        let g_min = 16u16;
        let c = computer(0);
        let losses = vec![
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, true, true, true, true, true, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false,
        ];
        let vals = c.get_rfc_densities(&losses, g_min);
        assert_near(vals.0, 1.0);
        assert_near(vals.1, 0.0);
        let losses2 = losses[..g_min as usize + 5].to_vec();
        let vals2 = c.get_rfc_densities(&losses2, g_min);
        assert_near(vals2.0, 1.0);
        assert_near(vals2.1, 0.0);
    }

    #[test]
    fn rfc_isolated_middle() {
        let g_min = 16u16;
        let losses = vec![
            false, false, false, false, false, false, false, false, true, false, false, false,
            false, false, false, false, false,
        ];
        let vals = computer(0).get_rfc_densities(&losses, g_min);
        assert_near(vals.0, 0.0);
        assert_near(vals.1, 1.0 / (g_min as f32 + 1.0));
    }

    #[test]
    fn rfc_isolated_beginning() {
        let g_min = 16u16;
        let c = computer(0);
        let mut losses = vec![false; 8];
        losses.push(true);
        losses.extend(vec![false; 500]);
        let vals = c.get_rfc_densities(&losses, g_min);
        assert_near(vals.0, 0.0);
        assert_near(vals.1, 1.0 / (g_min as f32 / 2.0 + 500.0 + 1.0));
        let losses2 = losses[(g_min / 2) as usize..].to_vec();
        let vals2 = c.get_rfc_densities(&losses2, g_min);
        assert_near(vals2.0, 0.0);
        assert_near(vals2.1, 1.0 / 501.0);
    }

    #[test]
    fn rfc_isolated_end() {
        let g_min = 16u16;
        let c = computer(0);
        let mut losses = vec![false; 500];
        losses.push(true);
        losses.extend(vec![false; (g_min / 2) as usize]);
        let vals = c.get_rfc_densities(&losses, g_min);
        assert_near(vals.0, 0.0);
        assert_near(vals.1, 1.0 / (500.0 + 1.0 + g_min as f32 / 2.0));
        let losses2 = losses[..501].to_vec();
        let vals2 = c.get_rfc_densities(&losses2, g_min);
        assert_near(vals2.0, 0.0);
        assert_near(vals2.1, 1.0 / 501.0);
    }

    #[test]
    fn rfc_burst_isolated_burst() {
        let g_min = 16u16;
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, true, true, false, false, false,
                false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 1.0);
            assert_near(vals.1, 0.018_518_519);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, true, true, false, false, false, false,
                false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.375);
            assert_near(vals.1, 0.0);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, true, true, false, false, false, false,
                false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.375);
            assert_near(vals.1, 0.0);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.230_769_23);
            assert_near(vals.1, 0.0);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, true, true, false, false, false,
                false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 1.0);
            assert_near(vals.1, 0.018_518_519);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, true, true, false, false, false, false,
                false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.375);
            assert_near(vals.1, 0.0);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, true, true, false, false, false, false,
                false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.375);
            assert_near(vals.1, 0.0);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.1, 0.0);
        }
    }

    #[test]
    fn rfc_isolated_burst_isolated() {
        let g_min = 16u16;
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, true, true,
                true, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 1.0);
            assert_near(vals.1, 0.030_303_031);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, true, true, true, true,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.285_714_3);
            assert_near(vals.1, 0.020_408_163);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, true, true,
                true, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.285_714_3);
            assert_near(vals.1, 0.020_408_163);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, true, true, true, true,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.189_189_2);
            assert_near(vals.1, 0.0);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, true, true,
                true, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 1.0);
            assert_near(vals.1, 0.030_303_031);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, true, true, true, true,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.285_714_3);
            assert_near(vals.1, 0.020_408_163);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, true, true,
                true, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.285_714_3);
            assert_near(vals.1, 0.020_408_163);
        }
        {
            let losses = [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, true, true, true, true,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false,
            ];
            let vals = computer(0).get_rfc_densities(&losses, g_min);
            assert_near(vals.0, 0.189_189_2);
            assert_near(vals.1, 0.0);
        }
    }

    #[test]
    fn loss_fraction_cases() {
        assert_eq!(LossMetricComputer::compute_loss_fraction(&[]), 0.0);
        assert_eq!(
            LossMetricComputer::compute_loss_fraction(&[true, false, true, false]),
            0.5
        );
        assert_eq!(LossMetricComputer::compute_loss_fraction(&[true; 4]), 1.0);
        assert_eq!(LossMetricComputer::compute_loss_fraction(&[false; 4]), 0.0);
        assert_eq!(
            LossMetricComputer::compute_loss_fraction(&[
                false, false, false, false, true, true, false, true, true, true
            ]),
            0.5
        );
    }

    #[test]
    fn mean_length_cases() {
        assert_near(LossMetricComputer::compute_mean_length(&map(&[])), 0.0);
        for i in 1..10u16 {
            for j in 3..8u16 {
                let stats = map(&[(i, j)]);
                assert_near(LossMetricComputer::compute_mean_length(&stats), i as f32);
            }
        }
        for i1 in 1..10u16 {
            for i2 in 3..8u16 {
                for j1 in 1..10u16 {
                    if j1 == i1 {
                        continue;
                    }
                    for j2 in 3..8u16 {
                        let stats = map(&[(i1, i2), (j1, j2)]);
                        let expected = (f32::from(i1) * f32::from(i2)
                            + f32::from(j1) * f32::from(j2))
                            / f32::from(i2 + j2);
                        assert_near(LossMetricComputer::compute_mean_length(&stats), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn multi_frame_insufficient_guardspace() {
        assert_eq!(
            LossMetricComputer::compute_multi_frame_insufficient_guardspace(&map(&[]), 3),
            0.0
        );
        assert_eq!(
            LossMetricComputer::compute_multi_frame_insufficient_guardspace(
                &map(&[(1, 2), (2, 3)]),
                3
            ),
            1.0
        );
        assert_eq!(
            LossMetricComputer::compute_multi_frame_insufficient_guardspace(
                &map(&[(3, 2), (4, 3)]),
                3
            ),
            0.0
        );
        assert_eq!(
            LossMetricComputer::compute_multi_frame_insufficient_guardspace(
                &map(&[(1, 5), (2, 7), (3, 1), (4, 3)]),
                3
            ),
            0.75
        );
        assert_eq!(
            LossMetricComputer::compute_multi_frame_insufficient_guardspace(
                &map(&[(1, 4), (2, 8), (3, 1), (4, 3)]),
                2
            ),
            0.25
        );
    }

    #[test]
    fn consecutive_burst_info_cases() {
        let c = computer(0);
        let empty = c.consecutive_burst_info(&[], &[]);
        assert_eq!(empty, (0, 0));

        let no_valid = c.consecutive_burst_info(&[false; 18], &[0, 3, 6, 10, 13, 15]);
        assert_eq!(no_valid, (0, 0));

        let less_min = c.consecutive_burst_info(
            &[
                false, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false,
            ],
            &[0, 3, 7, 10, 12],
        );
        assert_eq!(less_min, (1, 3));

        let meet = c.consecutive_burst_info(
            &[
                false, true, false, false, true, false, false, true, false, true, false, false,
                false, true, false, false, false, true, false, false, false, true, false, false,
                false, true, false, false, false, false, false, true, false, false, true, true,
                true, true, false, false, false,
            ],
            &[0, 3, 7, 10, 12, 16],
        );
        assert_eq!(meet, (3, 10));

        let exceed = c.consecutive_burst_info(
            &[
                false, true, false, true, false, false, false, false, false, true, true, true,
                true, true, true, true, false, false, false,
            ],
            &[0, 3, 7, 10, 12, 16],
        );
        assert_eq!(exceed, (5, 16));

        let after_start = c.consecutive_burst_info(
            &[
                false, false, true, true, true, true, true, true, true, false, false, false, true,
                true, true, true, true, true, false, false, false,
            ],
            &[0, 2, 5, 9, 12, 14, 18],
        );
        assert_eq!(after_start, (0, 0));

        let end_before = c.consecutive_burst_info(&[true; 21], &[0, 2, 5, 9, 12, 14, 18]);
        assert_eq!(end_before, (7, 21));

        let through_end = c.consecutive_burst_info(
            &[
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, false, true, false,
            ],
            &[0, 2, 5, 9, 12, 14, 18],
        );
        assert_eq!(through_end, (7, 21));

        let gap = c.consecutive_burst_info(
            &[
                true, true, true, true, true, false, false, false, false, true, true, true, true,
                true, true, true, true, true,
            ],
            &[0, 2, 5, 9, 12, 14],
        );
        assert_eq!(gap, (2, 5));
    }

    #[test]
    fn get_positions_consecutive_losses_cases() {
        let c = computer(0);
        let empty = c.get_positions_consecutive_losses(&[], &[], 2);
        assert_eq!(empty, (Vec::<u16>::new(), Vec::<u16>::new()));

        let no_valid = c.get_positions_consecutive_losses(&[false; 18], &[0, 3, 6, 10, 13, 15], 2);
        assert_eq!(no_valid, (Vec::<u16>::new(), Vec::<u16>::new()));

        let one = c.get_positions_consecutive_losses(
            &[
                false, false, true, true, true, true, true, true, true, false, false, false, false,
                false, false, false, false, false,
            ],
            &[0, 2, 5, 9, 12, 14, 18],
            2,
        );
        assert_eq!(one, (vec![2], vec![7]));

        let two = c.get_positions_consecutive_losses(
            &[
                false, false, true, true, true, true, true, true, true, false, false, false, true,
                true, true, true, true, true, false, false, false,
            ],
            &[0, 2, 5, 9, 12, 14, 18],
            2,
        );
        assert_eq!(two, (vec![2, 12], vec![7, 6]));

        let three = c.get_positions_consecutive_losses(
            &[
                false, false, true, true, true, true, true, true, true, false, false, false, true,
                true, true, true, true, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, true, true, true, true, true,
                true, true, false, false,
            ],
            &[0, 2, 5, 9, 12, 14, 18, 21, 26, 32, 35, 39],
            2,
        );
        assert_eq!(three, (vec![2, 12, 32], vec![7, 6, 7]));

        let beginning = c.get_positions_consecutive_losses(
            &[
                true, true, true, true, true, false, false, false, false, false, false, false,
                true, true, true, true, true, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, true, true, true, true,
                true, true, true, false, false,
            ],
            &[0, 2, 5, 9, 12, 14, 18, 21, 26, 32, 35, 39],
            2,
        );
        assert_eq!(beginning, (vec![0, 12, 32], vec![5, 6, 7]));

        let end_pos = c.get_positions_consecutive_losses(
            &[
                false, false, true, true, true, true, true, true, true, false, false, false, true,
                true, true, true, true, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, true, true, true, true, true,
                true, true,
            ],
            &[0, 2, 5, 9, 12, 14, 18, 21, 26, 32, 35],
            2,
        );
        assert_eq!(end_pos, (vec![2, 12, 32], vec![7, 6, 7]));

        let single_consecutive = c.get_positions_consecutive_losses(
            &[
                true, true, false, false, false, true, true, true, true, false, false, false, true,
                true, true, true, true, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, true, true, true, true, true,
                true, true,
            ],
            &[0, 2, 5, 9, 12, 14, 18, 21, 26, 32, 35],
            1,
        );
        assert_eq!(single_consecutive, (vec![0, 5, 12, 32], vec![2, 4, 6, 7]));

        let three_consecutive = c.get_positions_consecutive_losses(
            &[
                false, false, true, true, true, true, true, true, true, false, false, false, true,
                true, true, true, true, true, true, true, true, false, false, false, false, false,
                false, false, false, false, false, false, true, true, true, true, true, true, true,
            ],
            &[0, 2, 5, 9, 12, 14, 18, 21, 26, 32, 35],
            3,
        );
        assert_eq!(three_consecutive, (vec![12], vec![9]));

        let more_than_losses: Vec<bool> = {
            let mut v = vec![false, false];
            v.extend(vec![true; 19]);
            v.extend(vec![false; 5]);
            v.extend(vec![true; 9]);
            v.extend(vec![false; 4]);
            v
        };
        assert_eq!(more_than_losses.len(), 39);
        let more_than = c.get_positions_consecutive_losses(
            &more_than_losses,
            &[0, 2, 5, 9, 12, 14, 18, 21, 26, 32, 35],
            2,
        );
        assert_eq!(more_than, (vec![2, 26], vec![19, 9]));
    }

    #[test]
    fn compute_multi_frame_loss_fraction_cases() {
        let c = computer(0);
        assert_eq!(c.compute_multi_frame_loss_fraction(&[], &[], 2), 0.0);
        assert_eq!(
            c.compute_multi_frame_loss_fraction(&[false; 18], &[0, 2, 5, 9, 12, 14], 2),
            0.0
        );
        assert_eq!(
            c.compute_multi_frame_loss_fraction(
                &[
                    false, false, true, true, true, true, true, true, true, false, false, false,
                    false, false, false, false, false, false,
                ],
                &[0, 2, 5, 9, 12, 14],
                2
            ),
            1.0
        );
        assert_eq!(
            c.compute_multi_frame_loss_fraction(
                &[
                    false, false, true, true, true, true, true, true, true, false, false, false,
                    true, true, true, true, true, true, false, false, false,
                ],
                &[0, 2, 5, 9, 12, 14, 18],
                2
            ),
            1.0
        );
        let frac = c.compute_multi_frame_loss_fraction(
            &[
                false, false, true, false, true, true, true, true, true, false, false, false, true,
                false, true, false, true, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, true, true, true, true, true,
                true, true, false, false,
            ],
            &[0, 2, 5, 9, 12, 14, 18, 21, 26, 32, 35, 39],
            2,
        );
        assert_near(frac, 1.0 / 3.0 * (6.0 / 7.0 + 0.5 + 1.0));
    }

    fn sample_loss_info() -> LossInfo {
        LossInfo {
            indices: vec![0, 2, 5, 8, 28, 31, 34, 37, 57, 58, 59, 60, 61, 62],
            packet_losses: vec![
                true, true, false, false, false, true, true, true, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, true, true, false, true, true, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, false, false,
                false, false, false,
            ],
            frame_losses: vec![
                true, false, true, false, true, true, true, false, false, false, true, false,
                false, false,
            ],
        }
    }

    #[test]
    fn init_and_get_indicators() {
        let info = sample_loss_info();
        let m = LossMetricComputer::new(info.clone(), 1, 3, 3, 2, 2).compute_loss_metrics();
        assert_near(m.burst_density, 11.0 / 14.0);
        assert_near(m.frame_burst_density, 5.0 / 7.0);
        assert_near(m.gap_density, 1.0 / 49.0);
        assert_near(m.frame_gap_density, 1.0 / 7.0);
        assert_near(m.mean_burst_length, (2.0 + 3.0 + 3.0 + 4.0) / 7.0);
        assert_near(m.mean_frame_burst_length, 6.0 / 4.0);
        assert_near(m.loss_fraction, 12.0 / 63.0);
        assert_near(m.frame_loss_fraction, 6.0 / 14.0);
        assert_near(m.multi_frame_loss_fraction, 2.0 / 3.0);
        assert_near(m.mean_guardspace_length, 51.0 / 7.0);
        assert_near(m.mean_frame_guardspace_length, 2.0);
        assert_near(m.multi_frame_insufficient_guardspace, 0.5);
        assert_near(m.redundancy_wire_byte, 1.0);
    }

    #[test]
    fn get_window_metrics() {
        let info = sample_loss_info();
        let c = LossMetricComputer::new(info.clone(), 1, 3, 3, 2, 2);
        let vals = c.get_window_metrics(&info);
        assert_eq!(vals, (63.0, 12.0, 3.0));
    }

    #[test]
    fn metrics_eq_with_same_args() {
        let info = sample_loss_info();
        let a = LossMetricComputer::new(info.clone(), 1, 3, 3, 2, 2).compute_loss_metrics();
        let b = LossMetricComputer::new(info.clone(), 1, 3, 3, 2, 2).compute_loss_metrics();
        assert_eq!(a, b);
        let c = LossMetricComputer::new(info.clone(), 1, 4, 3, 2, 2).compute_loss_metrics();
        assert_ne!(a, c);
    }
}
