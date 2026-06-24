//! Max-flow parity selection for decode.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// Parent link in an augmenting path: internal arc or terminal connection.
enum Parent {
    Internal(usize),
    Terminal,
}

const NO_EDGE: usize = usize::MAX;

/// Residual-capacity graph matching Kolmogorov `Graph<int,int,int>`.
struct FlowGraph {
    n: usize,
    sink: usize,
    /// Kolmogorov terminal excess: SOURCE→i minus i→SINK residual.
    tr_cap: Vec<i32>,
    edge_head: Vec<usize>,
    edge_to: Vec<usize>,
    edge_cap: Vec<i32>,
    edge_next: Vec<usize>,
}

impl FlowGraph {
    fn new(n: usize) -> Self {
        Self {
            n,
            sink: n,
            tr_cap: vec![0; n],
            edge_head: vec![NO_EDGE; n + 1],
            edge_to: Vec::new(),
            edge_cap: Vec::new(),
            edge_next: Vec::new(),
        }
    }

    fn push_edge(&mut self, from: usize, to: usize, cap: i32) {
        let idx = self.edge_to.len();
        self.edge_to.push(to);
        self.edge_cap.push(cap);
        self.edge_next.push(self.edge_head[from]);
        self.edge_head[from] = idx;
    }

    fn add_edge(&mut self, from: usize, to: usize, cap: i32) {
        if cap <= 0 {
            return;
        }
        self.push_edge(from, to, cap);
        self.push_edge(to, from, 0);
    }

    fn cap_to(&self, from: usize, to: usize) -> i32 {
        let mut e = self.edge_head[from];
        while e != NO_EDGE {
            if self.edge_to[e] == to {
                return self.edge_cap[e];
            }
            e = self.edge_next[e];
        }
        0
    }

    fn add_tweights(&mut self, v: usize, source_cap: i32, sink_cap: i32) {
        let delta = self.tr_cap[v];
        let (mut cs, mut ck) = (source_cap, sink_cap);
        if delta > 0 {
            cs += delta;
        } else {
            ck -= delta;
        }
        self.tr_cap[v] = cs - ck;
    }

    fn maxflow(&mut self) -> i32 {
        let mut total = 0i32;
        loop {
            let mut parent: Vec<Option<Parent>> = (0..self.n + 1).map(|_| None).collect();
            let mut queue = VecDeque::new();
            for (v, cap) in self.tr_cap.iter().take(self.n).enumerate() {
                if *cap > 0 {
                    parent[v] = Some(Parent::Terminal);
                    queue.push_back(v);
                }
            }
            while let Some(u) = queue.pop_front() {
                if self.tr_cap[u] < 0 && parent[self.sink].is_none() {
                    parent[self.sink] = Some(Parent::Internal(u));
                }
                let mut e = self.edge_head[u];
                while e != NO_EDGE {
                    let v = self.edge_to[e];
                    let cap = self.edge_cap[e];
                    if cap > 0 && parent[v].is_none() {
                        parent[v] = Some(Parent::Internal(u));
                        queue.push_back(v);
                    }
                    e = self.edge_next[e];
                }
            }
            if parent[self.sink].is_none() {
                break;
            }

            let mut path_cap = i32::MAX;
            let mut v = self.sink;
            loop {
                match parent[v] {
                    Some(Parent::Internal(u)) if v == self.sink => {
                        path_cap = path_cap.min(-self.tr_cap[u]);
                        v = u;
                    }
                    Some(Parent::Internal(u)) => {
                        path_cap = path_cap.min(self.cap_to(u, v));
                        if self.tr_cap[u] > 0 {
                            path_cap = path_cap.min(self.tr_cap[u]);
                        }
                        v = u;
                    }
                    Some(Parent::Terminal) => break,
                    None => {
                        path_cap = 0;
                        break;
                    }
                }
            }
            if path_cap <= 0 {
                break;
            }

            v = self.sink;
            loop {
                match parent[v] {
                    Some(Parent::Internal(u)) if v == self.sink => {
                        self.tr_cap[u] += path_cap;
                        v = u;
                    }
                    Some(Parent::Internal(u)) => {
                        self.augment(u, v, path_cap);
                        self.augment(v, u, -path_cap);
                        if self.tr_cap[u] > 0 {
                            self.tr_cap[u] -= path_cap;
                        }
                        v = u;
                    }
                    Some(Parent::Terminal) | None => break,
                }
            }
            total += path_cap;
        }
        total
    }

    fn augment(&mut self, u: usize, v: usize, delta: i32) {
        let mut e = self.edge_head[u];
        while e != NO_EDGE {
            if self.edge_to[e] == v {
                self.edge_cap[e] -= delta;
                break;
            }
            e = self.edge_next[e];
        }
        let mut rev = self.edge_head[v];
        while rev != NO_EDGE {
            if self.edge_to[rev] == u {
                self.edge_cap[rev] += delta;
                return;
            }
            rev = self.edge_next[rev];
        }
        self.push_edge(v, u, delta);
    }

    /// Kolmogorov `Graph::get_trcap(i)` — terminal excess at node `i`.
    fn get_trcap(&self, v: usize) -> i32 {
        self.tr_cap[v]
    }
}

fn filter_unusable_into(counts: &[u16], available: &[bool], out: &mut Vec<u16>) {
    out.resize(counts.len(), 0);
    for (dst, (&count, &avail)) in out.iter_mut().zip(counts.iter().zip(available.iter())) {
        *dst = if avail { count } else { 0 };
    }
}

/// Reusable buffers for [`get_used_parity_counts`] (optimization §5.4).
#[derive(Debug, Clone, Default)]
pub(super) struct FlowDecodeScratch {
    pub(super) recovered_us: Vec<u16>,
    pub(super) recovered_vs: Vec<u16>,
    pub(super) usable: Vec<bool>,
    pub(super) use_parities: Vec<u16>,
}

fn construct_graph(
    num_recovered_us: &[u16],
    num_recovered_vs: &[u16],
    num_usable_parities: &[u16],
) -> FlowGraph {
    let num_frames = num_recovered_us.len();
    let delay = num_frames / 2;
    let num_vertices = 3 * num_frames - delay;
    let num_use_parities = num_frames - delay;

    let mut graph = FlowGraph::new(num_vertices);

    for parity in 0..num_use_parities {
        let capacity = num_usable_parities[delay + parity] as i32;
        graph.add_tweights(parity, capacity, 0);
        for v in parity..=parity + delay {
            graph.add_edge(parity, num_use_parities + v, capacity);
        }
        graph.add_edge(parity, num_use_parities + num_frames + parity, capacity);
        graph.add_edge(
            parity,
            num_use_parities + num_frames + parity + delay,
            capacity,
        );
    }

    let final_v = num_use_parities + num_frames - 1;
    for pos in num_use_parities..=final_v {
        graph.add_tweights(pos, 0, num_recovered_vs[pos - num_use_parities] as i32);
    }
    let final_u = final_v + num_frames;
    for pos in final_v + 1..=final_u {
        graph.add_tweights(pos, 0, num_recovered_us[pos - final_v - 1] as i32);
    }

    graph
}

fn construct_graph_from_losses(
    num_missing_us: &[u16],
    num_missing_vs: &[u16],
    num_received_parities: &[u16],
    recovered_us: &[bool],
    recovered_vs: &[bool],
    unusable_parities: &[bool],
    scratch: &mut FlowDecodeScratch,
) -> FlowGraph {
    filter_unusable_into(num_missing_us, recovered_us, &mut scratch.recovered_us);
    filter_unusable_into(num_missing_vs, recovered_vs, &mut scratch.recovered_vs);
    scratch.usable.resize(unusable_parities.len(), false);
    for (dst, &unusable) in scratch.usable.iter_mut().zip(unusable_parities.iter()) {
        *dst = !unusable;
    }
    filter_unusable_into(
        num_received_parities,
        &scratch.usable,
        &mut scratch.use_parities,
    );
    construct_graph(
        &scratch.recovered_us,
        &scratch.recovered_vs,
        &scratch.use_parities,
    )
}

fn get_used_parities(
    graph: &mut FlowGraph,
    num_usable_parities: &[u16],
    num_frames: u16,
) -> Vec<u16> {
    let delay = num_frames as usize / 2;
    graph.maxflow();
    let mut parities_used = vec![0u16; num_frames as usize];
    for frame_num in 0..=delay {
        let idx = frame_num + delay;
        parities_used[idx] =
            num_usable_parities[idx].saturating_sub(graph.get_trcap(frame_num) as u16);
    }
    parities_used
}

pub(super) fn get_used_parity_counts(
    num_missing_us: &[u16],
    num_missing_vs: &[u16],
    num_received_parities: &[u16],
    recovered_us: &[bool],
    recovered_vs: &[bool],
    unusable_parities: &[bool],
    scratch: &mut FlowDecodeScratch,
) -> Vec<u16> {
    debug_assert_eq!(num_missing_us.len(), num_missing_vs.len());
    let mut graph = construct_graph_from_losses(
        num_missing_us,
        num_missing_vs,
        num_received_parities,
        recovered_us,
        recovered_vs,
        unusable_parities,
        scratch,
    );
    filter_unusable_into(
        num_received_parities,
        &scratch.usable,
        &mut scratch.use_parities,
    );
    get_used_parities(
        &mut graph,
        &scratch.use_parities,
        num_missing_us.len() as u16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_timeslot6_flow_assigns_parities() {
        let missing_u = vec![0u16; 7];
        let missing_v = vec![2, 0, 0, 2, 2, 0, 0];
        let received = vec![2, 2, 2, 0, 0, 2, 2];
        let u_rec = vec![true; 7];
        let v_rec = vec![true; 7];
        let unusable = vec![true, true, true, true, true, false, false];
        let mut scratch = FlowDecodeScratch::default();
        let mut graph = construct_graph_from_losses(
            &missing_u,
            &missing_v,
            &received,
            &u_rec,
            &v_rec,
            &unusable,
            &mut scratch,
        );
        let flow = graph.maxflow();
        let used = get_used_parity_counts(
            &missing_u,
            &missing_v,
            &received,
            &u_rec,
            &v_rec,
            &unusable,
            &mut scratch,
        );
        eprintln!("burst flow={flow} used={used:?}");
        assert!(flow > 0, "maxflow must push parity for missing V stripes");
        assert!(
            used[5] > 0 || used[6] > 0,
            "frames 5/6 should contribute parities, got {used:?}"
        );
    }

    /// Byte-for-byte parity assignment against Kolmogorov maxflow
    /// on the burst-timeslot-6 scenario.
    #[test]
    fn parity_counts_match_cpp_bk_reference() {
        let missing_u = vec![0u16; 7];
        let missing_v = vec![2, 0, 0, 2, 2, 0, 0];
        let received = vec![2, 2, 2, 0, 0, 2, 2];
        let u_rec = vec![true; 7];
        let v_rec = vec![true; 7];
        let unusable = vec![true, true, true, true, true, false, false];
        let mut scratch = FlowDecodeScratch::default();
        let mut graph = construct_graph_from_losses(
            &missing_u,
            &missing_v,
            &received,
            &u_rec,
            &v_rec,
            &unusable,
            &mut scratch,
        );
        assert_eq!(graph.maxflow(), 4);
        let used = get_used_parity_counts(
            &missing_u,
            &missing_v,
            &received,
            &u_rec,
            &v_rec,
            &unusable,
            &mut scratch,
        );
        assert_eq!(used, vec![0, 0, 0, 0, 0, 2, 2]);
        for frame_num in 0..=3usize {
            let trcap = graph.get_trcap(frame_num);
            let idx = frame_num + 3;
            assert_eq!(
                used[idx],
                received[idx].saturating_sub(trcap as u16),
                "parity node {frame_num} trcap={trcap}"
            );
        }
    }
}
