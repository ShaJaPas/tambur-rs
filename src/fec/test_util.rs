//! Test helpers.

use alloc::vec::Vec;

/// Glibc-compatible `rand()` after `srand(seed)`.
pub(super) struct GlibcRand {
    next: i64,
}

impl GlibcRand {
    pub(super) fn new(seed: u32) -> Self {
        Self { next: seed as i64 }
    }

    pub(super) fn rand(&mut self) -> i32 {
        self.next = self.next.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.next / 65536) % 32768) as i32
    }

    pub(super) fn rand_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.rand() as usize) % bound
    }
}

/// Mark `num_missing` random stripes/packets as lost.
pub(super) fn get_missing(num_pkts: u8, num_missing: u8, rng: &mut GlibcRand) -> Vec<bool> {
    let mut is_missing = vec![false; num_pkts as usize];
    debug_assert!(num_missing <= num_pkts);
    let mut num_dropped = 0u8;
    while num_dropped < num_missing {
        let index = rng.rand_usize(num_pkts as usize);
        if !is_missing[index] {
            is_missing[index] = true;
            num_dropped += 1;
        }
    }
    is_missing
}
