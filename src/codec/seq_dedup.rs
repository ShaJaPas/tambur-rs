//! Bounded duplicate detection for monotonic datagram sequence numbers.

use alloc::vec::Vec;

/// Track recently seen [`seq_num`] values in a fixed-size bitmap window.
#[derive(Debug, Clone, Default)]
pub(crate) struct SeqDedup {
    base: u32,
    words: Vec<u64>,
}

/// How many distinct sequence numbers we retain (covers several FEC windows).
const TRACK_BITS: u32 = 8192;

impl SeqDedup {
    pub(crate) fn insert(&mut self, seq: u32) -> bool {
        if seq + TRACK_BITS < self.base {
            return false;
        }
        if seq >= self.base + TRACK_BITS {
            self.compact_to(seq.saturating_sub(TRACK_BITS / 2));
        }
        if seq < self.base {
            return false;
        }
        let offset = seq - self.base;
        let word = (offset / 64) as usize;
        let bit = offset % 64;
        while self.words.len() <= word {
            self.words.push(0);
        }
        let mask = 1u64 << bit;
        if self.words[word] & mask != 0 {
            return false;
        }
        self.words[word] |= mask;
        true
    }

    pub(crate) fn remove(&mut self, seq: u32) {
        if seq < self.base {
            return;
        }
        let offset = seq - self.base;
        if offset >= TRACK_BITS {
            return;
        }
        let word = (offset / 64) as usize;
        let bit = offset % 64;
        if let Some(slot) = self.words.get_mut(word) {
            *slot &= !(1u64 << bit);
        }
    }

    fn compact_to(&mut self, new_base: u32) {
        if new_base <= self.base {
            return;
        }
        let shift = new_base - self.base;
        if shift >= TRACK_BITS {
            self.words.clear();
        } else {
            let word_shift = (shift / 64) as usize;
            let bit_shift = shift % 64;
            let mut new_words = Vec::new();
            for (i, &word) in self.words.iter().enumerate().skip(word_shift) {
                let dst = i - word_shift;
                while new_words.len() <= dst {
                    new_words.push(0);
                }
                new_words[dst] |= word >> bit_shift;
                if bit_shift > 0 && i + 1 < self.words.len() {
                    while new_words.len() <= dst + 1 {
                        new_words.push(0);
                    }
                    new_words[dst + 1] |= word << (64 - bit_shift);
                }
            }
            self.words = new_words;
        }
        self.base = new_base;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicates() {
        let mut d = SeqDedup::default();
        assert!(d.insert(10));
        assert!(!d.insert(10));
        assert!(d.insert(11));
    }

    #[test]
    fn rejects_seq_before_window() {
        let mut d = SeqDedup::default();
        for seq in 0..100 {
            assert!(d.insert(seq));
        }
        assert!(!d.insert(0));
    }

    #[test]
    fn accepts_after_large_jump() {
        let mut d = SeqDedup::default();
        assert!(d.insert(0));
        assert!(d.insert(10_000));
        assert!(!d.insert(10_000));
    }
}
