//! Galois-field word size for streaming and header FEC.
//!
//! Tambur uses GF(2^32) for payload coding and GF(2^8) for the frame-size
//! header code. This crate exposes only those two as a closed enum so that
//! every Galois-field operation can dispatch exhaustively without a runtime
//! "unsupported word size" error path.

/// GF(2^w) word size in bits.
///
/// The enum is closed by design: gf-complete supports more sizes, but every
/// production Tambur configuration uses one of these two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WordSize {
    /// GF(2^8). Used for the frame-size header code.
    W8,
    /// GF(2^32). Used for the payload streaming code.
    W32,
}

impl WordSize {
    /// Word size in bits.
    pub const fn bits(self) -> u16 {
        match self {
            WordSize::W8 => 8,
            WordSize::W32 => 32,
        }
    }
}

impl From<WordSize> for u16 {
    fn from(w: WordSize) -> u16 {
        w.bits()
    }
}

impl From<WordSize> for u32 {
    fn from(w: WordSize) -> u32 {
        u32::from(w.bits())
    }
}
