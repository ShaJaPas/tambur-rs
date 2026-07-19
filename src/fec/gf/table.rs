//! Precomputed GF(2^8) tables and GF(2^32) split-multiply helpers.

/// Irreducible polynomial for GF(2^8) (Jerasure default).
pub(super) const GF8_POLY: u8 = 0x1D;
/// Irreducible polynomial for GF(2^32) (Jerasure / GF-Complete default).
pub(super) const GF32_POLY: u32 = 0x0040_0007;

const fn gf8_mul_bit(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let mut aa = a;
    let mut bb = b;
    let mut prod = 0u8;
    while bb != 0 {
        if bb & 1 != 0 {
            prod ^= aa;
        }
        bb >>= 1;
        let carry = aa & 0x80;
        aa <<= 1;
        if carry != 0 {
            aa ^= GF8_POLY;
        }
    }
    prod
}

pub(super) const fn gf32_mul_bit(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        return 0;
    }
    let mut aa = a;
    let mut bb = b;
    let mut prod = 0u32;
    while bb != 0 {
        if bb & 1 != 0 {
            prod ^= aa;
        }
        bb >>= 1;
        let carry = aa & 0x8000_0000;
        aa <<= 1;
        if carry != 0 {
            aa ^= GF32_POLY;
        }
    }
    prod
}

const fn build_gf8_mul_table() -> [[u8; 256]; 256] {
    let mut table = [[0u8; 256]; 256];
    let mut a = 0usize;
    while a < 256 {
        let mut b = 0usize;
        while b < 256 {
            table[a][b] = gf8_mul_bit(a as u8, b as u8);
            b += 1;
        }
        a += 1;
    }
    table
}

const fn build_gf8_inv_table(mul: &[[u8; 256]; 256]) -> [u8; 256] {
    let mut inv = [0u8; 256];
    inv[0] = 0;
    let mut a = 1usize;
    while a < 256 {
        let mut b = 1usize;
        while b < 256 {
            if mul[a][b] == 1 {
                inv[a] = b as u8;
                break;
            }
            b += 1;
        }
        a += 1;
    }
    inv
}

static GF8_MUL_TABLE: [[u8; 256]; 256] = build_gf8_mul_table();
const GF8_INV_TABLE: [u8; 256] = build_gf8_inv_table(&GF8_MUL_TABLE);

#[inline]
pub(super) fn gf8_mul(a: u8, b: u8) -> u8 {
    GF8_MUL_TABLE[a as usize][b as usize]
}

#[inline]
pub(super) fn gf8_inv(a: u8) -> u8 {
    GF8_INV_TABLE[a as usize]
}

#[inline]
#[cfg(test)]
pub(super) fn gf32_mul(a: u32, b: u32) -> u32 {
    gf32_mul_bit(a, b)
}

/// Split-table scalar multiply (lazy tables cached by `b`).
#[inline]
pub(super) fn gf32_mul_split(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        return 0;
    }
    Gf32SplitTables::for_coeff(b).mul_word(a)
}

/// Low/high nibble multiply tables for SSSE3 `pshufb` region multiply.
#[derive(Clone, Copy)]
pub(super) struct Gf8ShuffleTables {
    pub lo: [u8; 16],
    pub hi: [u8; 16],
}

const fn gf8_shuffle_table_for_coeff(coeff: u8) -> Gf8ShuffleTables {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    let mut i = 0usize;
    while i < 16 {
        lo[i] = gf8_mul_bit(i as u8, coeff);
        hi[i] = gf8_mul_bit((i as u8) << 4, coeff);
        i += 1;
    }
    Gf8ShuffleTables { lo, hi }
}

const fn build_gf8_shuffle_tables() -> [Gf8ShuffleTables; 256] {
    let mut tables = [gf8_shuffle_table_for_coeff(0); 256];
    let mut coeff = 0usize;
    while coeff < 256 {
        tables[coeff] = gf8_shuffle_table_for_coeff(coeff as u8);
        coeff += 1;
    }
    tables
}

static GF8_SHUFFLE_TABLES: [Gf8ShuffleTables; 256] = build_gf8_shuffle_tables();

impl Gf8ShuffleTables {
    #[inline]
    pub(super) fn get(coeff: u8) -> &'static Self {
        &GF8_SHUFFLE_TABLES[coeff as usize]
    }
}

/// Eight nibble-position tables for GF(2^32) split region multiply.
#[derive(Clone, Copy)]
pub(super) struct Gf32SplitTables {
    pub tables: [[u32; 16]; 8],
}

impl Gf32SplitTables {
    pub(super) fn new(coeff: u32) -> Self {
        let mut tables = [[0u32; 16]; 8];
        for (pos, row) in tables.iter_mut().enumerate() {
            for (nibble, cell) in row.iter_mut().enumerate() {
                *cell = gf32_mul_bit((nibble as u32) << (pos * 4), coeff);
            }
        }
        Self { tables }
    }

    #[inline]
    pub(super) fn for_coeff(coeff: u32) -> Self {
        #[cfg(feature = "std")]
        {
            use std::collections::HashMap;
            use std::sync::{Mutex, OnceLock};

            static CACHE: OnceLock<Mutex<HashMap<u32, Gf32SplitTables>>> = OnceLock::new();
            let mut cache = CACHE
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .expect("gf32 cache lock");
            if let Some(&tables) = cache.get(&coeff) {
                return tables;
            }
            let tables = Gf32SplitTables::new(coeff);
            cache.insert(coeff, tables);
            tables
        }
        #[cfg(not(feature = "std"))]
        {
            Self::new(coeff)
        }
    }

    #[inline]
    pub(super) fn mul_word(&self, word: u32) -> u32 {
        let n0 = (word & 0xf) as usize;
        let n1 = ((word >> 4) & 0xf) as usize;
        let n2 = ((word >> 8) & 0xf) as usize;
        let n3 = ((word >> 12) & 0xf) as usize;
        let n4 = ((word >> 16) & 0xf) as usize;
        let n5 = ((word >> 20) & 0xf) as usize;
        let n6 = ((word >> 24) & 0xf) as usize;
        let n7 = ((word >> 28) & 0xf) as usize;
        self.tables[0][n0]
            ^ self.tables[1][n1]
            ^ self.tables[2][n2]
            ^ self.tables[3][n3]
            ^ self.tables[4][n4]
            ^ self.tables[5][n5]
            ^ self.tables[6][n6]
            ^ self.tables[7][n7]
    }
}

/// Pre-populate the thread-local GF(2^32) coeff cache (call once per `BlockCode`).
#[cfg(feature = "std")]
pub(crate) fn warm_gf32_coeff_cache(coeffs: &[i32]) {
    for &c in coeffs {
        let coeff = c as u32;
        if coeff > 1 {
            let _ = Gf32SplitTables::for_coeff(coeff);
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    crate::fec::gf::backend::warm_gf32_shuffle_cache(coeffs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf8_table_matches_bit_mul() {
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                assert_eq!(gf8_mul(a, b), gf8_mul_bit(a, b));
            }
        }
    }

    #[test]
    fn gf8_inv_table_roundtrip() {
        for a in 1u8..=255 {
            assert_eq!(gf8_mul(a, gf8_inv(a)), 1);
        }
    }

    #[test]
    fn gf32_split_matches_scalar() {
        let coeff = 0x1234_5678;
        let tables = Gf32SplitTables::new(coeff);
        for word in [0, 1, 0xdead_beef, 0x8000_0001] {
            assert_eq!(tables.mul_word(word), gf32_mul(word, coeff));
            assert_eq!(gf32_mul_split(word, coeff), gf32_mul(word, coeff));
        }
    }
}
