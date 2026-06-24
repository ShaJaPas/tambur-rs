//! Portable (table-based) GF region kernels.

use crate::fec::error::{FecError, FecResult};
use crate::fec::gf::table::{Gf8ShuffleTables, Gf32SplitTables, gf8_mul, gf32_mul_split};

#[inline]
pub(super) fn region_xor_tail(src: &[u8], dest: &mut [u8], start: usize) {
    dest[start..]
        .iter_mut()
        .zip(&src[start..])
        .for_each(|(d, s)| *d ^= *s);
}

pub(super) fn region_xor(src: &[u8], dest: &mut [u8]) {
    debug_assert_eq!(src.len(), dest.len());
    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

pub(super) fn region_mul_add_gf8(
    coeff: u8,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    if src.len() != dest.len() {
        return Err(FecError::BufferLengthMismatch {
            expected: dest.len(),
            actual: src.len(),
        });
    }
    if coeff == 0 {
        if !xor_into {
            dest.fill(0);
        }
        return Ok(());
    }
    if coeff == 1 {
        if xor_into {
            region_xor(src, dest);
        } else {
            dest.copy_from_slice(src);
        }
        return Ok(());
    }
    region_mul_add_gf8_tables(Gf8ShuffleTables::get(coeff), src, dest, xor_into)
}

pub(super) fn region_mul_add_gf8_tables(
    tables: &Gf8ShuffleTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    if src.len() != dest.len() {
        return Err(FecError::BufferLengthMismatch {
            expected: dest.len(),
            actual: src.len(),
        });
    }
    for i in 0..src.len() {
        let b = src[i];
        let product = tables.lo[(b & 0x0f) as usize] ^ tables.hi[(b >> 4) as usize];
        if xor_into {
            dest[i] ^= product;
        } else {
            dest[i] = product;
        }
    }
    Ok(())
}

pub(super) fn region_mul_add_gf32(
    coeff: u32,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    if src.len() != dest.len() {
        return Err(FecError::BufferLengthMismatch {
            expected: dest.len(),
            actual: src.len(),
        });
    }
    if !src.len().is_multiple_of(4) {
        return Err(FecError::UnalignedRegion { len: src.len() });
    }
    if coeff == 0 {
        if !xor_into {
            dest.fill(0);
        }
        return Ok(());
    }
    if coeff == 1 {
        if xor_into {
            region_xor(src, dest);
        } else {
            dest.copy_from_slice(src);
        }
        return Ok(());
    }

    let tables = Gf32SplitTables::for_coeff(coeff);
    region_mul_add_gf32_tables(&tables, src, dest, xor_into)
}

pub(super) fn region_mul_add_gf32_tables(
    tables: &Gf32SplitTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    if src.len() != dest.len() {
        return Err(FecError::BufferLengthMismatch {
            expected: dest.len(),
            actual: src.len(),
        });
    }
    if !src.len().is_multiple_of(4) {
        return Err(FecError::UnalignedRegion { len: src.len() });
    }

    let words = src.len() / 4;
    for i in 0..words {
        let off = i * 4;
        let s = u32::from_le_bytes(src[off..off + 4].try_into().expect("word"));
        let product = tables.mul_word(s);
        if xor_into {
            let d = u32::from_le_bytes(dest[off..off + 4].try_into().expect("word"));
            dest[off..off + 4].copy_from_slice(&(product ^ d).to_le_bytes());
        } else {
            dest[off..off + 4].copy_from_slice(&product.to_le_bytes());
        }
    }
    Ok(())
}

pub(super) fn gf32_mul_dispatch(a: u32, b: u32) -> u32 {
    gf32_mul_split(a, b)
}

pub(super) fn gf8_mul_dispatch(a: u8, b: u8) -> u8 {
    gf8_mul(a, b)
}
