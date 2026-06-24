//! Scalar GF(2^32) arithmetic compatible with Jerasure 2.0 / GF-Complete defaults.

use crate::fec::error::{FecError, FecResult};
use crate::fec::gf::backend;
use crate::fec::gf::table;

pub(crate) const PRIM_POLY: u32 = table::GF32_POLY;

#[inline]
pub(crate) fn add(a: u32, b: u32) -> u32 {
    a ^ b
}

/// Multiply two field elements in GF(2^32).
pub(crate) fn mul(a: u32, b: u32) -> u32 {
    backend::gf32_mul(a, b)
}

/// Divide `a` by `b` in GF(2^32).
pub(crate) fn div(a: u32, b: u32) -> FecResult<u32> {
    if b == 0 {
        return Err(FecError::GfDivisionByZero);
    }
    if a == 0 {
        return Ok(0);
    }
    Ok(mul(a, inv(b)?))
}

/// Multiplicative inverse in GF(2^32).
pub(crate) fn inv(a: u32) -> FecResult<u32> {
    if a == 0 {
        return Err(FecError::GfDivisionByZero);
    }
    let modulus = (1u64 << 32) | u64::from(PRIM_POLY);
    let mut r0 = modulus;
    let mut r1 = u64::from(a);
    let mut t0 = 0u64;
    let mut t1 = 1u64;

    while r1 != 0 {
        if poly_degree(r0) < poly_degree(r1) {
            core::mem::swap(&mut r0, &mut r1);
            core::mem::swap(&mut t0, &mut t1);
            continue;
        }
        let shift = poly_degree(r0) - poly_degree(r1);
        r0 ^= r1 << shift;
        t0 ^= t1 << shift;
    }

    if r0 != 1 {
        return Err(FecError::GfNotInvertible);
    }
    Ok(truncate_poly(t0))
}

#[inline]
fn poly_degree(v: u64) -> u32 {
    if v == 0 { 0 } else { 64 - v.leading_zeros() }
}

#[inline]
fn truncate_poly(v: u64) -> u32 {
    (v & 0xffff_ffff) as u32
}

/// XOR `src` into `dest` (Jerasure `galois_region_xor`).
pub(crate) fn region_xor(src: &[u8], dest: &mut [u8]) {
    backend::region_xor(src, dest);
}

/// `dest = src * coeff` when `xor_into` is false, else `dest ^= src * coeff`.
pub(crate) fn region_mul_add(
    coeff: u32,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    backend::region_mul_add_gf32(coeff, src, dest, xor_into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_is_xor() {
        assert_eq!(add(0x1234_5678, 0x8765_4321), 0x9551_1559);
    }

    #[test]
    fn mul_identity_and_zero() {
        assert_eq!(mul(0, 123), 0);
        assert_eq!(mul(123, 0), 0);
        assert_eq!(mul(1, 0xdead_beef), 0xdead_beef);
        assert_eq!(mul(0xdead_beef, 1), 0xdead_beef);
    }

    #[test]
    fn inv_roundtrip() {
        for a in [1u32, 2, 3, 0x400007, 0xdead_beef, 0x8000_0001] {
            let inv_a = inv(a).unwrap();
            assert_eq!(mul(a, inv_a), 1, "failed for {a:#x}");
        }
    }

    #[test]
    fn div_consistency() {
        let a = 0x1234_5678;
        let b = 0x9abc_def0;
        assert_eq!(div(mul(a, b), b).unwrap(), a);
    }

    #[test]
    fn region_mul_add_matches_scalar() {
        let coeff = 0x1234_5678;
        let src: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut dest = [0u8; 8];
        region_mul_add(coeff, &src, &mut dest, false).unwrap();
        let s0 = u32::from_le_bytes(src[0..4].try_into().unwrap());
        let s1 = u32::from_le_bytes(src[4..8].try_into().unwrap());
        assert_eq!(
            u32::from_le_bytes(dest[0..4].try_into().unwrap()),
            mul(s0, coeff)
        );
        assert_eq!(
            u32::from_le_bytes(dest[4..8].try_into().unwrap()),
            mul(s1, coeff)
        );
    }
}
