//! Scalar GF(2^8) arithmetic (Jerasure default `w = 8`, prim poly `0435` / `0x11D`).

use crate::fec::error::{FecError, FecResult};
use crate::fec::gf::backend;
use crate::fec::gf::table;

#[inline]
pub(crate) fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

pub(crate) fn mul(a: u8, b: u8) -> u8 {
    backend::gf8_mul(a, b)
}

pub(crate) fn div(a: u8, b: u8) -> FecResult<u8> {
    if b == 0 {
        return Err(FecError::GfDivisionByZero);
    }
    if a == 0 {
        return Ok(0);
    }
    Ok(mul(a, inv(b)?))
}

pub(crate) fn inv(a: u8) -> FecResult<u8> {
    if a == 0 {
        return Err(FecError::GfDivisionByZero);
    }
    Ok(table::gf8_inv(a))
}

pub(crate) fn region_xor(src: &[u8], dest: &mut [u8]) {
    backend::region_xor(src, dest);
}

/// `dest = src * coeff` or `dest ^= src * coeff` (Jerasure `galois_w8_region_multiply`).
pub(crate) fn region_mul_add(
    coeff: u8,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    backend::region_mul_add_gf8(coeff, src, dest, xor_into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_inv_roundtrip() {
        for a in 1u8..=255 {
            let inv_a = inv(a).unwrap();
            assert_eq!(mul(a, inv_a), 1);
        }
    }

    #[test]
    fn div_matches_mul_inv() {
        for a in 0u8..=255 {
            for b in 1u8..=255 {
                assert_eq!(div(a, b).unwrap(), mul(a, inv(b).unwrap()));
            }
        }
    }

    #[test]
    fn known_cauchy_entry_w8() {
        let denom = add(0, 255);
        let entry = div(1, denom).unwrap();
        assert_ne!(entry, 0);
        assert_eq!(mul(entry, denom), 1);
    }
}
