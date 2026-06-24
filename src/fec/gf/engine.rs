//! Monomorphic GF dispatch for Jerasure hot loops (optimization §1.6).

use crate::fec::error::{FecError, FecResult};
use crate::fec::gf::{gf8, gf32};

/// GF kernel set specialized at compile time for `w = 8` or `w = 32`.
pub(crate) trait GfEngine {
    fn add(a: u32, b: u32) -> u32;
    fn mul(a: u32, b: u32) -> u32;
    fn div(a: u32, b: u32) -> FecResult<u32>;
    fn region_xor(src: &[u8], dest: &mut [u8]);
    fn region_mul_add(coeff: u32, src: &[u8], dest: &mut [u8], xor_into: bool) -> FecResult<()>;
    fn validate_region_size(size: usize) -> FecResult<()>;
    fn max_field_elements() -> usize;
}

#[derive(Copy, Clone)]
pub(crate) struct EngineW8;

#[derive(Copy, Clone)]
pub(crate) struct EngineW32;

impl GfEngine for EngineW8 {
    #[inline]
    fn add(a: u32, b: u32) -> u32 {
        u32::from(gf8::add(a as u8, b as u8))
    }

    #[inline]
    fn mul(a: u32, b: u32) -> u32 {
        u32::from(gf8::mul(a as u8, b as u8))
    }

    fn div(a: u32, b: u32) -> FecResult<u32> {
        Ok(u32::from(gf8::div(a as u8, b as u8)?))
    }

    fn region_xor(src: &[u8], dest: &mut [u8]) {
        gf8::region_xor(src, dest);
    }

    fn region_mul_add(coeff: u32, src: &[u8], dest: &mut [u8], xor_into: bool) -> FecResult<()> {
        gf8::region_mul_add(coeff as u8, src, dest, xor_into)
    }

    fn validate_region_size(_size: usize) -> FecResult<()> {
        Ok(())
    }

    fn max_field_elements() -> usize {
        256
    }
}

impl GfEngine for EngineW32 {
    #[inline]
    fn add(a: u32, b: u32) -> u32 {
        gf32::add(a, b)
    }

    #[inline]
    fn mul(a: u32, b: u32) -> u32 {
        gf32::mul(a, b)
    }

    fn div(a: u32, b: u32) -> FecResult<u32> {
        gf32::div(a, b)
    }

    fn region_xor(src: &[u8], dest: &mut [u8]) {
        gf32::region_xor(src, dest);
    }

    fn region_mul_add(coeff: u32, src: &[u8], dest: &mut [u8], xor_into: bool) -> FecResult<()> {
        gf32::region_mul_add(coeff, src, dest, xor_into)
    }

    fn validate_region_size(size: usize) -> FecResult<()> {
        if !size.is_multiple_of(4) {
            return Err(FecError::UnalignedRegion { len: size });
        }
        Ok(())
    }

    fn max_field_elements() -> usize {
        1 << 16
    }
}
