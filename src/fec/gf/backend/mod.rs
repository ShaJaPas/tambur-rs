//! GF region kernel dispatch (scalar on all targets; SIMD on x86/aarch64 via autodetect).

pub(super) mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_gf32_split4;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod autodetect;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
pub(crate) fn warm_gf32_shuffle_cache(coeffs: &[i32]) {
    x86_gf32_split4::Gf32Split4ShuffleTables::warm_cache_for_coefficients(coeffs);
}

pub(super) fn region_xor(src: &[u8], dest: &mut [u8]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        autodetect::region_xor(src, dest);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::region_xor(src, dest);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::region_xor(src, dest);
    }
}

pub(super) fn region_mul_add_gf8(
    coeff: u8,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> crate::fec::error::FecResult<()> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        autodetect::region_mul_add_gf8(coeff, src, dest, xor_into)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::region_mul_add_gf8(coeff, src, dest, xor_into)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::region_mul_add_gf8(coeff, src, dest, xor_into)
    }
}

pub(super) fn region_mul_add_gf32(
    coeff: u32,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> crate::fec::error::FecResult<()> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        autodetect::region_mul_add_gf32(coeff, src, dest, xor_into)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::region_mul_add_gf32(coeff, src, dest, xor_into)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::region_mul_add_gf32(coeff, src, dest, xor_into)
    }
}

pub(super) fn gf32_mul(a: u32, b: u32) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        autodetect::gf32_mul(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::gf32_mul(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::gf32_mul_dispatch(a, b)
    }
}

pub(super) fn gf8_mul(a: u8, b: u8) -> u8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        autodetect::gf8_mul(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::gf8_mul(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar::gf8_mul_dispatch(a, b)
    }
}
