//! AArch64 NEON GF region kernels.

use crate::fec::error::FecResult;
use crate::fec::gf::table::Gf8ShuffleTables;

pub(super) fn region_xor(src: &[u8], dest: &mut [u8]) {
    debug_assert_eq!(src.len(), dest.len());
    // SAFETY: NEON XOR on valid slice pointers; tail handled scalar.
    unsafe {
        use core::arch::aarch64::*;
        let mut off = 0usize;
        for (s_chunk, d_chunk) in src.chunks_exact(16).zip(dest.chunks_exact_mut(16)) {
            let s = vld1q_u8(s_chunk.as_ptr());
            let d = vld1q_u8(d_chunk.as_ptr());
            vst1q_u8(d_chunk.as_mut_ptr(), veorq_u8(s, d));
            off += 16;
        }
        super::scalar::region_xor_tail(src, dest, off);
    }
}

pub(super) fn region_mul_add_gf8(
    coeff: u8,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    if coeff == 0 || coeff == 1 {
        return super::scalar::region_mul_add_gf8(coeff, src, dest, xor_into);
    }

    let tables = Gf8ShuffleTables::get(coeff);
    // SAFETY: NEON table multiply on valid pointers; tail handled scalar.
    unsafe {
        for (s_chunk, d_chunk) in src.chunks_exact(16).zip(dest.chunks_exact_mut(16)) {
            gf8_shuffle_lane_neon(s_chunk.as_ptr(), d_chunk.as_mut_ptr(), tables, xor_into);
        }
        let off = src.len() - src.len() % 16;
        if off < src.len() {
            super::scalar::region_mul_add_gf8_tables(
                tables,
                &src[off..],
                &mut dest[off..],
                xor_into,
            )?;
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
    super::scalar::region_mul_add_gf32(coeff, src, dest, xor_into)
}

pub(super) fn gf32_mul(a: u32, b: u32) -> u32 {
    super::scalar::gf32_mul_dispatch(a, b)
}

pub(super) fn gf8_mul(a: u8, b: u8) -> u8 {
    super::scalar::gf8_mul_dispatch(a, b)
}

unsafe fn gf8_shuffle_lane_neon(
    src: *const u8,
    dest: *mut u8,
    tables: &Gf8ShuffleTables,
    xor_into: bool,
) {
    // SAFETY: NEON `vtbl` region multiply on 16-byte lane.
    unsafe {
        use core::arch::aarch64::*;
        let lo_m = vld1q_u8(tables.lo.as_ptr());
        let hi_m = vld1q_u8(tables.hi.as_ptr());
        let mask = vdupq_n_u8(0x0f);
        let s = vld1q_u8(src);
        let slo = vandq_u8(s, mask);
        let shi = vandq_u8(vshrq_n_u8(s, 4), mask);
        let mut prod = veorq_u8(vqtbl1q_u8(lo_m, slo), vqtbl1q_u8(hi_m, shi));
        if xor_into {
            let d = vld1q_u8(dest);
            prod = veorq_u8(prod, d);
        }
        vst1q_u8(dest, prod);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::gf::table::gf8_mul as gf8_mul_table;

    #[test]
    fn neon_region_mul_matches_table() {
        let coeff = 43u8;
        let src: [u8; 32] = core::array::from_fn(|i| (i * 7 + 3) as u8);
        let mut dest = [0u8; 32];
        let mut expected = [0u8; 32];
        for i in 0..32 {
            expected[i] = gf8_mul_table(src[i], coeff);
        }
        region_mul_add_gf8(coeff, &src, &mut dest, false).unwrap();
        assert_eq!(dest, expected);
    }
}
