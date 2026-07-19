//! AArch64 NEON GF region kernels.

use core::arch::aarch64::*;

use crate::fec::error::FecResult;
use crate::fec::gf::table::{Gf8ShuffleTables, Gf32SplitTables};

pub(super) fn region_xor(src: &[u8], dest: &mut [u8]) {
    debug_assert_eq!(src.len(), dest.len());
    unsafe {
        let mut off = 0usize;
        while off + 64 <= src.len() {
            let s0 = vld1q_u8(src.as_ptr().add(off));
            let s1 = vld1q_u8(src.as_ptr().add(off + 16));
            let s2 = vld1q_u8(src.as_ptr().add(off + 32));
            let s3 = vld1q_u8(src.as_ptr().add(off + 48));
            let d0 = vld1q_u8(dest.as_ptr().add(off));
            let d1 = vld1q_u8(dest.as_ptr().add(off + 16));
            let d2 = vld1q_u8(dest.as_ptr().add(off + 32));
            let d3 = vld1q_u8(dest.as_ptr().add(off + 48));
            vst1q_u8(dest.as_mut_ptr().add(off), veorq_u8(s0, d0));
            vst1q_u8(dest.as_mut_ptr().add(off + 16), veorq_u8(s1, d1));
            vst1q_u8(dest.as_mut_ptr().add(off + 32), veorq_u8(s2, d2));
            vst1q_u8(dest.as_mut_ptr().add(off + 48), veorq_u8(s3, d3));
            off += 64;
        }
        while off + 16 <= src.len() {
            let s = vld1q_u8(src.as_ptr().add(off));
            let d = vld1q_u8(dest.as_ptr().add(off));
            vst1q_u8(dest.as_mut_ptr().add(off), veorq_u8(s, d));
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
    if coeff == 0 || coeff == 1 || src.len() < 16 {
        return super::scalar::region_mul_add_gf32(coeff, src, dest, xor_into);
    }
    #[cfg(feature = "std")]
    let tables = Gf32Split4NeonTables::for_coeff(coeff);
    #[cfg(not(feature = "std"))]
    let tables = {
        let split = Gf32SplitTables::for_coeff(coeff);
        Gf32Split4NeonTables::from_split(&split)
    };
    unsafe {
        let off = region_mul_add_gf32_neon(&tables, src, dest, xor_into);
        if off < src.len() {
            super::scalar::region_mul_add_gf32_tables(
                &Gf32SplitTables::for_coeff(coeff),
                &src[off..],
                &mut dest[off..],
                xor_into,
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "std")]
pub(crate) fn warm_gf32_shuffle_cache(coeffs: &[i32]) {
    for &c in coeffs {
        let coeff = c as u32;
        if coeff > 1 {
            let _ = Gf32Split4NeonTables::for_coeff(coeff);
        }
    }
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
    unsafe {
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

/// NEON split-4/32 tables using `tbl` instead of x86 `pshufb`.
#[derive(Clone, Copy)]
struct Gf32Split4NeonTables {
    tables: [[uint8x16_t; 4]; 8],
}

impl Gf32Split4NeonTables {
    fn from_split(split: &Gf32SplitTables) -> Self {
        unsafe {
            let mut tables = [[vdupq_n_u8(0); 4]; 8];
            for (pos, row) in split.tables.iter().enumerate() {
                let mut tmp = *row;
                for table_entry in tables[pos].iter_mut() {
                    let mut btable = [0u8; 16];
                    for (k, cell) in btable.iter_mut().enumerate() {
                        *cell = tmp[k] as u8;
                        tmp[k] >>= 8;
                    }
                    *table_entry = vld1q_u8(btable.as_ptr());
                }
            }
            Self { tables }
        }
    }

    #[cfg(feature = "std")]
    fn for_coeff(coeff: u32) -> Self {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};

        static CACHE: OnceLock<Mutex<HashMap<u32, Gf32Split4NeonTables>>> = OnceLock::new();
        {
            let cache = CACHE
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .expect("neon gf32 split4 cache lock");
            if let Some(&tables) = cache.get(&coeff) {
                return tables;
            }
        }
        let split = Gf32SplitTables::for_coeff(coeff);
        let tables = Self::from_split(&split);
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("neon gf32 split4 cache lock");
        cache.entry(coeff).or_insert(tables);
        tables
    }
}

/// Multiply 4 packed `u32` lanes by `coeff` using NEON split-4/32 `tbl`.
unsafe fn gf32_mul_words_x4_neon(tables: &Gf32Split4NeonTables, src: uint8x16_t) -> uint8x16_t {
    unsafe {
        let mask_nibble = vreinterpretq_u8_u32(vdupq_n_u32(0x0f));

        let mut p0 = vdupq_n_u8(0);
        let mut p1 = vdupq_n_u8(0);
        let mut p2 = vdupq_n_u8(0);
        let mut p3 = vdupq_n_u8(0);

        for (pos, row) in tables.tables.iter().enumerate() {
            let shifted = match pos {
                0 => src,
                1 => vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(src), 4)),
                2 => vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(src), 8)),
                3 => vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(src), 12)),
                4 => vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(src), 16)),
                5 => vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(src), 20)),
                6 => vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(src), 24)),
                7 => vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(src), 28)),
                _ => unreachable!(),
            };
            let si = vandq_u8(shifted, mask_nibble);
            p0 = veorq_u8(p0, vqtbl1q_u8(row[0], si));
            p1 = veorq_u8(p1, vqtbl1q_u8(row[1], si));
            p2 = veorq_u8(p2, vqtbl1q_u8(row[2], si));
            p3 = veorq_u8(p3, vqtbl1q_u8(row[3], si));
        }

        let mut out = p0;
        out = veorq_u8(
            out,
            vreinterpretq_u8_u32(vshlq_n_u32(vreinterpretq_u32_u8(p1), 8)),
        );
        out = veorq_u8(
            out,
            vreinterpretq_u8_u32(vshlq_n_u32(vreinterpretq_u32_u8(p2), 16)),
        );
        out = veorq_u8(
            out,
            vreinterpretq_u8_u32(vshlq_n_u32(vreinterpretq_u32_u8(p3), 24)),
        );
        out
    }
}

/// NEON region multiply-add for GF(2³²) — 64-byte (16-word) blocks, then 16-byte tail.
unsafe fn region_mul_add_gf32_neon(
    tables: &Gf32Split4NeonTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> usize {
    unsafe {
        let mut off = 0usize;

        while off + 64 <= src.len() {
            let mut v0 = vld1q_u8(src.as_ptr().add(off));
            let mut v1 = vld1q_u8(src.as_ptr().add(off + 16));
            let mut v2 = vld1q_u8(src.as_ptr().add(off + 32));
            let mut v3 = vld1q_u8(src.as_ptr().add(off + 48));

            gf32_split4_transpose_16words_in(&mut v0, &mut v1, &mut v2, &mut v3);

            let (mut p0, mut p1, mut p2, mut p3) =
                gf32_split4_neon_accum_16words(tables, v0, v1, v2, v3);

            gf32_split4_transpose_16words_out(&mut p0, &mut p1, &mut p2, &mut p3);

            if xor_into {
                p0 = veorq_u8(p0, vld1q_u8(dest.as_ptr().add(off)));
                p1 = veorq_u8(p1, vld1q_u8(dest.as_ptr().add(off + 16)));
                p2 = veorq_u8(p2, vld1q_u8(dest.as_ptr().add(off + 32)));
                p3 = veorq_u8(p3, vld1q_u8(dest.as_ptr().add(off + 48)));
            }

            vst1q_u8(dest.as_mut_ptr().add(off), p0);
            vst1q_u8(dest.as_mut_ptr().add(off + 16), p1);
            vst1q_u8(dest.as_mut_ptr().add(off + 32), p2);
            vst1q_u8(dest.as_mut_ptr().add(off + 48), p3);
            off += 64;
        }

        while off + 16 <= src.len() {
            let s = vld1q_u8(src.as_ptr().add(off));
            let prod = gf32_mul_words_x4_neon(tables, s);
            let out = if xor_into {
                veorq_u8(prod, vld1q_u8(dest.as_ptr().add(off)))
            } else {
                prod
            };
            vst1q_u8(dest.as_mut_ptr().add(off), out);
            off += 16;
        }
        off
    }
}

/// Transpose 16 words (4 vectors) into byte-lane grouped layout.
///
/// Input:  v0..v3 each contain 4 u32 words (16 bytes per vector).
/// Output: v0 = byte 3 of all 16 words,
///         v1 = byte 2,
///         v2 = byte 1,
///         v3 = byte 0.
unsafe fn gf32_split4_transpose_16words_in(
    v0: &mut uint8x16_t,
    v1: &mut uint8x16_t,
    v2: &mut uint8x16_t,
    v3: &mut uint8x16_t,
) {
    unsafe {
        let mask8 = vdupq_n_u16(0x00FF);

        // Level 1: split each u16 into low/high byte
        let p0 = vshrq_n_u16(vreinterpretq_u16_u8(*v0), 8);
        let p1 = vshrq_n_u16(vreinterpretq_u16_u8(*v1), 8);
        let p2 = vshrq_n_u16(vreinterpretq_u16_u8(*v2), 8);
        let p3 = vshrq_n_u16(vreinterpretq_u16_u8(*v3), 8);

        let tv0 = vandq_u16(vreinterpretq_u16_u8(*v0), mask8);
        let tv1 = vandq_u16(vreinterpretq_u16_u8(*v1), mask8);
        let tv2 = vandq_u16(vreinterpretq_u16_u8(*v2), mask8);
        let tv3 = vandq_u16(vreinterpretq_u16_u8(*v3), mask8);

        *v0 = vcombine_u8(vqmovn_u16(p1), vqmovn_u16(p0));
        *v1 = vcombine_u8(vqmovn_u16(tv1), vqmovn_u16(tv0));
        *v2 = vcombine_u8(vqmovn_u16(p3), vqmovn_u16(p2));
        *v3 = vcombine_u8(vqmovn_u16(tv3), vqmovn_u16(tv2));

        // Level 2: further interleave
        let p0 = vshrq_n_u16(vreinterpretq_u16_u8(*v0), 8);
        let p1 = vshrq_n_u16(vreinterpretq_u16_u8(*v1), 8);
        let p2 = vshrq_n_u16(vreinterpretq_u16_u8(*v2), 8);
        let p3 = vshrq_n_u16(vreinterpretq_u16_u8(*v3), 8);

        let tv0 = vandq_u16(vreinterpretq_u16_u8(*v0), mask8);
        let tv1 = vandq_u16(vreinterpretq_u16_u8(*v1), mask8);
        let tv2 = vandq_u16(vreinterpretq_u16_u8(*v2), mask8);
        let tv3 = vandq_u16(vreinterpretq_u16_u8(*v3), mask8);

        *v0 = vcombine_u8(vqmovn_u16(p2), vqmovn_u16(p0));
        *v1 = vcombine_u8(vqmovn_u16(p3), vqmovn_u16(p1));
        *v2 = vcombine_u8(vqmovn_u16(tv2), vqmovn_u16(tv0));
        *v3 = vcombine_u8(vqmovn_u16(tv3), vqmovn_u16(tv1));
    }
}

/// Transpose back: byte-lane grouped → 4 word vectors.
unsafe fn gf32_split4_transpose_16words_out(
    p0: &mut uint8x16_t,
    p1: &mut uint8x16_t,
    p2: &mut uint8x16_t,
    p3: &mut uint8x16_t,
) {
    unsafe {
        let z13 = vzipq_u8(*p1, *p3);
        let z02 = vzipq_u8(*p0, *p2);

        let tv0 = z13.1;
        let tv1 = z02.1;
        let tv2 = z13.0;
        let tv3 = z02.0;

        let z_tv10 = vzipq_u8(tv1, tv0);
        let z_tv32 = vzipq_u8(tv3, tv2);

        *p0 = z_tv10.1;
        *p1 = z_tv10.0;
        *p2 = z_tv32.1;
        *p3 = z_tv32.0;
    }
}

/// Accumulate all 8 nibble positions for 16 transposed words.
///
/// Input:  v0 = byte 3 of all 16 words, v1 = byte 2, v2 = byte 1, v3 = byte 0.
/// Output: p0..p3 = byte lanes 0..3 of the product, still in transposed layout.
unsafe fn gf32_split4_neon_accum_16words(
    tables: &Gf32Split4NeonTables,
    mut v0: uint8x16_t,
    mut v1: uint8x16_t,
    mut v2: uint8x16_t,
    mut v3: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    unsafe {
        let mask1 = vdupq_n_u8(0x0f);

        // v0 = byte 3 of each word → nibble positions 6 (low), 7 (high)
        let mut si = vandq_u8(v0, mask1);
        let mut p0 = vqtbl1q_u8(tables.tables[6][0], si);
        let mut p1 = vqtbl1q_u8(tables.tables[6][1], si);
        let mut p2 = vqtbl1q_u8(tables.tables[6][2], si);
        let mut p3 = vqtbl1q_u8(tables.tables[6][3], si);
        v0 = vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(v0), 4));
        si = vandq_u8(v0, mask1);
        p0 = veorq_u8(p0, vqtbl1q_u8(tables.tables[7][0], si));
        p1 = veorq_u8(p1, vqtbl1q_u8(tables.tables[7][1], si));
        p2 = veorq_u8(p2, vqtbl1q_u8(tables.tables[7][2], si));
        p3 = veorq_u8(p3, vqtbl1q_u8(tables.tables[7][3], si));

        // v1 = byte 2 → nibble positions 4, 5
        si = vandq_u8(v1, mask1);
        p0 = veorq_u8(p0, vqtbl1q_u8(tables.tables[4][0], si));
        p1 = veorq_u8(p1, vqtbl1q_u8(tables.tables[4][1], si));
        p2 = veorq_u8(p2, vqtbl1q_u8(tables.tables[4][2], si));
        p3 = veorq_u8(p3, vqtbl1q_u8(tables.tables[4][3], si));
        v1 = vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(v1), 4));
        si = vandq_u8(v1, mask1);
        p0 = veorq_u8(p0, vqtbl1q_u8(tables.tables[5][0], si));
        p1 = veorq_u8(p1, vqtbl1q_u8(tables.tables[5][1], si));
        p2 = veorq_u8(p2, vqtbl1q_u8(tables.tables[5][2], si));
        p3 = veorq_u8(p3, vqtbl1q_u8(tables.tables[5][3], si));

        // v2 = byte 1 → nibble positions 2, 3
        si = vandq_u8(v2, mask1);
        p0 = veorq_u8(p0, vqtbl1q_u8(tables.tables[2][0], si));
        p1 = veorq_u8(p1, vqtbl1q_u8(tables.tables[2][1], si));
        p2 = veorq_u8(p2, vqtbl1q_u8(tables.tables[2][2], si));
        p3 = veorq_u8(p3, vqtbl1q_u8(tables.tables[2][3], si));
        v2 = vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(v2), 4));
        si = vandq_u8(v2, mask1);
        p0 = veorq_u8(p0, vqtbl1q_u8(tables.tables[3][0], si));
        p1 = veorq_u8(p1, vqtbl1q_u8(tables.tables[3][1], si));
        p2 = veorq_u8(p2, vqtbl1q_u8(tables.tables[3][2], si));
        p3 = veorq_u8(p3, vqtbl1q_u8(tables.tables[3][3], si));

        // v3 = byte 0 → nibble positions 0, 1
        si = vandq_u8(v3, mask1);
        p0 = veorq_u8(p0, vqtbl1q_u8(tables.tables[0][0], si));
        p1 = veorq_u8(p1, vqtbl1q_u8(tables.tables[0][1], si));
        p2 = veorq_u8(p2, vqtbl1q_u8(tables.tables[0][2], si));
        p3 = veorq_u8(p3, vqtbl1q_u8(tables.tables[0][3], si));
        v3 = vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(v3), 4));
        si = vandq_u8(v3, mask1);
        p0 = veorq_u8(p0, vqtbl1q_u8(tables.tables[1][0], si));
        p1 = veorq_u8(p1, vqtbl1q_u8(tables.tables[1][1], si));
        p2 = veorq_u8(p2, vqtbl1q_u8(tables.tables[1][2], si));
        p3 = veorq_u8(p3, vqtbl1q_u8(tables.tables[1][3], si));

        (p0, p1, p2, p3)
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

    #[test]
    fn neon_gf32_split4_word_matches_scalar() {
        let coeff = 0x1234_5678;
        let split = Gf32SplitTables::for_coeff(coeff);
        let tables = Gf32Split4NeonTables::from_split(&split);
        let words = [0xdead_beef_u32, 0x8000_0001, 0x400007, 0x12345678];
        let mut src = [0u8; 16];
        for (i, w) in words.iter().enumerate() {
            src[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        let got = unsafe { gf32_mul_words_x4_neon(&tables, vld1q_u8(src.as_ptr())) };
        let mut got_bytes = [0u8; 16];
        unsafe {
            vst1q_u8(got_bytes.as_mut_ptr(), got);
        }
        for (i, &w) in words.iter().enumerate() {
            let expected = crate::fec::gf::table::gf32_mul_bit(w, coeff);
            let got_word = u32::from_le_bytes(got_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(got_word, expected, "word {i}");
        }
    }

    #[test]
    fn neon_gf32_split4_region_matches_scalar() {
        let coeff = 0x9abc_def0;
        let split = Gf32SplitTables::for_coeff(coeff);
        let tables = Gf32Split4NeonTables::from_split(&split);
        let src: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(5)).collect();
        let mut neon_dest = vec![0u8; 256];
        let mut scalar_dest = vec![0u8; 256];
        unsafe {
            region_mul_add_gf32_neon(&tables, &src, &mut neon_dest, false);
        }
        crate::fec::gf::backend::scalar::region_mul_add_gf32_tables(
            &split,
            &src,
            &mut scalar_dest,
            false,
        )
        .unwrap();
        assert_eq!(neon_dest, scalar_dest);
    }
}
