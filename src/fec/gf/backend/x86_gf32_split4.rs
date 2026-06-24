//! GF(2^32) region multiply via SPLIT(4,32) + SSSE3 `pshufb` (GF-Complete style).
//!
//! Decomposes each 32-bit product into four byte lanes and uses byte shuffle tables
//! instead of `vpgatherd` on 32-bit split tables.

use core::arch::x86_64::__m128i;

use crate::fec::gf::table::Gf32SplitTables;

/// Byte shuffle tables: `tables[nibble_pos][byte_lane]` is a 16-byte `pshufb` lookup row.
#[derive(Clone, Copy)]
pub(super) struct Gf32Split4ShuffleTables {
    tables: [[__m128i; 4]; 8],
}

impl Gf32Split4ShuffleTables {
    pub(super) fn from_split(split: &Gf32SplitTables) -> Self {
        // SAFETY: SSSE3 intrinsics used only behind runtime feature checks.
        unsafe {
            use core::arch::x86_64::*;
            let mut tables = [[_mm_setzero_si128(); 4]; 8];
            for (pos, row) in split.tables.iter().enumerate() {
                let mut tmp = *row;
                for table_entry in tables[pos].iter_mut() {
                    let mut btable = [0u8; 16];
                    for (k, cell) in btable.iter_mut().enumerate() {
                        *cell = tmp[k] as u8;
                        tmp[k] >>= 8;
                    }
                    *table_entry = _mm_loadu_si128(btable.as_ptr() as *const __m128i);
                }
            }
            Self { tables }
        }
    }

    #[cfg(feature = "std")]
    pub(super) fn for_coeff(coeff: u32) -> Self {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};

        // Note: accesses Gf32SplitTables directly (not via its cached for_coeff)
        // to avoid deadlock on two-level caching.
        static CACHE: OnceLock<Mutex<HashMap<u32, Gf32Split4ShuffleTables>>> = OnceLock::new();
        {
            let cache = CACHE
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .expect("gf32 split4 cache lock");
            if let Some(&tables) = cache.get(&coeff) {
                return tables;
            }
        }
        let split = Gf32SplitTables::for_coeff(coeff);
        let tables = Self::from_split(&split);
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("gf32 split4 cache lock");
        cache.entry(coeff).or_insert(tables);
        tables
    }

    #[cfg(feature = "std")]
    pub(crate) fn warm_cache_for_coefficients(coeffs: &[i32]) {
        for &c in coeffs {
            let coeff = c as u32;
            if coeff > 1 {
                let _ = Self::for_coeff(coeff);
            }
        }
    }
}

/// Multiply 4 packed `u32` lanes by `coeff` using split-4/32 `pshufb`.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn gf32_mul_words_x4_ssse3(
    tables: &Gf32Split4ShuffleTables,
    src: __m128i,
) -> __m128i {
    use core::arch::x86_64::*;
    let mask_nibble = _mm_set1_epi32(0xf);
    let mask_byte = _mm_set_epi8(0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1);

    let mut p0 = _mm_setzero_si128();
    let mut p1 = _mm_setzero_si128();
    let mut p2 = _mm_setzero_si128();
    let mut p3 = _mm_setzero_si128();

    for (pos, row) in tables.tables.iter().enumerate() {
        let shifted = match pos {
            0 => src,
            1 => _mm_srli_epi32(src, 4),
            2 => _mm_srli_epi32(src, 8),
            3 => _mm_srli_epi32(src, 12),
            4 => _mm_srli_epi32(src, 16),
            5 => _mm_srli_epi32(src, 20),
            6 => _mm_srli_epi32(src, 24),
            7 => _mm_srli_epi32(src, 28),
            _ => unreachable!(),
        };
        let si = _mm_and_si128(shifted, mask_nibble);
        p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(row[0], si));
        p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(row[1], si));
        p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(row[2], si));
        p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(row[3], si));
    }

    let p0 = _mm_and_si128(p0, mask_byte);
    let p1 = _mm_and_si128(p1, mask_byte);
    let p2 = _mm_and_si128(p2, mask_byte);
    let p3 = _mm_and_si128(p3, mask_byte);

    let mut out = p0;
    out = _mm_xor_si128(out, _mm_slli_epi32(p1, 8));
    out = _mm_xor_si128(out, _mm_slli_epi32(p2, 16));
    out = _mm_xor_si128(out, _mm_slli_epi32(p3, 24));
    out
}

/// Region multiply-add: 64-byte (16-word) blocks, then 16-byte tail, matching GF-Complete.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn region_mul_add_gf32_split4_ssse3(
    tables: &Gf32Split4ShuffleTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> usize {
    // SAFETY: caller verified SSSE3 at runtime.
    unsafe {
        use core::arch::x86_64::*;
        let mask1 = _mm_set1_epi8(0xf);
        let mask8 = _mm_set1_epi16(0xff);
        let mut off = 0usize;

        while off + 64 <= src.len() {
            let mut v0 = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            let mut v1 = _mm_loadu_si128(src.as_ptr().add(off + 16) as *const __m128i);
            let mut v2 = _mm_loadu_si128(src.as_ptr().add(off + 32) as *const __m128i);
            let mut v3 = _mm_loadu_si128(src.as_ptr().add(off + 48) as *const __m128i);

            gf32_split4_transpose_16words_in(&mut v0, &mut v1, &mut v2, &mut v3, mask8);

            let (mut p0, mut p1, mut p2, mut p3) =
                gf32_split4_pshufb_accum_16words(tables, v0, v1, v2, v3, mask1);

            gf32_split4_transpose_16words_out(&mut p0, &mut p1, &mut p2, &mut p3);

            if xor_into {
                p0 = _mm_xor_si128(
                    p0,
                    _mm_loadu_si128(dest.as_ptr().add(off) as *const __m128i),
                );
                p1 = _mm_xor_si128(
                    p1,
                    _mm_loadu_si128(dest.as_ptr().add(off + 16) as *const __m128i),
                );
                p2 = _mm_xor_si128(
                    p2,
                    _mm_loadu_si128(dest.as_ptr().add(off + 32) as *const __m128i),
                );
                p3 = _mm_xor_si128(
                    p3,
                    _mm_loadu_si128(dest.as_ptr().add(off + 48) as *const __m128i),
                );
            }

            _mm_storeu_si128(dest.as_mut_ptr().add(off) as *mut __m128i, p0);
            _mm_storeu_si128(dest.as_mut_ptr().add(off + 16) as *mut __m128i, p1);
            _mm_storeu_si128(dest.as_mut_ptr().add(off + 32) as *mut __m128i, p2);
            _mm_storeu_si128(dest.as_mut_ptr().add(off + 48) as *mut __m128i, p3);
            off += 64;
        }

        while off + 16 <= src.len() {
            let s = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            let prod = gf32_mul_words_x4_ssse3(tables, s);
            let out = if xor_into {
                let d = _mm_loadu_si128(dest.as_ptr().add(off) as *const __m128i);
                _mm_xor_si128(prod, d)
            } else {
                prod
            };
            _mm_storeu_si128(dest.as_mut_ptr().add(off) as *mut __m128i, out);
            off += 16;
        }
        off
    }
}

#[target_feature(enable = "ssse3")]
unsafe fn gf32_split4_transpose_16words_in(
    v0: &mut __m128i,
    v1: &mut __m128i,
    v2: &mut __m128i,
    v3: &mut __m128i,
    mask8: __m128i,
) {
    use core::arch::x86_64::*;
    let mut p0 = _mm_srli_epi16(*v0, 8);
    let mut p1 = _mm_srli_epi16(*v1, 8);
    let mut p2 = _mm_srli_epi16(*v2, 8);
    let mut p3 = _mm_srli_epi16(*v3, 8);
    let tv0 = _mm_and_si128(*v0, mask8);
    let tv1 = _mm_and_si128(*v1, mask8);
    let tv2 = _mm_and_si128(*v2, mask8);
    let tv3 = _mm_and_si128(*v3, mask8);
    *v0 = _mm_packus_epi16(p1, p0);
    *v1 = _mm_packus_epi16(tv1, tv0);
    *v2 = _mm_packus_epi16(p3, p2);
    *v3 = _mm_packus_epi16(tv3, tv2);
    p0 = _mm_srli_epi16(*v0, 8);
    p1 = _mm_srli_epi16(*v1, 8);
    p2 = _mm_srli_epi16(*v2, 8);
    p3 = _mm_srli_epi16(*v3, 8);
    let tv0 = _mm_and_si128(*v0, mask8);
    let tv1 = _mm_and_si128(*v1, mask8);
    let tv2 = _mm_and_si128(*v2, mask8);
    let tv3 = _mm_and_si128(*v3, mask8);
    *v0 = _mm_packus_epi16(p2, p0);
    *v1 = _mm_packus_epi16(p3, p1);
    *v2 = _mm_packus_epi16(tv2, tv0);
    *v3 = _mm_packus_epi16(tv3, tv1);
}

#[target_feature(enable = "ssse3")]
unsafe fn gf32_split4_transpose_16words_out(
    p0: &mut __m128i,
    p1: &mut __m128i,
    p2: &mut __m128i,
    p3: &mut __m128i,
) {
    use core::arch::x86_64::*;
    let tv0 = _mm_unpackhi_epi8(*p1, *p3);
    let tv1 = _mm_unpackhi_epi8(*p0, *p2);
    let tv2 = _mm_unpacklo_epi8(*p1, *p3);
    let tv3 = _mm_unpacklo_epi8(*p0, *p2);
    *p0 = _mm_unpackhi_epi8(tv1, tv0);
    *p1 = _mm_unpacklo_epi8(tv1, tv0);
    *p2 = _mm_unpackhi_epi8(tv3, tv2);
    *p3 = _mm_unpacklo_epi8(tv3, tv2);
}

#[target_feature(enable = "ssse3")]
unsafe fn gf32_split4_pshufb_accum_16words(
    tables: &Gf32Split4ShuffleTables,
    mut v0: __m128i,
    mut v1: __m128i,
    mut v2: __m128i,
    mut v3: __m128i,
    mask1: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    use core::arch::x86_64::*;
    let mut si = _mm_and_si128(v0, mask1);
    let mut p0 = _mm_shuffle_epi8(tables.tables[6][0], si);
    let mut p1 = _mm_shuffle_epi8(tables.tables[6][1], si);
    let mut p2 = _mm_shuffle_epi8(tables.tables[6][2], si);
    let mut p3 = _mm_shuffle_epi8(tables.tables[6][3], si);
    v0 = _mm_srli_epi32(v0, 4);
    si = _mm_and_si128(v0, mask1);
    p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(tables.tables[7][0], si));
    p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(tables.tables[7][1], si));
    p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(tables.tables[7][2], si));
    p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(tables.tables[7][3], si));

    si = _mm_and_si128(v1, mask1);
    p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(tables.tables[4][0], si));
    p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(tables.tables[4][1], si));
    p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(tables.tables[4][2], si));
    p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(tables.tables[4][3], si));
    v1 = _mm_srli_epi32(v1, 4);
    si = _mm_and_si128(v1, mask1);
    p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(tables.tables[5][0], si));
    p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(tables.tables[5][1], si));
    p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(tables.tables[5][2], si));
    p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(tables.tables[5][3], si));

    si = _mm_and_si128(v2, mask1);
    p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(tables.tables[2][0], si));
    p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(tables.tables[2][1], si));
    p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(tables.tables[2][2], si));
    p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(tables.tables[2][3], si));
    v2 = _mm_srli_epi32(v2, 4);
    si = _mm_and_si128(v2, mask1);
    p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(tables.tables[3][0], si));
    p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(tables.tables[3][1], si));
    p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(tables.tables[3][2], si));
    p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(tables.tables[3][3], si));

    si = _mm_and_si128(v3, mask1);
    p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(tables.tables[0][0], si));
    p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(tables.tables[0][1], si));
    p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(tables.tables[0][2], si));
    p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(tables.tables[0][3], si));
    v3 = _mm_srli_epi32(v3, 4);
    si = _mm_and_si128(v3, mask1);
    p0 = _mm_xor_si128(p0, _mm_shuffle_epi8(tables.tables[1][0], si));
    p1 = _mm_xor_si128(p1, _mm_shuffle_epi8(tables.tables[1][1], si));
    p2 = _mm_xor_si128(p2, _mm_shuffle_epi8(tables.tables[1][2], si));
    p3 = _mm_xor_si128(p3, _mm_shuffle_epi8(tables.tables[1][3], si));
    (p0, p1, p2, p3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::gf::table::{Gf32SplitTables, gf32_mul};

    cpufeatures::new!(ssse3_token, "ssse3");

    #[test]
    #[ignore = "micro-benchmark: cargo test split4_vs_scalar_throughput --release -- --ignored --nocapture"]
    fn split4_vs_scalar_throughput() {
        use std::time::Instant;

        let (_, has_ssse3) = ssse3_token::init_get();
        if !has_ssse3 {
            eprintln!("SSSE3 unavailable, skipping");
            return;
        }
        let coeff = 0x1234_5678;
        let split = Gf32SplitTables::for_coeff(coeff);
        let shuffle = Gf32Split4ShuffleTables::for_coeff(coeff);
        let src: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(3)).collect();
        let mut simd_dest = vec![0u8; 256];
        let mut scalar_dest = vec![0u8; 256];
        const ITERS: u32 = 50_000;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            // SAFETY: SSSE3 was detected above.
            unsafe {
                region_mul_add_gf32_split4_ssse3(&shuffle, &src, &mut simd_dest, false);
            }
        }
        let simd_elapsed = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..ITERS {
            crate::fec::gf::backend::scalar::region_mul_add_gf32_tables(
                &split,
                &src,
                &mut scalar_dest,
                false,
            )
            .unwrap();
        }
        let scalar_elapsed = t1.elapsed();

        assert_eq!(simd_dest, scalar_dest);
        eprintln!(
            "GF32 region 256B x{ITERS}: scalar={scalar_elapsed:?} split4_pshufb={simd_elapsed:?} ratio={:.2}x",
            simd_elapsed.as_secs_f64() / scalar_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn split4_x4_matches_scalar() {
        let (_, has_ssse3) = ssse3_token::init_get();
        if !has_ssse3 {
            return;
        }
        let coeff = 0x1234_5678;
        let split = Gf32SplitTables::for_coeff(coeff);
        let shuffle = Gf32Split4ShuffleTables::from_split(&split);
        let words = [0xdead_beef_u32, 0x8000_0001, 0x400007, 0x12345678];
        let mut src = [0u8; 16];
        for (i, w) in words.iter().enumerate() {
            src[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        // SAFETY: SSSE3 was detected above.
        let got = unsafe {
            use core::arch::x86_64::*;
            let s = _mm_loadu_si128(src.as_ptr() as *const __m128i);
            gf32_mul_words_x4_ssse3(&shuffle, s)
        };
        let mut got_bytes = [0u8; 16];
        // SAFETY: storing xmm to bytes.
        unsafe {
            use core::arch::x86_64::*;
            _mm_storeu_si128(got_bytes.as_mut_ptr() as *mut __m128i, got);
        }
        for (i, &w) in words.iter().enumerate() {
            let expected = gf32_mul(w, coeff);
            let got_word = u32::from_le_bytes(got_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(got_word, expected, "word {i}");
        }
    }

    #[test]
    fn split4_region_matches_scalar() {
        let (_, has_ssse3) = ssse3_token::init_get();
        if !has_ssse3 {
            return;
        }
        let coeff = 0x9abc_def0;
        let split = Gf32SplitTables::for_coeff(coeff);
        let shuffle = Gf32Split4ShuffleTables::from_split(&split);
        let src: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(5)).collect();
        let mut simd_dest = vec![0u8; 256];
        let mut scalar_dest = vec![0u8; 256];
        // SAFETY: SSSE3 was detected above.
        unsafe {
            region_mul_add_gf32_split4_ssse3(&shuffle, &src, &mut simd_dest, false);
        }
        crate::fec::gf::backend::scalar::region_mul_add_gf32_tables(
            &split,
            &src,
            &mut scalar_dest,
            false,
        )
        .unwrap();
        assert_eq!(simd_dest, scalar_dest);
    }
}
