//! x86/x86_64 SIMD GF region kernels (SSE2 / SSSE3 / AVX2 / AVX-512BW / PCLMUL).

use crate::fec::gf::table::{Gf8ShuffleTables, Gf32SplitTables};

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(super) unsafe fn region_xor_avx512(src: &[u8], dest: &mut [u8]) -> usize {
    // SAFETY: caller verified AVX-512BW at runtime.
    unsafe {
        use core::arch::x86_64::*;
        let mut off = 0usize;
        for (s_chunk, d_chunk) in src.chunks_exact(64).zip(dest.chunks_exact_mut(64)) {
            let s = _mm512_loadu_si512(s_chunk.as_ptr() as *const __m512i);
            let d = _mm512_loadu_si512(d_chunk.as_ptr() as *const __m512i);
            let r = _mm512_xor_si512(s, d);
            _mm512_storeu_si512(d_chunk.as_mut_ptr() as *mut __m512i, r);
            off += 64;
        }
        off
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn region_xor_avx2(src: &[u8], dest: &mut [u8]) -> usize {
    // SAFETY: caller verified AVX2 at runtime.
    unsafe {
        use core::arch::x86_64::*;
        let mut off = 0usize;
        for (s_chunk, d_chunk) in src.chunks_exact(32).zip(dest.chunks_exact_mut(32)) {
            let s = _mm256_loadu_si256(s_chunk.as_ptr() as *const __m256i);
            let d = _mm256_loadu_si256(d_chunk.as_ptr() as *const __m256i);
            let r = _mm256_xor_si256(s, d);
            _mm256_storeu_si256(d_chunk.as_mut_ptr() as *mut __m256i, r);
            off += 32;
        }
        off
    }
}

#[target_feature(enable = "sse2")]
pub(super) unsafe fn region_xor_sse2(src: &[u8], dest: &mut [u8]) -> usize {
    // SAFETY: caller verified SSE2 at runtime.
    unsafe {
        use core::arch::x86_64::*;
        let mut off = 0usize;
        for (s_chunk, d_chunk) in src.chunks_exact(16).zip(dest.chunks_exact_mut(16)) {
            let s = _mm_loadu_si128(s_chunk.as_ptr() as *const __m128i);
            let d = _mm_loadu_si128(d_chunk.as_ptr() as *const __m128i);
            let r = _mm_xor_si128(s, d);
            _mm_storeu_si128(d_chunk.as_mut_ptr() as *mut __m128i, r);
            off += 16;
        }
        off
    }
}

#[target_feature(enable = "avx512bw")]
pub(super) unsafe fn region_mul_add_gf8_avx512(
    tables: &Gf8ShuffleTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> usize {
    // SAFETY: caller verified AVX-512BW at runtime.
    unsafe {
        let mut i = 0usize;
        while i + 64 <= src.len() {
            for lane in (0..64).step_by(16) {
                gf8_shuffle_lane(
                    src.as_ptr().add(i + lane),
                    dest.as_mut_ptr().add(i + lane),
                    tables,
                    xor_into,
                );
            }
            i += 64;
        }
        i
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn region_mul_add_gf8_avx2(
    tables: &Gf8ShuffleTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> usize {
    // SAFETY: caller verified AVX2 at runtime.
    unsafe {
        let mut i = 0usize;
        while i + 32 <= src.len() {
            for lane in [0usize, 16] {
                gf8_shuffle_lane(
                    src.as_ptr().add(i + lane),
                    dest.as_mut_ptr().add(i + lane),
                    tables,
                    xor_into,
                );
            }
            i += 32;
        }
        i
    }
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn region_mul_add_gf8_ssse3(
    tables: &Gf8ShuffleTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> usize {
    // SAFETY: caller verified SSSE3 at runtime.
    unsafe {
        let mut i = 0usize;
        while i + 16 <= src.len() {
            gf8_shuffle_lane(
                src.as_ptr().add(i),
                dest.as_mut_ptr().add(i),
                tables,
                xor_into,
            );
            i += 16;
        }
        i
    }
}

#[target_feature(enable = "ssse3")]
unsafe fn gf8_shuffle_lane(
    src: *const u8,
    dest: *mut u8,
    tables: &Gf8ShuffleTables,
    xor_into: bool,
) {
    // SAFETY: SSSE3 shuffle multiply on valid slice pointers.
    unsafe {
        use core::arch::x86_64::*;
        let lo_m = _mm_loadu_si128(tables.lo.as_ptr() as *const __m128i);
        let hi_m = _mm_loadu_si128(tables.hi.as_ptr() as *const __m128i);
        let mask = _mm_set1_epi8(0x0f);
        let s = _mm_loadu_si128(src as *const __m128i);
        let slo = _mm_and_si128(s, mask);
        let shi = _mm_and_si128(_mm_srli_epi64(s, 4), mask);
        let mut prod = _mm_xor_si128(_mm_shuffle_epi8(lo_m, slo), _mm_shuffle_epi8(hi_m, shi));
        if xor_into {
            let d = _mm_loadu_si128(dest as *const __m128i);
            prod = _mm_xor_si128(prod, d);
        }
        _mm_storeu_si128(dest as *mut __m128i, prod);
    }
}

#[target_feature(enable = "pclmulqdq", enable = "sse2")]
pub(super) unsafe fn gf32_mul_clmul(a: u32, b: u32) -> u32 {
    use core::arch::x86_64::*;

    let aa = _mm_cvtsi32_si128(a as i32);
    let bb = _mm_cvtsi32_si128(b as i32);

    // Carryless multiply: a × b in GF(2)[x] → 64-bit product.
    let prod = _mm_clmulepi64_si128(aa, bb, 0);

    // prim_poly = {0, 0, 1, GF32_POLY} → lower 64 bits = x³² + p(x)
    let prim_poly = _mm_set_epi32(0, 0, 1, super::super::table::GF32_POLY as i32);

    // CLMUL reduction (GF-Complete gf_w32_clm_multiply_N).
    let w = _mm_clmulepi64_si128(prim_poly, _mm_srli_si128(prod, 4), 0);
    let mut result = _mm_xor_si128(prod, w);
    let w = _mm_clmulepi64_si128(prim_poly, _mm_srli_si128(result, 4), 0);
    result = _mm_xor_si128(result, w);
    let w = _mm_clmulepi64_si128(prim_poly, _mm_srli_si128(result, 4), 0);
    result = _mm_xor_si128(result, w);
    let w = _mm_clmulepi64_si128(prim_poly, _mm_srli_si128(result, 4), 0);
    result = _mm_xor_si128(result, w);

    _mm_cvtsi128_si32(result) as u32
}

/// Split-table region multiply: 8 nibble lookups per lane via `vpgatherd` (AVX2).
///
/// Correct but ~2.8× slower than scalar split-tables on 256 B stripes (see
/// `gather_vs_scalar_throughput` test); not used in production dispatch.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn region_mul_add_gf32_avx2(
    tables: &Gf32SplitTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> usize {
    // SAFETY: caller verified AVX2 at runtime.
    unsafe {
        use core::arch::x86_64::*;
        let mask_f = _mm256_set1_epi32(0xf);
        let mut off = 0usize;
        while off + 32 <= src.len() {
            let s = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
            let mut acc = _mm256_setzero_si256();
            for (pos, row) in tables.tables.iter().enumerate() {
                let shifted = match pos {
                    0 => _mm256_srli_epi32(s, 0),
                    1 => _mm256_srli_epi32(s, 4),
                    2 => _mm256_srli_epi32(s, 8),
                    3 => _mm256_srli_epi32(s, 12),
                    4 => _mm256_srli_epi32(s, 16),
                    5 => _mm256_srli_epi32(s, 20),
                    6 => _mm256_srli_epi32(s, 24),
                    7 => _mm256_srli_epi32(s, 28),
                    _ => unreachable!(),
                };
                let idx = _mm256_and_si256(shifted, mask_f);
                let gathered = _mm256_i32gather_epi32(row.as_ptr() as *const i32, idx, 4);
                acc = _mm256_xor_si256(acc, gathered);
            }
            let out = if xor_into {
                let d = _mm256_loadu_si256(dest.as_ptr().add(off) as *const __m256i);
                _mm256_xor_si256(acc, d)
            } else {
                acc
            };
            _mm256_storeu_si256(dest.as_mut_ptr().add(off) as *mut __m256i, out);
            off += 32;
        }
        off
    }
}

/// Split-table region multiply with 16 parallel u32 lanes (AVX-512F gather).
///
/// Not used in production dispatch — see [`region_mul_add_gf32_avx2`].
#[allow(dead_code)]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn region_mul_add_gf32_avx512(
    tables: &Gf32SplitTables,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> usize {
    // SAFETY: caller verified AVX-512F at runtime.
    unsafe {
        use core::arch::x86_64::*;
        let mask_f = _mm512_set1_epi32(0xf);
        let mut off = 0usize;
        while off + 64 <= src.len() {
            let s = _mm512_loadu_si512(src.as_ptr().add(off) as *const __m512i);
            let mut acc = _mm512_setzero_si512();
            for (pos, row) in tables.tables.iter().enumerate() {
                let shifted = match pos {
                    0 => _mm512_srli_epi32(s, 0),
                    1 => _mm512_srli_epi32(s, 4),
                    2 => _mm512_srli_epi32(s, 8),
                    3 => _mm512_srli_epi32(s, 12),
                    4 => _mm512_srli_epi32(s, 16),
                    5 => _mm512_srli_epi32(s, 20),
                    6 => _mm512_srli_epi32(s, 24),
                    7 => _mm512_srli_epi32(s, 28),
                    _ => unreachable!(),
                };
                let idx = _mm512_and_si512(shifted, mask_f);
                let gathered = _mm512_i32gather_epi32(idx, row.as_ptr() as *const i32, 4);
                acc = _mm512_xor_si512(acc, gathered);
            }
            let out = if xor_into {
                let d = _mm512_loadu_si512(dest.as_ptr().add(off) as *const __m512i);
                _mm512_xor_si512(acc, d)
            } else {
                acc
            };
            _mm512_storeu_si512(dest.as_mut_ptr().add(off) as *mut __m512i, out);
            off += 64;
        }
        off
    }
}

#[target_feature(enable = "pclmulqdq", enable = "sse2")]
#[cfg(test)]
pub(super) unsafe fn region_mul_add_gf32_clmul(
    coeff: u32,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> crate::fec::error::FecResult<()> {
    use crate::fec::error::FecError;
    use crate::fec::gf::backend::scalar;
    // SAFETY: caller verified PCLMULQDQ at runtime.
    unsafe {
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
                scalar::region_xor(src, dest);
            } else {
                dest.copy_from_slice(src);
            }
            return Ok(());
        }

        let words = src.len() / 4;
        for i in 0..words {
            let off = i * 4;
            let s = u32::from_le_bytes(src[off..off + 4].try_into().expect("word"));
            let product = gf32_mul_clmul(s, coeff);
            if xor_into {
                let d = u32::from_le_bytes(dest[off..off + 4].try_into().expect("word"));
                dest[off..off + 4].copy_from_slice(&(product ^ d).to_le_bytes());
            } else {
                dest[off..off + 4].copy_from_slice(&product.to_le_bytes());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::gf::table::gf32_mul as gf32_mul_table;

    cpufeatures::new!(pclmul_token, "pclmulqdq");
    cpufeatures::new!(avx2_token, "avx2");
    cpufeatures::new!(avx512f_token, "avx512f");

    #[test]
    #[ignore = "micro-benchmark: cargo test gather_vs_scalar_throughput -- --ignored --nocapture"]
    fn gather_vs_scalar_throughput() {
        use std::time::Instant;

        let (_, has_avx2) = avx2_token::init_get();
        if !has_avx2 {
            eprintln!("AVX2 unavailable, skipping");
            return;
        }
        let coeff = 0x1234_5678;
        let tables = Gf32SplitTables::for_coeff(coeff);
        let src: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(3)).collect();
        let mut simd_dest = vec![0u8; 256];
        let mut scalar_dest = vec![0u8; 256];
        const ITERS: u32 = 50_000;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            // SAFETY: AVX2 was detected above.
            unsafe {
                region_mul_add_gf32_avx2(&tables, &src, &mut simd_dest, false);
            }
        }
        let simd_elapsed = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..ITERS {
            crate::fec::gf::backend::scalar::region_mul_add_gf32_tables(
                &tables,
                &src,
                &mut scalar_dest,
                false,
            )
            .unwrap();
        }
        let scalar_elapsed = t1.elapsed();

        assert_eq!(simd_dest, scalar_dest);
        eprintln!(
            "GF32 region 256B x{ITERS}: scalar={scalar_elapsed:?} simd_gather={simd_elapsed:?} ratio={:.2}x",
            simd_elapsed.as_secs_f64() / scalar_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn clmul_matches_table_when_available() {
        let (_, has_pclmul) = pclmul_token::init_get();
        if !has_pclmul {
            return;
        }
        for a in [1u32, 2, 0x400007, 0xdead_beef, 0x8000_0001] {
            for b in [1u32, 3, 0x9abc_def0, 0x1234_5678] {
                let expected = gf32_mul_table(a, b);
                // SAFETY: PCLMULQDQ was detected above.
                let got = unsafe { gf32_mul_clmul(a, b) };
                assert_eq!(expected, got, "a={a:#x} b={b:#x}");
            }
        }
    }

    #[test]
    fn clmul_region_matches_split() {
        let (_, has_pclmul) = pclmul_token::init_get();
        if !has_pclmul {
            return;
        }
        let coeff = 0x1234_5678;
        let src: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut clmul_dest = [0u8; 16];
        let mut split_dest = [0u8; 16];
        // SAFETY: PCLMULQDQ was detected above.
        unsafe {
            region_mul_add_gf32_clmul(coeff, &src, &mut clmul_dest, false).unwrap();
        }
        crate::fec::gf::backend::scalar::region_mul_add_gf32(coeff, &src, &mut split_dest, false)
            .unwrap();
        assert_eq!(clmul_dest, split_dest);
    }
}
