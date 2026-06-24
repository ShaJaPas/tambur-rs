//! Runtime CPU feature dispatch for GF kernels (x86: AVX-512BW > AVX2 > SSSE3 > scalar).

use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crate::fec::error::FecResult;
use crate::fec::gf::backend::{scalar, x86, x86_gf32_split4};
use crate::fec::gf::table::{Gf8ShuffleTables, Gf32SplitTables};

cpufeatures::new!(avx512bw_token, "avx512bw");
cpufeatures::new!(avx2_token, "avx2");
cpufeatures::new!(ssse3_token, "ssse3");
cpufeatures::new!(sse2_token, "sse2");
cpufeatures::new!(pclmul_token, "pclmulqdq");

#[derive(Copy, Clone)]
struct Features {
    avx512bw: bool,
    avx2: bool,
    ssse3: bool,
    sse2: bool,
    pclmul: bool,
}

impl Features {
    fn detect() -> Self {
        let (_, avx512bw) = avx512bw_token::init_get();
        let (_, avx2) = avx2_token::init_get();
        let (_, ssse3) = ssse3_token::init_get();
        let (_, sse2) = sse2_token::init_get();
        let (_, pclmul) = pclmul_token::init_get();
        Self {
            avx512bw,
            avx2,
            ssse3,
            sse2,
            pclmul,
        }
    }

    fn from_bits(bits: u8) -> Self {
        Self {
            avx512bw: bits & 1 != 0,
            avx2: bits & 2 != 0,
            ssse3: bits & 4 != 0,
            sse2: bits & 8 != 0,
            pclmul: bits & 16 != 0,
        }
    }

    fn to_bits(self) -> u8 {
        (if self.avx512bw { 1 } else { 0 })
            | ((if self.avx2 { 1 } else { 0 }) << 1)
            | ((if self.ssse3 { 1 } else { 0 }) << 2)
            | ((if self.sse2 { 1 } else { 0 }) << 3)
            | ((if self.pclmul { 1 } else { 0 }) << 4)
    }
}

static FEATURE_BITS: AtomicU8 = AtomicU8::new(0);
const FEATURES_INIT: u8 = 1 << 7;

fn features() -> Features {
    let cached = FEATURE_BITS.load(Ordering::Relaxed);
    if cached & FEATURES_INIT != 0 {
        return Features::from_bits(cached & !FEATURES_INIT);
    }
    let detected = Features::detect();
    FEATURE_BITS.store(FEATURES_INIT | detected.to_bits(), Ordering::Relaxed);
    detected
}

pub(super) fn region_xor(src: &[u8], dest: &mut [u8]) {
    debug_assert_eq!(src.len(), dest.len());
    let f = features();
    let mut off = 0usize;

    if f.avx512bw && src.len() >= 64 {
        // SAFETY: AVX-512BW was detected at runtime.
        off = unsafe { x86::region_xor_avx512(src, dest) };
    } else if f.avx2 && src.len() >= 32 {
        // SAFETY: AVX2 was detected at runtime.
        off = unsafe { x86::region_xor_avx2(src, dest) };
    } else if f.sse2 && src.len() >= 16 {
        // SAFETY: SSE2 was detected at runtime.
        off = unsafe { x86::region_xor_sse2(src, dest) };
    }

    scalar::region_xor_tail(src, dest, off);
}

pub(super) fn region_mul_add_gf8(
    coeff: u8,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    if coeff == 0 || coeff == 1 {
        return scalar::region_mul_add_gf8(coeff, src, dest, xor_into);
    }

    let tables = Gf8ShuffleTables::get(coeff);
    let f = features();
    let mut off = 0usize;

    if f.avx512bw && src.len() >= 64 {
        // SAFETY: AVX-512BW was detected at runtime.
        off = unsafe { x86::region_mul_add_gf8_avx512(tables, src, dest, xor_into) };
    } else if f.avx2 && src.len() >= 32 {
        // SAFETY: AVX2 (includes SSSE3 shuffles) was detected at runtime.
        off = unsafe { x86::region_mul_add_gf8_avx2(tables, src, dest, xor_into) };
    } else if f.ssse3 && src.len() >= 16 {
        // SAFETY: SSSE3 was detected at runtime.
        off = unsafe { x86::region_mul_add_gf8_ssse3(tables, src, dest, xor_into) };
    }

    if off < src.len() {
        scalar::region_mul_add_gf8_tables(tables, &src[off..], &mut dest[off..], xor_into)?;
    }
    Ok(())
}

pub(super) fn region_mul_add_gf32(
    coeff: u32,
    src: &[u8],
    dest: &mut [u8],
    xor_into: bool,
) -> FecResult<()> {
    if coeff == 0 || coeff == 1 {
        return scalar::region_mul_add_gf32(coeff, src, dest, xor_into);
    }

    let f = features();
    let mut off = 0usize;

    if f.ssse3 && src.len() >= 16 {
        #[cfg(feature = "std")]
        let shuffle = x86_gf32_split4::Gf32Split4ShuffleTables::for_coeff(coeff);
        #[cfg(not(feature = "std"))]
        let shuffle = x86_gf32_split4::Gf32Split4ShuffleTables::from_split(
            &Gf32SplitTables::for_coeff(coeff),
        );
        // SAFETY: SSSE3 was detected at runtime.
        off = unsafe {
            x86_gf32_split4::region_mul_add_gf32_split4_ssse3(&shuffle, src, dest, xor_into)
        };
    }

    if off < src.len() {
        let split = Gf32SplitTables::for_coeff(coeff);
        scalar::region_mul_add_gf32_tables(&split, &src[off..], &mut dest[off..], xor_into)?;
    }
    Ok(())
}

type Gf32Mul = fn(u32, u32) -> u32;

fn gf32_mul_pclmul(a: u32, b: u32) -> u32 {
    // SAFETY: PCLMULQDQ verified once at init; this fn is only installed when present.
    unsafe { x86::gf32_mul_clmul(a, b) }
}

fn detect_gf32_mul_impl() -> Gf32Mul {
    let (_, pclmul) = pclmul_token::init_get();
    if pclmul {
        gf32_mul_pclmul
    } else {
        scalar::gf32_mul_dispatch
    }
}

static GF32_MUL_IMPL: AtomicUsize = AtomicUsize::new(0);

pub(super) fn gf32_mul(a: u32, b: u32) -> u32 {
    let ptr = GF32_MUL_IMPL.load(Ordering::Relaxed);
    if ptr != 0 {
        // SAFETY: stores only valid fn pointers from detect_gf32_mul_impl().
        let f: Gf32Mul = unsafe { core::mem::transmute(ptr) };
        return f(a, b);
    }
    let f = detect_gf32_mul_impl();
    GF32_MUL_IMPL.store(f as usize, Ordering::Relaxed);
    f(a, b)
}

pub(super) fn gf8_mul(a: u8, b: u8) -> u8 {
    scalar::gf8_mul_dispatch(a, b)
}
