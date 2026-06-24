//! Cauchy distribution matrix generation (Jerasure `cauchy.c`).

use alloc::vec::Vec;

use crate::fec::error::{FecError, FecResult};
use crate::fec::gf::engine::{EngineW8, EngineW32, GfEngine};
use crate::word_size::WordSize;

/// Build the original Cauchy coding matrix (`cauchy_original_coding_matrix`).
///
/// Entry `(i, j)` is `1 / (i XOR (m + j))` in GF(2^w), row-major over `m × k`.
pub(crate) fn original_coding_matrix(k: usize, m: usize, w: WordSize) -> FecResult<Vec<i32>> {
    match w {
        WordSize::W8 => original_coding_matrix_impl::<EngineW8>(k, m),
        WordSize::W32 => original_coding_matrix_impl::<EngineW32>(k, m),
    }
}

fn original_coding_matrix_impl<G: GfEngine>(k: usize, m: usize) -> FecResult<Vec<i32>> {
    if k == 0 || m == 0 {
        return Err(FecError::InvalidMatrixDimensions);
    }
    if k + m > G::max_field_elements() {
        return Err(FecError::InvalidMatrixDimensions);
    }

    let mut matrix = Vec::with_capacity(k * m);
    for i in 0..m {
        for j in 0..k {
            let denom = G::add(i as u32, (m + j) as u32);
            let entry = G::div(1, denom)? as i32;
            matrix.push(entry);
        }
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::gf::gf32;

    #[test]
    fn small_matrix_entries() {
        let m = original_coding_matrix(3, 2, WordSize::W32).unwrap();
        // row 0, col 0: 1 / (0 ^ (2+0)) = 1/2
        assert_eq!(m[0], gf32::div(1, 2).unwrap() as i32);
        // row 0, col 1: 1 / (0 ^ 3) = 1/3
        assert_eq!(m[1], gf32::div(1, 3).unwrap() as i32);
        // row 1, col 0: 1 / (1 ^ 2) = 1/3
        assert_eq!(m[3], gf32::div(1, 3).unwrap() as i32);
    }

    #[test]
    fn dimensions() {
        let m = original_coding_matrix(10, 5, WordSize::W32).unwrap();
        assert_eq!(m.len(), 50);
    }
}
