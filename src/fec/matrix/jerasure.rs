//! Jerasure matrix encode/decode for GF(2^8) and GF(2^32).
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::any::TypeId;

use crate::fec::error::{FecError, FecResult};
use crate::fec::gf::engine::{EngineW8, EngineW32, GfEngine};
use crate::fec::matrix::Matrix;
use crate::fec::payload_matrix::PayloadMatrix;
use crate::word_size::WordSize;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Encode `m` parity stripes from `k` data stripes using a row-major `m × k` matrix.
///
/// `stripes[0..k]` are data devices; `stripes[k..k+m]` are parity outputs.
#[cfg(test)]
pub(crate) fn matrix_encode(
    w: WordSize,
    k: usize,
    m: usize,
    matrix: &[i32],
    stripes: &mut [Vec<u8>],
    size: usize,
) -> FecResult<()> {
    match w {
        WordSize::W8 => matrix_encode_impl::<EngineW8>(k, m, matrix, stripes, size),
        WordSize::W32 => matrix_encode_impl::<EngineW32>(k, m, matrix, stripes, size),
    }
}

#[cfg(test)]
fn matrix_encode_impl<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &[i32],
    stripes: &mut [Vec<u8>],
    size: usize,
) -> FecResult<()> {
    validate_stripes::<G>(k, m, matrix, stripes, size)?;
    for i in 0..m {
        matrix_dotprod::<G>(k, &matrix[i * k..(i + 1) * k], None, k + i, stripes, size)?;
    }
    Ok(())
}

/// Encode using coefficient rows stored in a strided [`Matrix`] (no dense row copy).
pub(crate) fn matrix_encode_payloads_strided(
    w: WordSize,
    k: usize,
    m: usize,
    matrix: &Matrix,
    first_row: usize,
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    match w {
        WordSize::W8 => matrix_encode_payloads_strided_impl::<EngineW8>(
            k, m, matrix, first_row, data, coding, size,
        ),
        WordSize::W32 => matrix_encode_payloads_strided_impl::<EngineW32>(
            k, m, matrix, first_row, data, coding, size,
        ),
    }
}

fn matrix_encode_payloads_strided_impl<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &Matrix,
    first_row: usize,
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    validate_payload_matrices_strided::<G>(k, m, matrix, first_row, data, coding, size)?;

    for i in 0..m {
        matrix_dotprod_payloads::<G>(
            k,
            matrix.row_coefficients(first_row + i)?,
            None,
            i,
            false,
            data,
            coding,
            size,
        )?;
    }
    Ok(())
}

/// Like [`matrix_dotprod_payloads`], but writes into a caller-provided `dest` slice
/// (must be exactly `size` bytes) and takes `data` as shared reference.
///
/// Safe for parallel calls with disjoint `dest` slices.
#[cfg(feature = "parallel")]
fn matrix_dotprod_payloads_dest<G: GfEngine>(
    k: usize,
    matrix_row: &[i32],
    data: &PayloadMatrix,
    dest: &mut [u8],
    size: usize,
) -> FecResult<()> {
    let mut init = false;
    for i in 0..k {
        let coeff = matrix_row[i];
        if coeff == 0 {
            continue;
        }
        // Encode always reads from data via the identity mapping (src_ids = None).
        let src = data.row(i)?;
        if coeff == 1 {
            if !init {
                dest[..size].copy_from_slice(&src[..size]);
                init = true;
            } else {
                G::region_xor(&src[..size], dest);
            }
        } else {
            G::region_mul_add(coeff_to_u32(coeff), &src[..size], dest, init)?;
            init = true;
        }
    }
    Ok(())
}

/// Parallel variant of [`matrix_encode_payloads_strided_impl`] using rayon.
///
/// Spawns one rayon task per parity row (`i in 0..m`). Each task reads from the
/// shared `data` matrix (read-only) and writes to a disjoint row of `coding`.
#[cfg(feature = "parallel")]
pub(crate) fn matrix_encode_payloads_strided_parallel<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &Matrix,
    first_row: usize,
    data: &PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    let mut dest_rows = coding.chunk_rows_mut(0, m)?;
    dest_rows
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(i, dest)| {
            let coeffs = matrix
                .row_coefficients(first_row + i)
                .map_err(|_| FecError::InvalidMatrixDimensions)?;
            matrix_dotprod_payloads_dest::<G>(k, coeffs, data, dest, size)
        })
}

/// Dispatch to parallel or sequential encode.
#[cfg(feature = "parallel")]
pub(crate) fn matrix_encode_payloads_strided_dispatch(
    w: WordSize,
    k: usize,
    m: usize,
    matrix: &Matrix,
    first_row: usize,
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    if m >= 4 {
        match w {
            WordSize::W8 => matrix_encode_payloads_strided_parallel::<EngineW8>(
                k, m, matrix, first_row, &*data, coding, size,
            ),
            WordSize::W32 => matrix_encode_payloads_strided_parallel::<EngineW32>(
                k, m, matrix, first_row, &*data, coding, size,
            ),
        }
    } else {
        matrix_encode_payloads_strided(w, k, m, matrix, first_row, data, coding, size)
    }
}

/// Decode with [`PayloadMatrix`] data/parity storage.
pub(crate) fn matrix_decode_payloads(
    w: WordSize,
    k: usize,
    m: usize,
    matrix: &[i32],
    row_k_ones: bool,
    erasures: &[i32],
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    match w {
        WordSize::W8 => matrix_decode_payloads_impl::<EngineW8>(
            k, m, matrix, row_k_ones, erasures, data, coding, size,
        ),
        WordSize::W32 => matrix_decode_payloads_impl::<EngineW32>(
            k, m, matrix, row_k_ones, erasures, data, coding, size,
        ),
    }
}

fn matrix_decode_payloads_impl<G: GfEngine + 'static>(
    k: usize,
    m: usize,
    matrix: &[i32],
    row_k_ones: bool,
    erasures: &[i32],
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    validate_payload_matrices::<G>(k, m, matrix, data, coding, size)?;
    let erased = erasures_to_erased(k, m, erasures)?;

    let mut edd = 0usize;
    let mut lastdrive = k;
    for i in 0..k {
        if erased[i] != 0 {
            edd += 1;
            lastdrive = i;
        }
    }
    if !row_k_ones || erased[k] != 0 {
        lastdrive = k;
    }

    if edd > 1 || (edd > 0 && (!row_k_ones || erased[k] != 0)) {
        // --- Reduced edd×edd inversion (§3.2) ---
        // Instead of building a k×k matrix (k = 448) and inverting it via
        // Gaussian elimination (17.5M CLMULs), we build an edd×edd matrix
        // from the surviving parity rows and invert that (~90K CLMULs).
        let mut survivor_data = Vec::with_capacity(k);
        let mut survivor_parity = Vec::with_capacity(m);
        let mut erased_data = Vec::with_capacity(edd);

        for i in 0..k {
            if erased[i] == 0 {
                survivor_data.push(i as i32);
            } else {
                erased_data.push(i as i32);
            }
        }
        let ndead = erased_data.len();
        for i in k..(k + m) {
            if erased[i] == 0 {
                survivor_parity.push((i - k) as i32);
            }
        }
        if survivor_parity.len() < ndead {
            return Err(FecError::TooManyErasures);
        }

        // A[eq][e] = Cauchy[parity_id][erased_data[e]]    (ndead×ndead)
        let mut a_tmp = vec![0i32; ndead * ndead];
        for eq in 0..ndead {
            let pid = survivor_parity[eq] as usize;
            let off = eq * ndead;
            for e in 0..ndead {
                let eid = erased_data[e] as usize;
                a_tmp[off + e] = matrix[pid * k + eid];
            }
        }

        let mut a_inv = vec![0i32; ndead * ndead];
        invert_matrix_impl::<G>(&mut a_tmp, &mut a_inv, ndead)?;

        #[cfg(feature = "std")]
        if TypeId::of::<G>() == TypeId::of::<EngineW32>() {
            crate::fec::gf::table::warm_gf32_coeff_cache(&a_inv);
        }

        // Compute b_p for each equation: b[eq] = parity[p] - sum_{known j} Cauchy[p][j] * data[j]
        let known_stripes: Vec<&[u8]> = survivor_data
            .iter()
            .map(|&id| data.row(id as usize))
            .collect::<FecResult<Vec<_>>>()?;
        let parity_stripes: Vec<&[u8]> = survivor_parity
            .iter()
            .map(|&id| coding.row(id as usize))
            .collect::<FecResult<Vec<_>>>()?;

        let mut scratch: Vec<Vec<u8>> = Vec::with_capacity(ndead);
        for eq in 0..ndead {
            let pid = survivor_parity[eq] as usize;
            let mut b = parity_stripes[eq][..size].to_vec();
            for (&did, src) in survivor_data.iter().zip(&known_stripes) {
                let coeff = coeff_to_u32(matrix[pid * k + did as usize]);
                if coeff == 0 {
                    continue;
                }
                if coeff == 1 {
                    G::region_xor(&src[..size], &mut b[..size]);
                } else {
                    G::region_mul_add(coeff, &src[..size], &mut b[..size], true)?;
                }
            }
            scratch.push(b);
        }
        drop(known_stripes);
        drop(parity_stripes);

        // Recover erased data: data[e] = sum_{eq} a_inv[e][eq] * b[eq]
        let limit = core::cmp::min(lastdrive, k);
        for e_idx in 0..ndead {
            let eid = erased_data[e_idx] as usize;
            if eid >= limit {
                continue;
            }
            let dest = data.row_mut(eid)?;
            let mut init = false;
            for eq in 0..ndead {
                let coeff = coeff_to_u32(a_inv[e_idx * ndead + eq]);
                if coeff == 0 {
                    continue;
                }
                if coeff == 1 {
                    if !init {
                        dest[..size].copy_from_slice(&scratch[eq][..size]);
                        init = true;
                    } else {
                        G::region_xor(&scratch[eq][..size], &mut dest[..size]);
                    }
                } else {
                    G::region_mul_add(coeff, &scratch[eq][..size], &mut dest[..size], init)?;
                    init = true;
                }
            }
        }
    }

    if edd > 0 && lastdrive < k && erased[lastdrive] != 0 {
        let tmpids: Vec<i32> = (0..k)
            .map(|i| {
                if i < lastdrive {
                    i as i32
                } else {
                    (i + 1) as i32
                }
            })
            .collect();
        matrix_dotprod_payloads::<G>(
            k,
            matrix,
            Some(&tmpids),
            lastdrive,
            true,
            data,
            coding,
            size,
        )?;
    }

    for i in 0..m {
        if erased[k + i] != 0 {
            matrix_dotprod_payloads::<G>(
                k,
                &matrix[i * k..(i + 1) * k],
                None,
                i,
                false,
                data,
                coding,
                size,
            )?;
        }
    }
    Ok(())
}

/// Parallel variant of [`matrix_decode_payloads_impl`].
///
/// Phase 2 (erased data stripes) and Phase 4 (erased parity stripes) are each
/// dispatched across rayon threads. Reads are done from shared `data`/`coding`
/// references while each task writes to a disjoint output row via a raw pointer.
#[cfg(feature = "parallel")]
pub(crate) fn matrix_decode_payloads_parallel<G: GfEngine + 'static>(
    k: usize,
    m: usize,
    matrix: &[i32],
    row_k_ones: bool,
    erasures: &[i32],
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    validate_payload_matrices::<G>(k, m, matrix, data, coding, size)?;
    let erased = erasures_to_erased(k, m, erasures)?;

    let mut edd = 0usize;
    let mut lastdrive = k;
    for i in 0..k {
        if erased[i] != 0 {
            edd += 1;
            lastdrive = i;
        }
    }
    if !row_k_ones || erased[k] != 0 {
        lastdrive = k;
    }

    let (dm_ids, decoding_matrix) = if edd > 1 || (edd > 0 && (!row_k_ones || erased[k] != 0)) {
        let mut dm_ids = vec![0i32; k];
        let mut decoding_matrix = vec![0i32; k * k];
        make_decoding_matrix_impl::<G>(k, m, matrix, &erased, &mut decoding_matrix, &mut dm_ids)?;

        #[cfg(feature = "std")]
        if TypeId::of::<G>() == TypeId::of::<EngineW32>() {
            crate::fec::gf::table::warm_gf32_coeff_cache(&decoding_matrix);
        }

        (Some(dm_ids), Some(decoding_matrix))
    } else {
        (None, None)
    };

    // Phase 2 — parallel decode of erased data stripes
    if edd > 0
        && let (Some(dm_ids), Some(decoding_matrix)) = (&dm_ids, &decoding_matrix)
    {
        let erased_data_rows: Vec<usize> = (0..lastdrive).filter(|&i| erased[i] != 0).collect();

        if !erased_data_rows.is_empty() {
            let data_addrs: Vec<usize> = data
                .get_row_ptrs(&erased_data_rows)?
                .into_iter()
                .map(|p| p as usize)
                .collect();

            let mut src_ptrs: Vec<usize> = Vec::with_capacity(k);
            for i in 0..k {
                let id = dm_ids[i] as usize;
                let src = if id < k {
                    data.row(id)?
                } else {
                    coding.row(id - k)?
                };
                src_ptrs.push(src.as_ptr() as usize);
            }

            erased_data_rows
                .par_iter()
                .enumerate()
                .try_for_each(|(idx, &row)| {
                    let dest = unsafe {
                        core::slice::from_raw_parts_mut(data_addrs[idx] as *mut u8, size)
                    };
                    let matrix_row = &decoding_matrix[row * k..(row + 1) * k];
                    let mut init = false;
                    for i in 0..k {
                        let coeff = matrix_row[i];
                        if coeff == 0 {
                            continue;
                        }
                        let src =
                            unsafe { core::slice::from_raw_parts(src_ptrs[i] as *const u8, size) };
                        if coeff == 1 {
                            if !init {
                                dest[..size].copy_from_slice(&src[..size]);
                                init = true;
                            } else {
                                G::region_xor(&src[..size], dest);
                            }
                        } else {
                            G::region_mul_add(coeff_to_u32(coeff), &src[..size], dest, init)?;
                            init = true;
                        }
                    }
                    Ok(())
                })?;
        }
    }

    // Phase 3 — sequential leftover (single stripe, not worth parallelising)
    if edd > 0 && lastdrive < k && erased[lastdrive] != 0 {
        let tmpids: Vec<i32> = (0..k)
            .map(|i| {
                if i < lastdrive {
                    i as i32
                } else {
                    (i + 1) as i32
                }
            })
            .collect();
        matrix_dotprod_payloads::<G>(
            k,
            matrix,
            Some(&tmpids),
            lastdrive,
            true,
            data,
            coding,
            size,
        )?;
    }

    // Phase 4 — parallel decode of erased parity stripes
    let erased_parity_rows: Vec<usize> = (0..m).filter(|&i| erased[k + i] != 0).collect();

    if !erased_parity_rows.is_empty() {
        let coding_addrs: Vec<usize> = coding
            .get_row_ptrs(&erased_parity_rows)?
            .into_iter()
            .map(|p| p as usize)
            .collect();

        let mut data_ptrs: Vec<usize> = Vec::with_capacity(k);
        for i in 0..k {
            data_ptrs.push(data.row(i)?.as_ptr() as usize);
        }

        erased_parity_rows
            .par_iter()
            .enumerate()
            .try_for_each(|(idx, &i)| {
                let dest =
                    unsafe { core::slice::from_raw_parts_mut(coding_addrs[idx] as *mut u8, size) };
                let matrix_row = &matrix[i * k..(i + 1) * k];
                let mut init = false;
                for j in 0..k {
                    let coeff = matrix_row[j];
                    if coeff == 0 {
                        continue;
                    }
                    let src =
                        unsafe { core::slice::from_raw_parts(data_ptrs[j] as *const u8, size) };
                    if coeff == 1 {
                        if !init {
                            dest[..size].copy_from_slice(&src[..size]);
                            init = true;
                        } else {
                            G::region_xor(&src[..size], dest);
                        }
                    } else {
                        G::region_mul_add(coeff_to_u32(coeff), &src[..size], dest, init)?;
                        init = true;
                    }
                }
                Ok(())
            })?;
    }

    Ok(())
}

/// WordSize-dispatched entry for parallel decode (called from block_code.rs).
#[cfg(feature = "parallel")]
pub(crate) fn matrix_decode_payloads_parallel_dispatch(
    w: WordSize,
    k: usize,
    m: usize,
    matrix: &[i32],
    row_k_ones: bool,
    erasures: &[i32],
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    match w {
        WordSize::W8 => matrix_decode_payloads_parallel::<EngineW8>(
            k, m, matrix, row_k_ones, erasures, data, coding, size,
        ),
        WordSize::W32 => matrix_decode_payloads_parallel::<EngineW32>(
            k, m, matrix, row_k_ones, erasures, data, coding, size,
        ),
    }
}

/// Decode erased stripes listed in `erasures` (Jerasure list format, `-1` sentinel).
#[cfg(test)]
pub(crate) fn matrix_decode(
    w: WordSize,
    k: usize,
    m: usize,
    matrix: &[i32],
    row_k_ones: bool,
    erasures: &[i32],
    stripes: &mut [Vec<u8>],
    size: usize,
) -> FecResult<()> {
    match w {
        WordSize::W8 => {
            matrix_decode_impl::<EngineW8>(k, m, matrix, row_k_ones, erasures, stripes, size)
        }
        WordSize::W32 => {
            matrix_decode_impl::<EngineW32>(k, m, matrix, row_k_ones, erasures, stripes, size)
        }
    }
}

#[cfg(test)]
fn matrix_decode_impl<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &[i32],
    row_k_ones: bool,
    erasures: &[i32],
    stripes: &mut [Vec<u8>],
    size: usize,
) -> FecResult<()> {
    validate_stripes::<G>(k, m, matrix, stripes, size)?;
    let erased = erasures_to_erased(k, m, erasures)?;

    let mut edd = 0usize;
    let mut lastdrive = k;
    for i in 0..k {
        if erased[i] != 0 {
            edd += 1;
            lastdrive = i;
        }
    }

    if !row_k_ones || erased[k] != 0 {
        lastdrive = k;
    }

    let (dm_ids, decoding_matrix) = if edd > 1 || (edd > 0 && (!row_k_ones || erased[k] != 0)) {
        let mut dm_ids = vec![0i32; k];
        let mut decoding_matrix = vec![0i32; k * k];
        make_decoding_matrix_impl::<G>(k, m, matrix, &erased, &mut decoding_matrix, &mut dm_ids)?;
        (Some(dm_ids), Some(decoding_matrix))
    } else {
        (None, None)
    };

    if edd > 0
        && let (Some(dm_ids), Some(decoding_matrix)) = (&dm_ids, &decoding_matrix)
    {
        for i in 0..lastdrive {
            if erased[i] != 0 {
                matrix_dotprod::<G>(
                    k,
                    &decoding_matrix[i * k..(i + 1) * k],
                    Some(dm_ids),
                    i,
                    stripes,
                    size,
                )?;
            }
        }
    }

    if edd > 0 && lastdrive < k && erased[lastdrive] != 0 {
        let tmpids: Vec<i32> = (0..k)
            .map(|i| {
                if i < lastdrive {
                    i as i32
                } else {
                    (i + 1) as i32
                }
            })
            .collect();
        matrix_dotprod::<G>(k, matrix, Some(&tmpids), lastdrive, stripes, size)?;
    }

    for i in 0..m {
        if erased[k + i] != 0 {
            matrix_dotprod::<G>(k, &matrix[i * k..(i + 1) * k], None, k + i, stripes, size)?;
        }
    }

    Ok(())
}

/// Invert a `rows × rows` matrix in GF(2^w).
#[cfg(test)]
pub(crate) fn invert_matrix(
    w: WordSize,
    mat: &mut [i32],
    inv: &mut [i32],
    rows: usize,
) -> FecResult<()> {
    match w {
        WordSize::W8 => invert_matrix_impl::<EngineW8>(mat, inv, rows),
        WordSize::W32 => invert_matrix_impl::<EngineW32>(mat, inv, rows),
    }
}

fn invert_matrix_impl<G: GfEngine>(mat: &mut [i32], inv: &mut [i32], rows: usize) -> FecResult<()> {
    if mat.len() < rows * rows || inv.len() < rows * rows {
        return Err(FecError::InvalidMatrixDimensions);
    }

    for i in 0..rows {
        for j in 0..rows {
            inv[i * rows + j] = if i == j { 1 } else { 0 };
        }
    }

    for i in 0..rows {
        let row_start = rows * i;
        if mat[row_start + i] == 0 {
            let mut swapped = false;
            for j in (i + 1)..rows {
                if mat[rows * j + i] != 0 {
                    swap_rows(mat, inv, rows, i, j);
                    swapped = true;
                    break;
                }
            }
            if !swapped {
                return Err(FecError::SingularMatrix);
            }
        }

        let pivot = coeff_to_u32(mat[row_start + i]);
        if pivot != 1 {
            let inverse = G::div(1, pivot)?;
            for j in 0..rows {
                mat[row_start + j] = G::mul(coeff_to_u32(mat[row_start + j]), inverse) as i32;
                inv[row_start + j] = G::mul(coeff_to_u32(inv[row_start + j]), inverse) as i32;
            }
        }

        for j in (i + 1)..rows {
            let rs2 = rows * j;
            let factor = coeff_to_u32(mat[rs2 + i]);
            if factor == 0 {
                continue;
            }
            if factor == 1 {
                for x in 0..rows {
                    mat[rs2 + x] =
                        G::add(coeff_to_u32(mat[rs2 + x]), coeff_to_u32(mat[row_start + x])) as i32;
                    inv[rs2 + x] =
                        G::add(coeff_to_u32(inv[rs2 + x]), coeff_to_u32(inv[row_start + x])) as i32;
                }
            } else {
                for x in 0..rows {
                    mat[rs2 + x] = G::add(
                        coeff_to_u32(mat[rs2 + x]),
                        G::mul(factor, coeff_to_u32(mat[row_start + x])),
                    ) as i32;
                    inv[rs2 + x] = G::add(
                        coeff_to_u32(inv[rs2 + x]),
                        G::mul(factor, coeff_to_u32(inv[row_start + x])),
                    ) as i32;
                }
            }
        }
    }

    for i in (0..rows).rev() {
        let row_start = i * rows;
        for j in 0..i {
            let rs2 = j * rows;
            let factor = coeff_to_u32(mat[rs2 + i]);
            if factor == 0 {
                continue;
            }
            mat[rs2 + i] = 0;
            for col in 0..rows {
                inv[rs2 + col] = G::add(
                    coeff_to_u32(inv[rs2 + col]),
                    G::mul(factor, coeff_to_u32(inv[row_start + col])),
                ) as i32;
            }
        }
    }

    Ok(())
}

#[cfg(any(feature = "parallel", test))]
fn make_decoding_matrix_impl<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &[i32],
    erased: &[i32],
    decoding_matrix: &mut [i32],
    dm_ids: &mut [i32],
) -> FecResult<()> {
    let mut j = 0usize;
    for i in 0..(k + m) {
        if erased[i] == 0 {
            dm_ids[j] = i as i32;
            j += 1;
            if j == k {
                break;
            }
        }
    }
    if j != k {
        return Err(FecError::TooManyErasures);
    }

    let mut tmpmat = vec![0i32; k * k];
    for i in 0..k {
        if dm_ids[i] < k as i32 {
            for col in 0..k {
                tmpmat[i * k + col] = 0;
            }
            tmpmat[i * k + dm_ids[i] as usize] = 1;
        } else {
            let row = (dm_ids[i] as usize) - k;
            for col in 0..k {
                tmpmat[i * k + col] = matrix[row * k + col];
            }
        }
    }

    invert_matrix_impl::<G>(&mut tmpmat, decoding_matrix, k)
}

/// Jerasure list → per-device erased bitmap.
pub(crate) fn erasures_to_erased(k: usize, m: usize, erasures: &[i32]) -> FecResult<Vec<i32>> {
    let total = k + m;
    let mut erased = vec![0i32; total];
    let mut survivors = total;

    for &idx in erasures {
        if idx == -1 {
            break;
        }
        let idx = idx as usize;
        if idx >= total {
            return Err(FecError::InvalidErasureIndex { index: idx as i32 });
        }
        if erased[idx] == 0 {
            erased[idx] = 1;
            survivors -= 1;
            if survivors < k {
                return Err(FecError::TooManyErasures);
            }
        }
    }
    Ok(erased)
}

fn matrix_dotprod_payloads<G: GfEngine>(
    k: usize,
    matrix_row: &[i32],
    src_ids: Option<&[i32]>,
    dest_local: usize,
    dest_is_data: bool,
    data: &mut PayloadMatrix,
    coding: &mut PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    // Write directly into the destination stripe (Jerasure `dptr`). Survivor stripes
    // never alias the erased output row in valid encode/decode schedules.
    let dest_ptr: *mut u8 = if dest_is_data {
        data.row_mut(dest_local)?.as_mut_ptr()
    } else {
        coding.row_mut(dest_local)?.as_mut_ptr()
    };
    // SAFETY: `dest_ptr` is from `row_mut` on a stripe row of length `size`.
    let dest = unsafe { core::slice::from_raw_parts_mut(dest_ptr, size) };
    let mut init = false;

    for i in 0..k {
        let coeff = matrix_row[i];
        if coeff == 0 {
            continue;
        }
        let src = stripe_read_payloads(src_ids, i, k, data, coding)?;
        if coeff == 1 {
            if !init {
                dest.copy_from_slice(&src[..size]);
                init = true;
            } else {
                G::region_xor(&src[..size], dest);
            }
        } else {
            G::region_mul_add(coeff_to_u32(coeff), &src[..size], dest, init)?;
            init = true;
        }
    }
    Ok(())
}

fn stripe_read_payloads<'a>(
    src_ids: Option<&[i32]>,
    i: usize,
    k: usize,
    data: &'a PayloadMatrix,
    coding: &'a PayloadMatrix,
) -> FecResult<&'a [u8]> {
    let id = match src_ids {
        None => i,
        Some(ids) => ids[i] as usize,
    };
    if id < k {
        data.row(id)
    } else {
        coding.row(id - k)
    }
}

fn validate_payload_matrices_strided<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &Matrix,
    first_row: usize,
    data: &PayloadMatrix,
    coding: &PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    if k == 0 || m == 0 || first_row + m > matrix.rows() || matrix.cols() < k {
        return Err(FecError::InvalidMatrixDimensions);
    }
    if data.rows() < k || coding.rows() < m {
        return Err(FecError::InvalidStripeIndex { index: 0 });
    }
    for row in 0..k {
        if data.row(row)?.len() < size {
            return Err(FecError::BufferTooSmall {
                stripe: row as i32,
                needed: size,
                actual: data.row(row)?.len(),
            });
        }
    }
    for row in 0..m {
        if coding.row(row)?.len() < size {
            return Err(FecError::BufferTooSmall {
                stripe: (k + row) as i32,
                needed: size,
                actual: coding.row(row)?.len(),
            });
        }
    }
    let _ = matrix.row_coefficients(first_row)?;
    G::validate_region_size(size)
}

fn validate_payload_matrices<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &[i32],
    data: &PayloadMatrix,
    coding: &PayloadMatrix,
    size: usize,
) -> FecResult<()> {
    if k == 0 || m == 0 || matrix.len() < m * k {
        return Err(FecError::InvalidMatrixDimensions);
    }
    if data.rows() < k || coding.rows() < m {
        return Err(FecError::InvalidStripeIndex { index: 0 });
    }
    for row in 0..k {
        if data.row(row)?.len() < size {
            return Err(FecError::BufferTooSmall {
                stripe: row as i32,
                needed: size,
                actual: data.row(row)?.len(),
            });
        }
    }
    for row in 0..m {
        if coding.row(row)?.len() < size {
            return Err(FecError::BufferTooSmall {
                stripe: (k + row) as i32,
                needed: size,
                actual: coding.row(row)?.len(),
            });
        }
    }
    G::validate_region_size(size)
}

#[cfg(test)]
fn matrix_dotprod<G: GfEngine>(
    k: usize,
    matrix_row: &[i32],
    src_ids: Option<&[i32]>,
    dest_id: usize,
    stripes: &mut [Vec<u8>],
    size: usize,
) -> FecResult<()> {
    if dest_id >= stripes.len() {
        return Err(FecError::InvalidStripeIndex {
            index: dest_id as i32,
        });
    }
    let (left, right) = stripes.split_at_mut(dest_id);
    let (dest_vec, right_rest) = right
        .split_first_mut()
        .ok_or(FecError::InvalidStripeIndex {
            index: dest_id as i32,
        })?;
    let dest = &mut dest_vec[..size];
    let mut init = false;

    for i in 0..k {
        if matrix_row[i] == 1 {
            let src = stripe_read_split(src_ids, i, dest_id, left, right_rest)?;
            if !init {
                dest.copy_from_slice(&src[..size]);
                init = true;
            } else {
                G::region_xor(&src[..size], dest);
            }
        }
    }

    for i in 0..k {
        let coeff = matrix_row[i];
        if coeff != 0 && coeff != 1 {
            let src = stripe_read_split(src_ids, i, dest_id, left, right_rest)?;
            G::region_mul_add(coeff_to_u32(coeff), &src[..size], dest, init)?;
            init = true;
        }
    }
    Ok(())
}

#[cfg(test)]
fn stripe_read_split<'a>(
    src_ids: Option<&[i32]>,
    i: usize,
    dest_id: usize,
    left: &'a [Vec<u8>],
    right: &'a [Vec<u8>],
) -> FecResult<&'a [u8]> {
    let id = match src_ids {
        None => i,
        Some(ids) => ids[i] as usize,
    };
    if id == dest_id {
        return Err(FecError::InvalidStripeIndex { index: id as i32 });
    }
    if id < dest_id {
        left.get(id)
            .map(|s| s.as_slice())
            .ok_or(FecError::InvalidStripeIndex { index: id as i32 })
    } else {
        right
            .get(id - dest_id - 1)
            .map(|s| s.as_slice())
            .ok_or(FecError::InvalidStripeIndex { index: id as i32 })
    }
}

fn swap_rows(mat: &mut [i32], inv: &mut [i32], cols: usize, a: usize, b: usize) {
    for k in 0..cols {
        mat.swap(a * cols + k, b * cols + k);
        inv.swap(a * cols + k, b * cols + k);
    }
}

fn coeff_to_u32(c: i32) -> u32 {
    // Jerasure stores GF elements in signed `int`; values ≥ 2^31 appear negative.
    c as u32
}

#[cfg(test)]
fn validate_stripes<G: GfEngine>(
    k: usize,
    m: usize,
    matrix: &[i32],
    stripes: &[Vec<u8>],
    size: usize,
) -> FecResult<()> {
    if k == 0 || m == 0 {
        return Err(FecError::InvalidMatrixDimensions);
    }
    if matrix.len() < m * k {
        return Err(FecError::InvalidMatrixDimensions);
    }
    if stripes.len() < k + m {
        return Err(FecError::InvalidStripeIndex { index: 0 });
    }
    for (idx, stripe) in stripes.iter().enumerate().take(k + m) {
        if stripe.len() < size {
            return Err(FecError::BufferTooSmall {
                stripe: idx as i32,
                needed: size,
                actual: stripe.len(),
            });
        }
    }
    G::validate_region_size(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W32: WordSize = WordSize::W32;
    const W8: WordSize = WordSize::W8;

    fn zero_stripes(k: usize, m: usize, size: usize) -> Vec<Vec<u8>> {
        alloc::vec![vec![0u8; size]; k + m]
    }

    #[test]
    fn encode_decode_roundtrip_single_erasures() {
        let k = 4usize;
        let m = 2usize;
        let size = 16usize;
        let matrix = super::super::cauchy::original_coding_matrix(k, m, W32).unwrap();

        let mut stripes = zero_stripes(k, m, size);
        for (i, stripe) in stripes.iter_mut().take(k).enumerate() {
            for (j, byte) in stripe.iter_mut().enumerate() {
                *byte = ((i + 1) * (j + 1)) as u8;
            }
        }

        matrix_encode(W32, k, m, &matrix, &mut stripes, size).unwrap();

        stripes[2].fill(0);
        matrix_decode(W32, k, m, &matrix, false, &[2, -1], &mut stripes, size).unwrap();
        let expected: Vec<u8> = (0..size).map(|j| (3 * (j + 1)) as u8).collect();
        assert_eq!(&stripes[2][..size], expected.as_slice());

        stripes[k].fill(0);
        matrix_decode(
            W32,
            k,
            m,
            &matrix,
            false,
            &[k as i32, -1],
            &mut stripes,
            size,
        )
        .unwrap();
        assert_ne!(&stripes[k][..size], &[0u8; 16][..]);
    }

    #[test]
    fn encode_decode_two_data_erasures() {
        let k = 4usize;
        let m = 2usize;
        let size = 32usize;
        let matrix = super::super::cauchy::original_coding_matrix(k, m, W32).unwrap();
        let mut stripes = zero_stripes(k, m, size);
        for (i, stripe) in stripes.iter_mut().take(k).enumerate() {
            for (j, byte) in stripe.iter_mut().enumerate() {
                *byte = ((i + 1) * (j + 1)) as u8;
            }
        }
        matrix_encode(W32, k, m, &matrix, &mut stripes, size).unwrap();
        let saved0 = stripes[0].clone();
        let saved1 = stripes[1].clone();
        stripes[0].fill(0);
        stripes[1].fill(0);
        matrix_decode(W32, k, m, &matrix, false, &[0, 1, -1], &mut stripes, size).unwrap();
        assert_eq!(stripes[0], saved0);
        assert_eq!(stripes[1], saved1);
    }

    #[test]
    fn invert_identity() {
        let mut mat = vec![1, 0, 0, 1];
        let mut inv = vec![0; 4];
        invert_matrix(W32, &mut mat, &mut inv, 2).unwrap();
        assert_eq!(inv, vec![1, 0, 0, 1]);
    }

    #[test]
    fn encode_decode_w8_header_matrix() {
        let k = 1usize;
        let m = 1usize;
        let size = 64usize;
        let matrix = super::super::cauchy::original_coding_matrix(k, m, W8).unwrap();
        let mut stripes = zero_stripes(k, m, size);
        stripes[0][7] = 4;
        matrix_encode(W8, k, m, &matrix, &mut stripes, size).unwrap();
        stripes[0].fill(0);
        matrix_decode(W8, k, m, &matrix, false, &[0, -1], &mut stripes, size).unwrap();
        assert_eq!(stripes[0][7], 4);
    }
}
