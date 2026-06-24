//! Matrix algebra for Jerasure coding: the dense [`Matrix`], Cauchy generator
//! construction, GF(2^w) encode/decode (`jerasure`), and coding-window geometry.
//!
//! Port of Tambur `util/matrix.hh` (`Matrix<int>`).

pub(crate) mod cauchy;
pub(crate) mod generator;
pub(crate) mod info;
pub(crate) mod jerasure;

use alloc::vec::Vec;

use crate::fec::error::{FecError, FecResult};

/// Fixed-capacity matrix of GF coefficients as `i32` (Jerasure `int` matrix entries).
///
/// Holds the generator or punctured submatrix consumed by Jerasure. Coefficients are
/// field elements already reduced to `0..=(2^w - 1)`; Jerasure performs the actual
/// Galois arithmetic on payload buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Matrix {
    max_rows: usize,
    max_cols: usize,
    rows: usize,
    cols: usize,
    buf: Vec<i32>,
}

impl Matrix {
    pub(crate) fn new(max_rows: usize, max_cols: usize) -> FecResult<Self> {
        if max_rows == 0 || max_cols == 0 {
            return Err(FecError::InvalidMatrixDimensions);
        }
        Ok(Self {
            max_rows,
            max_cols,
            rows: max_rows,
            cols: max_cols,
            buf: alloc::vec![0; max_rows * max_cols],
        })
    }

    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub(crate) fn max_cols(&self) -> usize {
        self.max_cols
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn get(&self, row: usize, col: usize) -> FecResult<i32> {
        self.bounds_check(row, col)?;
        Ok(self.buf[row * self.max_cols + col])
    }

    /// Active coefficient storage (row-major, `max_cols` stride).
    #[cfg(feature = "std")]
    pub(crate) fn coefficients(&self) -> &[i32] {
        &self.buf[..self.rows * self.max_cols]
    }

    /// Row `row` coefficients (`cols` entries, stride `max_cols`).
    pub(crate) fn row_coefficients(&self, row: usize) -> FecResult<&[i32]> {
        if row >= self.rows {
            return Err(FecError::MatrixOutOfBounds {
                row,
                col: 0,
                rows: self.rows,
                cols: self.cols,
            });
        }
        let start = row * self.max_cols;
        Ok(&self.buf[start..start + self.cols])
    }

    pub(crate) fn set(&mut self, row: usize, col: usize, value: i32) -> FecResult<()> {
        self.bounds_check(row, col)?;
        self.buf[row * self.max_cols + col] = value;
        Ok(())
    }

    /// Shrink the active region; capacity stays at the original maximum.
    pub(crate) fn resize(&mut self, rows: usize, cols: usize) -> FecResult<()> {
        if rows > self.max_rows || cols > self.max_cols {
            return Err(FecError::MatrixResizeTooLarge {
                rows,
                cols,
                max_rows: self.max_rows,
                max_cols: self.max_cols,
            });
        }
        self.rows = rows;
        self.cols = cols;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fill(&mut self, value: i32) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let _ = self.set(row, col, value);
            }
        }
    }

    /// Copy `vals` into row `row`; missing entries become zero.
    #[cfg(test)]
    pub(crate) fn fill_row_from_slice(&mut self, row: usize, vals: &[i32]) -> FecResult<()> {
        for col in 0..self.cols {
            let value = vals.get(col).copied().unwrap_or(0);
            self.set(row, col, value)?;
        }
        Ok(())
    }

    /// Row-major coefficient slice for the **active** `rows × cols` region.
    #[cfg(test)]
    pub(crate) fn active_coefficients(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                out.push(self.buf[row * self.max_cols + col]);
            }
        }
        out
    }

    fn bounds_check(&self, row: usize, col: usize) -> FecResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(FecError::MatrixOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_get_set() {
        let mut m = Matrix::new(2, 3).unwrap();
        m.set(0, 1, 7).unwrap();
        m.set(1, 2, 42).unwrap();
        assert_eq!(m.get(0, 1).unwrap(), 7);
        assert_eq!(m.get(1, 2).unwrap(), 42);
    }

    #[test]
    fn resize_smaller() {
        let mut m = Matrix::new(4, 4).unwrap();
        m.fill(1);
        m.resize(2, 2).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.get(1, 1).unwrap(), 1);
    }

    #[test]
    fn fill_row_from_slice_pads_with_zero() {
        let mut m = Matrix::new(2, 4).unwrap();
        m.fill(9);
        m.fill_row_from_slice(0, &[1, 2]).unwrap();
        assert_eq!(m.get(0, 0).unwrap(), 1);
        assert_eq!(m.get(0, 1).unwrap(), 2);
        assert_eq!(m.get(0, 2).unwrap(), 0);
    }

    #[test]
    fn active_coefficients_row_major() {
        let mut m = Matrix::new(2, 2).unwrap();
        m.set(0, 0, 1).unwrap();
        m.set(0, 1, 2).unwrap();
        m.set(1, 0, 3).unwrap();
        m.set(1, 1, 4).unwrap();
        assert_eq!(m.active_coefficients(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn out_of_bounds_and_oversize_resize() {
        let m = Matrix::new(2, 2).unwrap();
        assert!(matches!(
            m.get(2, 0),
            Err(FecError::MatrixOutOfBounds { .. })
        ));

        let mut m = Matrix::new(2, 2).unwrap();
        assert!(matches!(
            m.resize(3, 2),
            Err(FecError::MatrixResizeTooLarge { .. })
        ));
    }
}
