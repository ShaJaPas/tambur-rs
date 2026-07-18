//! Byte matrix for stripe payloads.

use alloc::vec::Vec;

use super::error::{FecError, FecResult};

/// Row-major matrix of stripe bytes (one row = one coding stripe).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PayloadMatrix {
    max_rows: usize,
    max_cols: usize,
    rows: usize,
    cols: usize,
    buf: Vec<u8>,
}

impl PayloadMatrix {
    pub(crate) fn new(max_rows: usize, max_cols: usize) -> FecResult<Self> {
        if max_rows == 0 || max_cols == 0 {
            return Err(FecError::InvalidMatrixDimensions);
        }
        Ok(Self {
            max_rows,
            max_cols,
            rows: max_rows,
            cols: max_cols,
            buf: vec![0u8; max_rows * max_cols],
        })
    }

    #[inline]
    fn row_offset(&self, row: usize) -> usize {
        row * self.max_cols
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

    pub(crate) fn get(&self, row: usize, col: usize) -> FecResult<u8> {
        self.bounds(row, col)?;
        Ok(self.buf[self.row_offset(row) + col])
    }

    pub(crate) fn set(&mut self, row: usize, col: usize, value: u8) -> FecResult<()> {
        self.bounds(row, col)?;
        let idx = self.row_offset(row) + col;
        self.buf[idx] = value;
        Ok(())
    }

    pub(crate) fn fill_row(&mut self, row: usize, value: u8) -> FecResult<()> {
        self.bounds_row(row)?;
        let start = self.row_offset(row);
        self.buf[start..start + self.cols].fill(value);
        Ok(())
    }

    pub(crate) fn fill_row_from_slice(&mut self, row: usize, vals: &[u8]) -> FecResult<()> {
        self.bounds_row(row)?;
        let start = self.row_offset(row);
        let row_slice = &mut self.buf[start..start + self.cols];
        let n = vals.len().min(self.cols);
        row_slice[..n].copy_from_slice(&vals[..n]);
        if n < self.cols {
            row_slice[n..].fill(0);
        }
        Ok(())
    }

    pub(crate) fn row(&self, row: usize) -> FecResult<&[u8]> {
        self.bounds_row(row)?;
        let start = self.row_offset(row);
        Ok(&self.buf[start..start + self.cols])
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> FecResult<&mut [u8]> {
        self.bounds_row(row)?;
        let start = self.row_offset(row);
        Ok(&mut self.buf[start..start + self.cols])
    }

    /// Return `*mut u8` pointers for non-contiguous rows (for parallel decode).
    /// SAFETY: rows must contain unique indices. The caller must not alias pointers.
    #[cfg(feature = "parallel")]
    pub(crate) fn get_row_ptrs(&mut self, rows: &[usize]) -> FecResult<Vec<*mut u8>> {
        let row_len = self.max_cols;
        let ptr = self.buf.as_mut_ptr();
        for &row in rows {
            if row >= self.rows {
                return Err(FecError::MatrixOutOfBounds {
                    row,
                    col: 0,
                    rows: self.rows,
                    cols: self.cols,
                });
            }
        }
        Ok(rows
            .iter()
            .map(|&row| {
                // SAFETY: `row` is bounds-checked above (`row < self.rows`), `row_len` is
                // `self.max_cols`, and `ptr` points to `self.buf` which has at least
                // `self.rows * self.max_cols` bytes allocated.
                unsafe { ptr.add(row * row_len) }
            })
            .collect())
    }

    /// Split rows `start..start+count` into disjoint `&mut [u8]` slices for parallel writes.
    #[cfg(feature = "parallel")]
    pub(crate) fn chunk_rows_mut(
        &mut self,
        start: usize,
        count: usize,
    ) -> FecResult<Vec<&mut [u8]>> {
        if start + count > self.rows {
            return Err(FecError::MatrixOutOfBounds {
                row: start + count,
                col: 0,
                rows: self.rows,
                cols: self.cols,
            });
        }
        let row_len = self.max_cols;
        let base = self.row_offset(start);
        let total = count * row_len;
        let buf = &mut self.buf[base..base + total];
        let ptr = buf.as_mut_ptr();
        let mut slices = Vec::with_capacity(count);
        for i in 0..count {
            let off = i * row_len;
            // SAFETY: each slice covers a disjoint row at offset `off`, length `row_len`.
            slices.push(unsafe { core::slice::from_raw_parts_mut(ptr.add(off), row_len) });
        }
        Ok(slices)
    }

    fn bounds(&self, row: usize, col: usize) -> FecResult<()> {
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

    fn bounds_row(&self, row: usize) -> FecResult<()> {
        if row >= self.rows {
            return Err(FecError::MatrixOutOfBounds {
                row,
                col: 0,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(())
    }
}
