//! Internal FEC errors (not part of the public API yet).

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum FecError {
    #[error("matrix row and column capacity must be non-zero")]
    InvalidMatrixDimensions,

    #[error("matrix index out of bounds: row {row}, col {col} (size {rows}x{cols})")]
    MatrixOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    #[error("matrix resize {rows}x{cols} exceeds capacity {max_rows}x{max_cols}")]
    MatrixResizeTooLarge {
        rows: usize,
        cols: usize,
        max_rows: usize,
        max_cols: usize,
    },

    #[error("invalid matrix row {row} for coding window")]
    InvalidMatrixRow { row: u16 },

    #[error("invalid matrix column {col} for coding window")]
    InvalidMatrixColumn { col: u16 },

    #[error("Galois field division by zero")]
    GfDivisionByZero,

    #[error("Galois field element is not invertible")]
    GfNotInvertible,

    #[error("coding matrix is singular")]
    SingularMatrix,

    #[error("stripe buffer {stripe} too small: need {needed}, have {actual}")]
    BufferTooSmall {
        stripe: i32,
        needed: usize,
        actual: usize,
    },

    #[error("buffer length mismatch: expected {expected}, got {actual}")]
    BufferLengthMismatch { expected: usize, actual: usize },

    #[error("region length {len} must be a multiple of 4 for GF(2^32)")]
    UnalignedRegion { len: usize },

    #[error("invalid erasure index {index}")]
    InvalidErasureIndex { index: i32 },

    #[error("too many erasures to decode")]
    TooManyErasures,

    #[error("invalid stripe index {index}")]
    InvalidStripeIndex { index: i32 },

    #[error("packet size must be a multiple of 8 (sizeof u64)")]
    InvalidPacketSize,

    #[error("invalid timeslot update: expected {expected}, got {actual}")]
    InvalidTimeslot { expected: u16, actual: u16 },

    #[error(
        "generator matrix validation failed at row {row}, col {col}: zerod={zerod}, expected={should_zero}"
    )]
    MatrixValidationFailed {
        row: u16,
        col: u16,
        zerod: bool,
        should_zero: bool,
    },

    #[error("invalid StreamingCode / BlockCode configuration")]
    InvalidStreamingCodeConfig,

    #[error("frame {frame_num} is not recovered")]
    FrameNotRecovered { frame_num: u16 },

    #[error("timeslot not set")]
    TimeslotNotSet,

    #[error("frame_sizes length must equal num_frames in window")]
    InvalidFrameSizeWindow,
}

pub(crate) type FecResult<T> = Result<T, FecError>;
