//! Internal streaming FEC implementation (scalar GF(2^32) + Jerasure matrix ops).

mod block_code;
mod error;
mod gf;
mod header;
mod matrix;
mod payload_matrix;
mod puncture;
mod streaming;

#[cfg(any(test, feature = "bench"))]
pub(crate) mod bench_e2e;

#[cfg(any(test, feature = "bench"))]
pub(crate) mod bench_harness;

#[cfg(test)]
mod test_util;

pub(crate) use block_code::BlockCode;
pub(crate) use error::FecError;
pub(crate) use header::{MultiFecHeaderCode, header_matrix_config};
pub(crate) use matrix::Matrix;
pub(crate) use matrix::generator::{
    Position, set_cauchy_matrix, zero_by_frame, zero_streaming_mask,
};
pub(crate) use matrix::info::CodingMatrixInfo;
pub(crate) use matrix::jerasure::matrix_decode_payloads;
#[cfg(not(feature = "parallel"))]
pub(crate) use matrix::jerasure::matrix_encode_payloads_strided;
pub(crate) use puncture::num_frames_for_delay;
pub(crate) use streaming::StreamingCode;
pub(crate) use streaming::recovery::{StreamingCodeHelper, VuRatio};
