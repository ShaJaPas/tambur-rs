//! Crate documentation is in README.md (also rendered on docs.rs).

#![doc = include_str!("../README.md")]

extern crate alloc;

mod codec;
mod config;
mod datagram;
mod error;
mod fec;
mod feedback;
mod frame;
mod loss;
mod packetization;
mod predictor;
mod util;
mod word_size;
pub use codec::{Decoder, DecoderEvent, Encoder, ReceiveOutcome, ReceiveStatus};
pub use config::{Config, Tau};
pub use datagram::{FecDatagram, FecDatagramHeader};
pub use error::{CodecError, ConfigError, DatagramError, Error, InternalError, Result};
pub use feedback::{Feedback, FeedbackCodec};
pub use frame::RecoveredFrame;
pub use loss::{LossMetrics, LossReport};
pub use predictor::{
    BandwidthPredictor, FeedbackManager, HighBandwidthPredictor, LowBandwidthPredictor,
    RecommendContext,
};
pub use word_size::WordSize;

/// Hidden helpers for Criterion benchmarks.
///
/// - Main realistic path: `cargo bench --features bench --bench e2e`
/// - Low-level FEC micro-benches: `cargo bench --features bench --bench fec`
#[doc(hidden)]
#[cfg(feature = "bench")]
pub mod bench {
    pub use crate::fec::bench_e2e::{
        E2eBenchParams, E2eBenchSession, E2eBenchStats, e2e_bench_config,
    };
    pub use crate::fec::bench_harness::{
        DECODE_BENCH_N_DATA, DECODE_BENCH_TIMESLOT_OFFSET, ENCODE_BENCH_TIMESLOT, N_DATA_PER_FRAME,
        N_PAR_PER_FRAME, decode_bench_timeslot, packet_size_for_w, prepare_decode_block,
        prepare_decode_block_large, prepare_encode_block, run_decode, run_encode,
    };
}
