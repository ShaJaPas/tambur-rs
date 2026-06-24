//! Encode/decode throughput benchmarks.
//!
//! `test_encode_time` / `test_decode_time` time only [`BlockCode::encode`] /
//! [`BlockCode::decode`] after window setup — setup is outside `clock()`. These benches
//! use Criterion `iter_batched` the same way.
//!
//! ```text
//! cargo bench --features bench --bench fec
//! ```

use core::hint::black_box;
use core::time::Duration;

use bytes::Bytes;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use tambur_rs::bench::{
    DECODE_BENCH_N_DATA, ENCODE_BENCH_TIMESLOT, N_DATA_PER_FRAME, decode_bench_timeslot,
    packet_size_for_w, prepare_decode_block, prepare_decode_block_large, prepare_encode_block,
    run_decode, run_encode,
};
use tambur_rs::{Config, Decoder, Encoder, Feedback, WordSize};

const W_VALUES: &[WordSize] = &[WordSize::W8, WordSize::W32];
const DELAY_VALUES: &[u16] = &[0, 1, 2, 3];
/// Frame sizes must be `<= u16::MAX` — [`Encoder::encode_payload`] tracks size as `u16`.
const SESSION_FRAME_SIZES: &[usize] = &[8 * 1024, 32 * 1024, 65535];

fn session_config(frame_size: usize) -> Config {
    Config::builder()
        .w(WordSize::W32)
        .packet_size(core::num::NonZeroU16::new(8).unwrap())
        .max_data_stripes(core::num::NonZeroU16::new(512).unwrap())
        // High feedback on max u16 frame needs ~128 parity stripes (256 data / 2).
        .max_fec_stripes(128)
        .max_frame_size(frame_size as u32)
        .feedback_interval(Duration::from_millis(0))
        .high_redundancy_byte(1)
        .build()
        .expect("bench config")
}

fn block_code_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_code/encode");
    group.sample_size(10);

    for &delay in DELAY_VALUES {
        for &w in W_VALUES {
            let packet_size = packet_size_for_w(w);
            let stripe_bytes = (u16::from(w) * packet_size) as u64;
            let stripes_per_call = N_DATA_PER_FRAME as u64;
            group.throughput(Throughput::Bytes(stripe_bytes * stripes_per_call));

            let id = BenchmarkId::new(format!("delay={delay}"), format!("w={}", u16::from(w)));
            group.bench_with_input(id, &(delay, w), |b, &(delay, w)| {
                let template = prepare_encode_block(delay, w, ENCODE_BENCH_TIMESLOT);
                b.iter_batched(
                    || template.clone(),
                    |mut block| {
                        run_encode(&mut block);
                        black_box(());
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

fn block_code_decode(c: &mut Criterion) {
    let delay = 3;
    let timeslot = decode_bench_timeslot(delay);
    let mut group = c.benchmark_group("block_code/decode");
    group.sample_size(10);

    for &w in W_VALUES {
        let stripe_bytes = (u16::from(w) * 8) as u64;
        // FullRecover drops two parity frames; recover up to `DECODE_BENCH_N_DATA` stripes each.
        let erased_stripes = (2 * DECODE_BENCH_N_DATA) as u64;
        group.throughput(Throughput::Bytes(stripe_bytes * erased_stripes));

        let id = BenchmarkId::from_parameter(format!("w={}", u16::from(w)));
        group.bench_with_input(id, &w, |b, &w| {
            let template = prepare_decode_block(delay, w, timeslot);
            b.iter_batched(
                || template.clone(),
                |mut block| black_box(run_decode(&mut block)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Decode bench with large stripes (≥4 KiB) and ≥12 erasures — triggers parallel path.
fn block_code_decode_large(c: &mut Criterion) {
    let delay = 3;
    let timeslot = decode_bench_timeslot(delay);
    let mut group = c.benchmark_group("block_code/decode_large");
    group.sample_size(10);

    // packet_size=160 → stripe_size = 32*160 = 5120 B (≥4096 threshold)
    // n_data=6 → 2*6 = 12 erasures (≥12 threshold)
    let packet_size = 160u16;
    let n_data = 6u16;
    let stripe_bytes = (u32::from(WordSize::W32) * u32::from(packet_size)) as u64;
    let erased_stripes = (2 * n_data) as u64;
    group.throughput(Throughput::Bytes(stripe_bytes * erased_stripes));

    let id = BenchmarkId::new(
        "parallel_trigger",
        format!("w=32_pktsz={packet_size}_ndata={n_data}"),
    );
    group.bench_with_input(id, &WordSize::W32, |b, &w| {
        let template = prepare_decode_block_large(delay, w, timeslot, packet_size, n_data);
        b.iter_batched(
            || template.clone(),
            |mut block| black_box(run_decode(&mut block)),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn session_encode_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("session/encode_frame");
    group.sample_size(10);

    for &frame_size in SESSION_FRAME_SIZES {
        group.throughput(Throughput::Bytes(frame_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(frame_size),
            &frame_size,
            |b, &frame_size| {
                let config = session_config(frame_size);
                let payload = Bytes::from(vec![0xABu8; frame_size]);
                b.iter_batched(
                    || {
                        let mut enc = Encoder::new(config.clone()).expect("encoder");
                        enc.apply_feedback(Feedback::High);
                        enc
                    },
                    |mut enc| {
                        black_box(enc.encode_payload(payload.clone()).expect("encode_payload"));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn session_decode_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("session/receive_and_recover");
    group.sample_size(10);

    for &frame_size in SESSION_FRAME_SIZES {
        group.throughput(Throughput::Bytes(frame_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(frame_size),
            &frame_size,
            |b, &frame_size| {
                let config = session_config(frame_size);
                let mut enc = Encoder::new(config.clone()).expect("encoder");
                enc.apply_feedback(Feedback::High);
                let pkts = enc
                    .encode_payload(Bytes::from(vec![0xCDu8; frame_size]))
                    .expect("encode");

                b.iter_batched(
                    || Decoder::new(config.clone()).expect("decoder"),
                    |mut dec| {
                        for pkt in pkts {
                            black_box(dec.receive_datagram(pkt.clone(), Duration::ZERO).status);
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    block_code_encode,
    block_code_decode,
    block_code_decode_large,
    session_encode_frame,
    session_decode_pipeline
);
criterion_main!(benches);
