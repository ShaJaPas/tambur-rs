//! End-to-end benchmark: encode → wire (`to_bytes`/`from_bytes`) → decode → feedback.
//!
//! **How to read results**
//!
//! - `e2e/one_frame/*` — **one** source frame per iteration. Criterion `time` ≈ CPU cost
//!   to process a single frame (not playback latency). `thrpt` = payload bytes / time.
//! - `e2e/session/*` — short stream (`8–15` main + `7` tail frames). `time` is total
//!   session work; divide by the frame count in the benchmark id for a rough per-frame average.
//!
//! Full run targets ~2–3 minutes (10 samples, shortened sessions, no 48 KiB + loss).
//!
//! ```text
//! cargo bench --features bench --bench e2e
//! ```

use core::hint::black_box;
use core::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use tambur_rs::bench::{E2eBenchParams, E2eBenchSession, e2e_bench_config};

const TAU: u8 = 3;
const LOSS: f32 = 0.05;

const SAMPLE_ONE_FRAME: usize = 10;
const SAMPLE_SESSION: usize = 10;

const FRAME_SIZES: &[usize] = &[4 * 1024, 16 * 1024, 48 * 1024];
/// 48 KiB + loss on a single frame is a multi-second CPU stress test; use session loss at 16 KiB.
const ONE_FRAME_LOSS_FRAME_SIZES: &[usize] = &[4 * 1024, 16 * 1024];
/// 48 KiB + 5% loss over many frames is prohibitively slow; session loss stops at 16 KiB.
const SESSION_LOSS_FRAME_SIZES: &[usize] = &[4 * 1024, 16 * 1024];

fn bench_one_frame(
    c: &mut Criterion,
    group_name: &str,
    frame_sizes: &[usize],
    loss_probability: f32,
) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_ONE_FRAME);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(2));

    for &frame_size in frame_sizes {
        let params = E2eBenchParams::single_frame(frame_size, loss_probability);
        group.throughput(Throughput::Bytes(frame_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{frame_size}B")),
            &params,
            |b, params| {
                let config = e2e_bench_config(TAU, frame_size as u32);
                let mut session = E2eBenchSession::new(config, *params, 0xAB);
                b.iter(|| black_box(session.run_single_frame()));
            },
        );
    }

    group.finish();
}

fn bench_session(
    c: &mut Criterion,
    group_name: &str,
    frame_sizes: &[usize],
    loss_probability: f32,
) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_SESSION);
    group.measurement_time(Duration::from_secs(12));
    group.warm_up_time(Duration::from_secs(2));

    for &frame_size in frame_sizes {
        let params = E2eBenchParams::short_session(frame_size, loss_probability);
        let n_frames = u64::from(params.total_frames());
        group.throughput(Throughput::Bytes(n_frames * frame_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{frame_size}B/{n_frames}frames")),
            &params,
            |b, params| {
                let config = e2e_bench_config(TAU, frame_size as u32);
                b.iter_batched(
                    || E2eBenchSession::new(config.clone(), *params, 0xAB),
                    |mut session| black_box(session.run()),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn e2e_one_frame_no_loss(c: &mut Criterion) {
    bench_one_frame(c, "e2e/one_frame/no_loss", FRAME_SIZES, 0.0);
}

fn e2e_one_frame_loss(c: &mut Criterion) {
    bench_one_frame(
        c,
        "e2e/one_frame/loss_5pct",
        ONE_FRAME_LOSS_FRAME_SIZES,
        LOSS,
    );
}

fn e2e_session_no_loss(c: &mut Criterion) {
    bench_session(c, "e2e/session/no_loss", FRAME_SIZES, 0.0);
}

fn e2e_session_loss(c: &mut Criterion) {
    bench_session(c, "e2e/session/loss_5pct", SESSION_LOSS_FRAME_SIZES, LOSS);
}

criterion_group!(
    benches,
    e2e_one_frame_no_loss,
    e2e_one_frame_loss,
    e2e_session_no_loss,
    e2e_session_loss,
);
criterion_main!(benches);
