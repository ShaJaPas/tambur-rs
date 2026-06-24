//! End-to-end integration tests.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use bytes::Bytes;
use core::num::NonZeroU16;
use core::time::Duration;
use tambur_rs::Tau;
use tambur_rs::{
    Config, Decoder, DecoderEvent, Encoder, FecDatagram, Feedback, FeedbackCodec, ReceiveOutcome,
    WordSize,
};

fn window_frames(tau: u8) -> u16 {
    u16::from(tau) * 2 + 1
}

extern crate alloc;

/// Inner trial count (except burst uses [`BURST_TRIALS`]).
const E2E_TRIALS: u16 = 20;
const BURST_TRIALS: u16 = 5;
const BURST_TAU: u8 = 3;

/// glibc-compatible `rand()` after `srand(0)`.
struct TestRng {
    state: u32,
}

impl TestRng {
    fn seeded() -> Self {
        Self { state: 0 }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.state / 65536) % 32768
    }

    fn next_bounded(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        self.next_u32() % bound
    }
}

enum E2eProfile {
    NoLoss,
    Loss,
    Burst,
}

fn e2e_config(profile: E2eProfile, tau: u8, size_factor: u16) -> Config {
    let tau = match profile {
        E2eProfile::Burst => BURST_TAU,
        _ => tau,
    };
    let builder = Config::builder()
        .tau(Tau::new(tau).unwrap())
        .w(WordSize::W32)
        .packet_size(NonZeroU16::new(8 * size_factor).unwrap())
        .feedback_interval(Duration::from_millis(0))
        .high_redundancy_byte(1);
    match profile {
        E2eProfile::NoLoss | E2eProfile::Burst => builder
            .max_data_stripes(NonZeroU16::new(100).unwrap())
            .max_fec_stripes(75)
            .max_frame_size(10_000),
        E2eProfile::Loss => builder
            .max_data_stripes(NonZeroU16::new(64).unwrap())
            .max_fec_stripes(32)
            .max_frame_size(12_500),
    }
    .build()
    .unwrap()
}

fn stripe_size(config: &Config) -> u16 {
    config.stripe_size().min(u32::from(u16::MAX)) as u16
}

fn random_payload(rng: &mut TestRng, size: usize) -> Bytes {
    let mut v = vec![0u8; size.max(1)];
    for b in &mut v {
        *b = rng.next_bounded(256) as u8;
    }
    Bytes::from(v)
}

fn wire_roundtrip(pkt: &FecDatagram) -> FecDatagram {
    FecDatagram::from_bytes(pkt.to_bytes().unwrap()).unwrap()
}

struct E2eClock(Duration);

impl E2eClock {
    const fn new() -> Self {
        Self(Duration::ZERO)
    }

    fn now(&self) -> Duration {
        self.0
    }
}

fn events_to_recovered(events: &[DecoderEvent]) -> BTreeMap<u64, (Bytes, bool)> {
    let mut out = BTreeMap::new();
    for event in events {
        if let DecoderEvent::FrameRecovered(rec) = event {
            out.insert(rec.frame_num, (rec.payload.clone(), rec.direct_reception));
        }
    }
    out
}

fn receive_wire(dec: &mut Decoder, clock: &E2eClock, pkt: &FecDatagram) -> ReceiveOutcome {
    dec.receive_datagram(wire_roundtrip(pkt), clock.now())
}

fn poll_recovered(dec: &mut Decoder, clock: &E2eClock) -> BTreeMap<u64, (Bytes, bool)> {
    events_to_recovered(&dec.poll(clock.now()))
}

/// Decoder plus monotonic clock for e2e tests.
struct E2eDecoder {
    dec: Decoder,
    clock: E2eClock,
}

impl E2eDecoder {
    fn new(config: Config) -> Self {
        Self {
            dec: Decoder::new(config).unwrap(),
            clock: E2eClock::new(),
        }
    }

    fn receive_wire(&mut self, pkt: &FecDatagram) -> ReceiveOutcome {
        receive_wire(&mut self.dec, &self.clock, pkt)
    }

    fn poll_recovered(&mut self) -> BTreeMap<u64, (Bytes, bool)> {
        poll_recovered(&mut self.dec, &self.clock)
    }
}

impl core::ops::Deref for E2eDecoder {
    type Target = Decoder;

    fn deref(&self) -> &Decoder {
        &self.dec
    }
}

impl core::ops::DerefMut for E2eDecoder {
    fn deref_mut(&mut self) -> &mut Decoder {
        &mut self.dec
    }
}

fn setup_sender(config: &Config) -> Encoder {
    let mut enc = Encoder::new(config.clone()).unwrap();
    enc.apply_feedback(Feedback::High);
    enc
}

fn assert_payload_eq(recovered: &[u8], expected: &[u8]) {
    assert_eq!(recovered.len(), expected.len());
    for (pos, (&a, &b)) in recovered.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, b, "payload mismatch at byte {pos}");
    }
}

fn get_missing(num_pkts: usize, num_missing: usize, rng: &mut TestRng) -> Vec<bool> {
    let mut is_missing = vec![false; num_pkts];
    let mut dropped = 0usize;
    while dropped < num_missing {
        let index = rng.next_bounded(num_pkts as u32) as usize;
        if !is_missing[index] {
            is_missing[index] = true;
            dropped += 1;
        }
    }
    is_missing
}

fn highest_received_pos(missing: &[bool]) -> u16 {
    missing
        .iter()
        .enumerate()
        .filter(|&(_, m)| !m)
        .map(|(i, _)| i as u16)
        .next_back()
        .unwrap_or(0)
}

fn frame_size_no_loss(rng: &mut TestRng, frame_num: u16, stripe: u16, trial: u16) -> usize {
    let base = 3000 + 200 * u32::from(frame_num) + rng.next_bounded(2000);
    let mut size = base as usize;
    if trial.is_multiple_of(2) {
        size = (size / stripe as usize) * stripe as usize;
    }
    size.max(1)
}

fn frame_size_loss(rng: &mut TestRng, stripe: u16, trial: u16) -> usize {
    let mut size = (3000 + rng.next_bounded(2000)) as usize;
    if trial.is_multiple_of(2) {
        size = (size / stripe as usize) * stripe as usize;
    }
    size.max(1)
}

fn insert_recovered_frames(set: &mut BTreeSet<u64>, events: &[DecoderEvent]) {
    for event in events {
        if let DecoderEvent::FrameRecovered(rec) = event {
            set.insert(rec.frame_num);
        }
    }
}

fn merge_recovered(dst: &mut BTreeMap<u64, bool>, src: BTreeMap<u64, (Bytes, bool)>) {
    for (n, (_, direct)) in src {
        dst.insert(n, direct);
    }
}

fn session_key(frame_num: u16) -> u64 {
    u64::from(frame_num)
}

fn check_window_recovered(
    recovered: &BTreeMap<u64, (Bytes, bool)>,
    payloads: &[Bytes],
    start: u16,
    end: u16,
) {
    for check_frame in start..=end {
        let key = session_key(check_frame);
        let (got, _) = recovered
            .get(&key)
            .unwrap_or_else(|| panic!("frame {check_frame} not recovered"));
        assert_payload_eq(got, &payloads[check_frame as usize]);
    }
}

/// No-loss encode-decode roundtrip.
#[test]
fn e2e_simple_encode_decode_no_loss() {
    let mut rng = TestRng::seeded();
    for trial in 0..E2E_TRIALS {
        for tau in 0..4u8 {
            for size_factor in 1..2u16 {
                let config = e2e_config(E2eProfile::NoLoss, tau, size_factor);
                let stripe = stripe_size(&config);
                let mut enc = setup_sender(&config);
                let mut dec = E2eDecoder::new(config.clone());

                let num_main = 5 * u16::from(tau + 1);
                let mut payloads: Vec<Bytes> = Vec::new();
                let mut recovered: BTreeMap<u64, (Bytes, bool)> = BTreeMap::new();

                for frame_num in 0..num_main {
                    let size = frame_size_no_loss(&mut rng, frame_num, stripe, trial);
                    let payload = random_payload(&mut rng, size);
                    payloads.push(payload.clone());

                    let pkts = enc.encode_payload(payload.clone()).unwrap();

                    for pkt in pkts {
                        if pkt.pos_in_frame > 0 && pkt.is_parity {
                            assert!(
                                recovered.contains_key(&session_key(frame_num)),
                                "frame {frame_num} recovered before parity pos {}",
                                pkt.pos_in_frame
                            );
                            assert_payload_eq(
                                &recovered[&session_key(frame_num)].0,
                                &payloads[frame_num as usize],
                            );
                        }
                        let outcome = dec.receive_wire(pkt);
                        recovered.extend(events_to_recovered(&outcome.events));
                    }
                    recovered.extend(dec.poll_recovered());
                }

                let delay = window_frames(tau);
                for j in 0..delay {
                    let size = frame_size_no_loss(&mut rng, j, stripe, trial);
                    let payload = random_payload(&mut rng, size);
                    payloads.push(payload.clone());
                    let pkts = enc.encode_payload(payload.clone()).unwrap();
                    for pkt in pkts {
                        let outcome = dec.receive_wire(pkt);
                        recovered.extend(events_to_recovered(&outcome.events));
                    }
                    recovered.extend(dec.poll_recovered());
                }

                assert_eq!(recovered.len(), (num_main + delay) as usize);
                for frame_num in 0..num_main + delay {
                    let (_, direct) = recovered.get(&session_key(frame_num)).expect("recovered");
                    assert!(direct, "frame {frame_num} direct");
                    assert_payload_eq(
                        &recovered[&session_key(frame_num)].0,
                        &payloads[frame_num as usize],
                    );
                }
            }
        }
    }
}

/// Single-frame loss test.
#[test]
fn e2e_loss_single_frame() {
    let mut rng = TestRng::seeded();
    for size_factor in 1..3u16 {
        for trial in 0..E2E_TRIALS {
            for tau in 0..4u8 {
                let config = e2e_config(E2eProfile::Loss, tau, size_factor);
                let stripe = stripe_size(&config);
                let mut enc = setup_sender(&config);
                let mut dec = E2eDecoder::new(config.clone());

                let num_main = 7 * u16::from(tau + 1) + 1;
                let mut payloads = Vec::new();
                let mut data_lost = Vec::new();
                let mut recovered: BTreeMap<u64, bool> = BTreeMap::new();
                let mut recovered_frames: BTreeMap<u64, (Bytes, bool)> = BTreeMap::new();

                for frame_num in 0..num_main {
                    let size = frame_size_loss(&mut rng, stripe, trial);
                    let payload = random_payload(&mut rng, size);
                    payloads.push(payload.clone());

                    let pkts = enc.encode_payload(payload.clone()).unwrap();

                    let missing_pkt = rng.next_bounded(pkts.len() as u32) as u16;
                    let highest_pos =
                        pkts.len() as u16 - 1 - u16::from(missing_pkt == pkts.len() as u16);
                    data_lost.push(!pkts[missing_pkt as usize].is_parity);

                    for pkt in pkts {
                        if pkt.pos_in_frame == missing_pkt && frame_num > 2 * u16::from(tau) + 1 {
                            continue;
                        }
                        let outcome = dec.receive_wire(pkt);
                        let frame_events = events_to_recovered(&outcome.events);
                        merge_recovered(&mut recovered, frame_events.clone());
                        recovered_frames.extend(frame_events);

                        if pkt.pos_in_frame == highest_pos && frame_num > 2 * u16::from(tau) + 1 {
                            let start = frame_num.saturating_sub(u16::from(tau));
                            check_window_recovered(&recovered_frames, &payloads, start, frame_num);
                        }
                    }
                    let polled = dec.poll_recovered();
                    merge_recovered(&mut recovered, polled.clone());
                    recovered_frames.extend(polled);
                }

                let delay = window_frames(tau);
                for j in 0..delay {
                    let mut size = (3000 + 200 * u32::from(j)) as usize;
                    if trial % 2 == 0 {
                        size = (size / stripe as usize) * stripe as usize;
                    }
                    let payload = random_payload(&mut rng, size.max(1));
                    payloads.push(payload);
                    let pkts = enc
                        .encode_payload(payloads.last().unwrap().clone())
                        .unwrap();
                    for pkt in pkts {
                        let outcome = dec.receive_wire(pkt);
                        let frame_events = events_to_recovered(&outcome.events);
                        merge_recovered(&mut recovered, frame_events.clone());
                        recovered_frames.extend(frame_events);
                    }
                    let polled = dec.poll_recovered();
                    merge_recovered(&mut recovered, polled.clone());
                    recovered_frames.extend(polled);
                }

                for (frame_pos, &lost_data) in data_lost.iter().enumerate() {
                    let frame_num = frame_pos as u16;
                    let direct = recovered
                        .get(&session_key(frame_num))
                        .copied()
                        .unwrap_or(false);
                    let expect_direct = frame_num <= 2 * u16::from(tau) + 1 || !lost_data;
                    assert_eq!(direct, expect_direct, "frame {frame_num}");
                    assert!(
                        recovered.contains_key(&session_key(frame_num)),
                        "frame {frame_num}"
                    );
                }
            }
        }
    }
}

/// Multi-loss single-frame test.
#[test]
fn e2e_multi_loss_single_frame() {
    let mut rng = TestRng::seeded();
    for size_factor in 1..3u16 {
        for trial in 0..E2E_TRIALS {
            for tau in 0..4u8 {
                let config = e2e_config(E2eProfile::Loss, tau, size_factor);
                let stripe = stripe_size(&config);
                let mut enc = setup_sender(&config);
                let mut dec = E2eDecoder::new(config.clone());

                let num_main = 7 * u16::from(tau + 1) + 1;
                let mut payloads = Vec::new();
                let mut data_lost = Vec::new();
                let mut recovered: BTreeMap<u64, bool> = BTreeMap::new();
                let mut recovered_frames: BTreeMap<u64, (Bytes, bool)> = BTreeMap::new();

                for frame_num in 0..num_main {
                    let size = frame_size_loss(&mut rng, stripe, trial);
                    let payload = random_payload(&mut rng, size);
                    payloads.push(payload.clone());

                    let pkts = enc.encode_payload(payload.clone()).unwrap();

                    let num_parity = pkts.iter().filter(|p| p.is_parity).count() as u32;
                    let stripes_per_pkt =
                        (pkts[0].payload.len() as u32 / u32::from(stripe.max(1))).max(1);
                    let num_data_stripes = size as u32 / u32::from(stripe)
                        + u32::from(!(size as u32).is_multiple_of(u32::from(stripe)));

                    let num_missing = if trial % 2 == 0 {
                        rng.next_bounded(num_parity + 1)
                    } else {
                        (num_data_stripes / 2).saturating_sub(1) / stripes_per_pkt
                    } as usize;

                    let missing = get_missing(pkts.len(), num_missing, &mut rng);
                    let highest_pos = highest_received_pos(&missing);
                    let any_data_missing = missing
                        .iter()
                        .zip(pkts.iter())
                        .any(|(&m, p)| m && !p.is_parity);
                    data_lost.push(any_data_missing);

                    for (i, pkt) in pkts.iter().enumerate() {
                        if missing[i] && frame_num > 2 * u16::from(tau) + 1 {
                            continue;
                        }
                        let outcome = dec.receive_wire(pkt);
                        let frame_events = events_to_recovered(&outcome.events);
                        merge_recovered(&mut recovered, frame_events.clone());
                        recovered_frames.extend(frame_events);

                        if pkt.pos_in_frame == highest_pos && frame_num > 2 * u16::from(tau) + 1 {
                            let start = frame_num.saturating_sub(u16::from(tau));
                            check_window_recovered(&recovered_frames, &payloads, start, frame_num);
                        }
                    }
                    let polled = dec.poll_recovered();
                    merge_recovered(&mut recovered, polled.clone());
                    recovered_frames.extend(polled);
                }

                let delay = window_frames(tau);
                for j in 0..delay {
                    let mut size = (3000 + 200 * u32::from(j)) as usize;
                    if trial % 2 == 0 {
                        size = (size / stripe as usize) * stripe as usize;
                    }
                    let payload = random_payload(&mut rng, size.max(1));
                    payloads.push(payload.clone());
                    let pkts = enc.encode_payload(payload.clone()).unwrap();
                    for pkt in pkts {
                        let outcome = dec.receive_wire(pkt);
                        let frame_events = events_to_recovered(&outcome.events);
                        merge_recovered(&mut recovered, frame_events.clone());
                        recovered_frames.extend(frame_events);
                    }
                    let polled = dec.poll_recovered();
                    merge_recovered(&mut recovered, polled.clone());
                    recovered_frames.extend(polled);
                }

                for (frame_pos, &lost_data) in data_lost.iter().enumerate() {
                    let frame_num = frame_pos as u16;
                    let direct = recovered
                        .get(&session_key(frame_num))
                        .copied()
                        .unwrap_or(false);
                    let expect_direct = frame_num <= 2 * u16::from(tau) + 1 || !lost_data;
                    assert_eq!(direct, expect_direct, "frame {frame_num}");
                    assert!(recovered.contains_key(&session_key(frame_num)));
                }
            }
        }
    }
}

/// Multi-loss non-recoverable frame test.
#[test]
fn e2e_multi_loss_non_recoverable() {
    let mut rng = TestRng::seeded();
    for size_factor in 1..3u16 {
        for trial in 0..E2E_TRIALS {
            for tau in 0..4u8 {
                let config = e2e_config(E2eProfile::Loss, tau, size_factor);
                let stripe = stripe_size(&config);
                let mut enc = setup_sender(&config);
                let mut dec = E2eDecoder::new(config.clone());

                let num_main = 7 * u16::from(tau + 1) + 1;
                let mut recovered: BTreeSet<u64> = BTreeSet::new();

                for frame_num in 0..num_main {
                    let size = frame_size_loss(&mut rng, stripe, trial);
                    let payload = random_payload(&mut rng, size);
                    let pkts = enc.encode_payload(payload.clone()).unwrap();

                    let num_parity = pkts.iter().filter(|p| p.is_parity).count() as u16;
                    let stripes_per_pkt = (pkts[0].payload.len() as u16 / stripe.max(1)).max(1);
                    let num_data_stripes =
                        size as u16 / stripe + u16::from(!(size as u16).is_multiple_of(stripe));
                    let mut num_missing = if trial % 2 == 0 {
                        num_parity
                            + 1
                            + rng
                                .next_bounded(pkts.len().saturating_sub(num_parity as usize) as u32)
                                as u16
                    } else {
                        1 + (num_data_stripes / 2 + 1) / stripes_per_pkt
                    };
                    num_missing = num_missing.max(num_parity + 1);
                    num_missing = num_missing.min(pkts.len() as u16 - 1);

                    let missing = get_missing(pkts.len(), num_missing as usize, &mut rng);
                    let highest_pos = highest_received_pos(&missing);

                    for (i, pkt) in pkts.iter().enumerate() {
                        if missing[i] {
                            continue;
                        }
                        let outcome = dec.receive_wire(pkt);
                        insert_recovered_frames(&mut recovered, &outcome.events);
                        if pkt.pos_in_frame == highest_pos {
                            let start = frame_num.saturating_sub(u16::from(tau));
                            for check_frame in start..=frame_num {
                                assert!(
                                    !recovered.contains(&session_key(check_frame)),
                                    "frame {check_frame} unexpectedly recovered"
                                );
                            }
                        }
                    }
                    for n in dec.poll_recovered().keys() {
                        recovered.insert(*n);
                    }
                }

                let delay = window_frames(tau);
                for j in 0..delay {
                    let mut size = (3000 + 200 * u32::from(j)) as usize;
                    if trial % 2 == 0 {
                        size = (size / stripe as usize) * stripe as usize;
                    }
                    let payload = random_payload(&mut rng, size.max(1));
                    let pkts = enc.encode_payload(payload.clone()).unwrap();
                    for pkt in pkts {
                        if !pkt.is_parity {
                            let outcome = dec.receive_wire(pkt);
                            insert_recovered_frames(&mut recovered, &outcome.events);
                        }
                    }
                    for n in dec.poll_recovered().keys() {
                        recovered.insert(*n);
                    }
                }

                for frame_num in 0..num_main {
                    assert!(
                        !recovered.contains(&session_key(frame_num)),
                        "frame {frame_num} should stay lost"
                    );
                }
            }
        }
    }
}

/// Burst loss frame test.
#[test]
fn e2e_burst_loss_frame() {
    let mut rng = TestRng::seeded();
    let tau = BURST_TAU;
    for size_factor in 1..3u16 {
        for trial in 0..BURST_TRIALS {
            for burst_start in 4..18u16 {
                let config = e2e_config(E2eProfile::Burst, BURST_TAU, size_factor);
                let stripe = stripe_size(&config);
                let mut enc = setup_sender(&config);
                let mut dec = E2eDecoder::new(config.clone());

                let last_frame = burst_start + u16::from(tau) + 1;
                let mut payloads = Vec::new();
                let mut recovered: BTreeMap<u64, bool> = BTreeMap::new();
                let mut recovered_frames: BTreeMap<u64, (Bytes, bool)> = BTreeMap::new();

                for frame_num in 0..=last_frame {
                    let mut size = 2 * stripe as usize + rng.next_bounded(1000) as usize;
                    if frame_num > burst_start + 1 {
                        size = 2 * size + 6 * stripe as usize;
                    }
                    if trial % 2 == 0 {
                        size = (size / stripe as usize) * stripe as usize;
                    }
                    size = size.max(stripe as usize);

                    let payload = random_payload(&mut rng, size);
                    payloads.push(payload.clone());

                    let pkts = enc.encode_payload(payload.clone()).unwrap();

                    for pkt in pkts {
                        if frame_num == burst_start || frame_num == burst_start + 1 {
                            continue;
                        }
                        let outcome = dec.receive_wire(pkt);
                        let frame_events = events_to_recovered(&outcome.events);
                        merge_recovered(&mut recovered, frame_events.clone());
                        recovered_frames.extend(frame_events);

                        if pkt.pos_in_frame + 1 == pkts.len() as u16
                            && frame_num == burst_start + u16::from(tau) + 1
                        {
                            for rec_frame in burst_start..burst_start + 2 {
                                let (got, _) = recovered_frames
                                    .get(&session_key(rec_frame))
                                    .unwrap_or_else(|| {
                                        panic!("burst frame {rec_frame} not recovered")
                                    });
                                assert_payload_eq(got, &payloads[rec_frame as usize]);
                            }
                        }
                    }
                    let polled = dec.poll_recovered();
                    merge_recovered(&mut recovered, polled.clone());
                    recovered_frames.extend(polled);
                }

                let delay = window_frames(tau);
                for j in 0..delay {
                    let mut size = (3000 + 200 * u32::from(j)) as usize;
                    if trial % 2 == 0 {
                        size = (size / stripe as usize) * stripe as usize;
                    }
                    let payload = random_payload(&mut rng, size.max(1));
                    let pkts = enc.encode_payload(payload.clone()).unwrap();
                    for pkt in pkts {
                        let outcome = dec.receive_wire(pkt);
                        let frame_events = events_to_recovered(&outcome.events);
                        merge_recovered(&mut recovered, frame_events.clone());
                        recovered_frames.extend(frame_events);
                    }
                    let polled = dec.poll_recovered();
                    merge_recovered(&mut recovered, polled.clone());
                    recovered_frames.extend(polled);
                }

                for (&frame_num, &direct) in &recovered {
                    if frame_num == u64::from(burst_start)
                        || frame_num == u64::from(burst_start) + 1
                    {
                        assert!(!direct, "burst frame {frame_num} FEC-recovered");
                    } else if frame_num <= u64::from(last_frame) {
                        assert!(direct, "frame {frame_num} direct");
                    }
                }
                assert!(recovered.contains_key(&u64::from(burst_start)));
                assert!(recovered.contains_key(&(u64::from(burst_start) + 1)));
            }
        }
    }
}

#[test]
fn e2e_feedback_wire_to_sender() {
    let config = e2e_config(E2eProfile::Loss, 3, 1);
    let mut enc = Encoder::new(config.clone()).unwrap();
    assert_eq!(enc.redundancy(), Feedback::None);

    let codec = FeedbackCodec::new(&config);
    enc.apply_feedback(
        codec
            .decode_bytes(&codec.encode_bytes(Feedback::High))
            .unwrap(),
    );
    assert_eq!(enc.redundancy(), Feedback::High);

    let pkts = enc
        .encode_payload(random_payload(&mut TestRng::seeded(), 2048))
        .unwrap();
    assert!(pkts.iter().any(|p| p.is_parity));
}
