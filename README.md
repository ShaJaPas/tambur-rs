# tambur-rs

**Streaming forward error correction for real-time applications.**

[![crates.io](https://img.shields.io/crates/v/tambur-rs.svg)](https://crates.io/crates/tambur-rs)
[![docs.rs](https://docs.rs/tambur-rs/badge.svg)](https://docs.rs/tambur-rs)
[![License](https://img.shields.io/crates/l/tambur-rs.svg)](https://crates.io/crates/tambur-rs)
[![CI](https://github.com/ShaJaPas/tambur-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ShaJaPas/tambur-rs/actions/workflows/ci.yml)
[![Coverage](https://coveralls.io/repos/github/ShaJaPas/tambur-rs/badge.svg)](https://coveralls.io/github/ShaJaPas/tambur-rs)

## What is this?

tambur-rs is a Rust implementation of **streaming FEC** (forward error correction) codes. It protects real-time data streams — video frames, sensor readings, game state, audio packets — against network loss without waiting for retransmission.

Unlike traditional **block FEC** (which collects a full block before sending any repair data), tambur-rs operates on a **sliding window** of recent frames. The encoder emits repair packets alongside the data; the decoder reconstructs lost packets as soon as enough information arrives, bounded by a configurable latency deadline (τ, tau).

The crate is **transport-agnostic** — it produces opaque `FecDatagram`s and expects `Feedback` bytes. Send them over UDP, QUIC, WebRTC DataChannel, or any unreliable channel.

## Features

- **Sliding-window FEC** — Cauchy-matrix codes over a `(2τ+1)`-frame window, not fixed blocks.
- **Transport-agnostic** — no assumptions about media format or protocol.
- **Adaptive redundancy** — the receiver reports loss; the sender adjusts how much repair data to send (none / low / high).
- **Pluggable bandwidth prediction** — implement `BandwidthPredictor` for your own adaptation logic. The crate ships two baselines: `HighBandwidthPredictor` and `LowBandwidthPredictor`.
- **Zero-copy IO path** — `FecDatagram::from_bytes` / `wire_parts` avoid copying payload bytes when using `Bytes`.
- **Discrete feedback levels** — exactly three wire-efficient modes: 0%, 25%, 50% overhead.

## When to use tambur-rs

| Good fit | Poor fit |
|----------|----------|
| Live video/audio streams | Reliable byte streams (TCP) |
| Game state / sensor telemetry | File download / batch transfer |
| Bursty lossy links | Streams with < 1 packet per frame |
| Custom bandwidth control | Full video stack (no codec or RTP) |

## Quick start

```rust
use bytes::Bytes;
use tambur_rs::{Config, Encoder, Decoder, FecDatagram, Feedback};

// 1. Both sides agree on the same session parameters.
let config = Config::builder()
    .tau(3u8.try_into().unwrap())        // latency budget: up to 3 frames
    .max_frame_size(4096)
    .build()
    .unwrap();

// 2. Sender: encode frames into FEC datagrams.
let mut enc = Encoder::new(config.clone()).unwrap();
enc.apply_feedback(Feedback::High);      // 50% parity
let datagrams = enc.encode_payload(Bytes::from_static(b"hello")).unwrap();

// 3. Receiver: feed datagrams, recover frames.
let mut dec = Decoder::new(config.clone()).unwrap();
for pkt in datagrams {
    let wire = pkt.to_bytes().unwrap();
    let datagram = FecDatagram::from_bytes(wire).unwrap();
    let outcome = dec.receive_datagram(datagram, Duration::ZERO);
    for event in outcome.events {
        if let DecoderEvent::FrameRecovered(frame) = event {
            assert_eq!(&frame.payload[..], b"hello");
        }
    }
}
```

See the **Examples** section below for a full sender/receiver with adaptive feedback.

## Configuration reference

Everything starts with [`Config::builder()`]: each parameter has a sensible default. Change only what you need.

### `tau` (latency deadline)

- **Default:** `3`
- **Range:** `1` – `8`
- **What it is:** How many frames the decoder is willing to wait before declaring a frame lost. One frame = one `encode_payload` call.
- **Effect on behavior:**
  - `tau=1`: the decoder tracks `2*1+1 = 3` frames. Loss that spans more than 3 frames causes unrecoverable gaps. Latency is low (good for voice / interactive).
  - `tau=8`: the decoder tracks `2*8+1 = 17` frames. Can survive longer bursts but adds more end-to-end delay and memory.
- **Must match on encoder and decoder.** If they disagree, recovery will fail.

### `w` (word size / field width)

- **Default:** `W32`
- **Options:** `W8` (GF(2⁸)) or `W32` (GF(2³²))
- **What it is:** The mathematical field width for FEC coding. Affects the stripe size calculation: `stripe_size = w * packet_size`.
- **Effect on behavior:**
  - `W32`: wider stripes (32 × packet_size bytes per stripe). Faster on 64-bit CPUs with SIMD. Default choice.
  - `W8`: narrower stripes (8 × packet_size bytes per stripe). Use when you need fine-grained striping (more stripes per frame = more datagrams but smaller each).
- **When to change:** If you need stripes smaller than `32 * packet_size` bytes. Otherwise keep the default.

### `packet_size`

- **Default:** `8`
- **What it is:** The FEC symbol size in bytes (this is **not** the network MTU). Multiplied by `w` to produce `stripe_size`.
- **Effect on behavior:**
  - Smaller values → more stripes per frame, more datagram headers, larger coding matrices.
  - Larger values → fewer stripes, less header overhead, but each stripe must still fit in `max_pkt_size`.
- **When to change:** In practice, `8` with `W32` gives 256-byte stripes — a good balance. Change only if you have unusual frame size / MTU constraints.

### `max_pkt_size` (wire MTU)

- **Default:** `1500`
- **What it is:** The maximum size of one FEC datagram payload on the wire. This should match your path MTU minus IP/UDP headers.
- **Effect on behavior:**
  - If `stripe_size > max_pkt_size`, `Config::builder().build()` returns `ConfigError::StripeLargerThanPacket` — the config is invalid.
  - Each datagram carries as many full stripes as fit: `floor(max_pkt_size / stripe_size)`.
- **When to change:** Set to `1200` for safe Internet UDP, `1500` for LAN, or lower for tunnels/VPN.

### `max_data_stripes`

- **Default:** `64`
- **What it is:** Upper limit on how many data stripes one source frame can be split into.
- **Effect on behavior:**
  - A frame bigger than `max_data_stripes * stripe_size` cannot be encoded. Either `max_frame_size` or this cap will reject it.
  - Higher values → larger coding matrices, more memory, more datagrams per frame.
- **When to change:** Set from your peak frame size: `ceil(max_payload / stripe_size)`. If you send 100 KB frames with 256 B stripes you need ~400 stripes.

### `max_fec_stripes`

- **Default:** `32`
- **What it is:** Upper limit on how many parity (repair) stripes can be generated per frame.
- **Effect on behavior:**
  - At `Feedback::High` (50% parity), the encoder emits `ceil(data_stripes / 2)` parity stripes, capped here.
  - If the cap is too low for your frame size + feedback level, the encoder may fail at runtime.
- **When to change:** If you use `Feedback::High` on large frames, set this high enough. E2e test defaults: 32–75.

### `max_frame_size`

- **Default:** `16383`
- **What it is:** Hard limit on the byte payload you pass to `encode_payload`.
- **Effect on behavior:**
  - `Encoder::encode_payload` returns `CodecError::PayloadTooLarge` if this is exceeded.
  - Independent of `max_data_stripes` — the tighter of the two limits wins.
- **When to change:** Set to the largest frame your application produces.

### `feedback_interval`

- **Default:** `2 seconds`
- **What it is:** Minimum time between loss report emissions from the decoder.
- **Effect on behavior:**
  - Short interval (e.g. `100ms`): fast adaptation, more control traffic.
  - Long interval (e.g. `5s`): stabler redundancy level, slower reaction, less traffic.
- **When to change:** Use `Duration::ZERO` in tests for instant feedback. In production, balance control overhead vs reaction time.

### `high_redundancy_byte` and `redundancy_step_factor`

- **Defaults:** `1` and `2`
- **What they are:** Wire encoding bytes for the `Feedback` enum. `Feedback::None` is always `0`.
- **Effect on behavior:**
  - Not visible to most users. Only matters for wire interoperability with non-default Tambur implementations.
  - `Feedback::High` → `high_redundancy_byte` (default `1`)
  - `Feedback::Low` → `high_redundancy_byte × redundancy_step_factor` (default `1×2 = 2`)
- **When to change:** Only when your wire protocol must match a different byte scheme.

### `feedback_history_len`

- **Default:** `3`
- **What it is:** How many past `LossReport`s to keep for the predictor.
- **Effect on behavior:**
  - `FeedbackManager` trims its internal deque to this length after each report.
  - Custom predictors receive this history in `RecommendContext::history`.
- **When to change:** `1` for memoryless heuristics; `3+` for ML or trend-based policies.

### `parity_delay`

- **Default:** `0`
- **What it is:** Number of frames to delay parity stripe emission.
- **Effect on behavior:**
  - `0`: parity packets for frame N are sent alongside data for frame N.
  - Non-zero: parity for frame N appears on the wire at frame `N + parity_delay`. Can help align repair data with downstream playout but complicates the mental model.
- **When to change:** Leave at `0` unless you have a specific scheduling requirement. Must match on encoder and decoder.

### `min_window_parity`

- **Default:** `0` (disabled)
- **What it is:** A floor on the total parity stripes in the sliding window. If the sum of parity stripes over the last `window_frames` emitted frames is below this, the next frame gets topped up.
- **Effect on behavior:**
  - `0`: no minimum — each frame gets just enough parity for its feedback level.
  - `N > 0`: guarantees at least `N` parity equations are in flight at any time, improving burst protection. This is especially important for **small frames**: a 50-byte payload may fit in a single stripe, so even `Feedback::High` only adds 1 parity stripe — barely enough to survive a burst. A minimum ensures resilience regardless of frame size.
- **When to change:** Use `target_burst_protection(burst_packets)` on the builder instead of setting this directly — it computes the right value from τ.

## How it works

### The sliding window

The encoder and decoder maintain a window of the last `2τ+1` frames. When the encoder processes frame N, it generates:

- **Data stripes**: the frame payload, split into `w × packet_size`-byte chunks.
- **Parity stripes**: repair data computed from the last `2τ+1` frames using a Cauchy matrix, at the requested redundancy level.

The decoder collects data and parity stripes. Once it has enough for a given frame (all data stripes, or enough data + parity to solve the linear system), it recovers the frame and emits `DecoderEvent::FrameRecovered`.

### Feedback loop

1. Decoder detects a loss window closing → emits `LossReportReady`.
2. Your code (or `FeedbackManager`) passes the report to a `BandwidthPredictor`.
3. The predictor chooses the next `Feedback` level.
4. You send the feedback byte to the encoder (via your transport).
5. The encoder adjusts future parity generation.

### Redundancy levels

| Feedback | Parity ratio | Extra bandwidth |
|----------|-------------|-----------------|
| `None` | 0% | +0% |
| `Low` | 25% | +25% |
| `High` | 50% | +50% |

## Examples

### Sender with adaptive feedback

```rust
use bytes::Bytes;
use std::time::Duration;
use tambur_rs::{
    Config, Encoder, Feedback, FeedbackCodec,
};

let config = Config::builder()
    .tau(3u8.try_into().unwrap())
    .feedback_interval(Duration::from_secs(2))
    .build()
    .unwrap();

let mut enc = Encoder::new(config.clone()).unwrap();
let feedback_codec = FeedbackCodec::new(&config);

// Encode a frame
let datagrams = enc.encode_payload(Bytes::from_static(b"frame data")).unwrap();
for pkt in &datagrams {
    let (header, payload) = pkt.wire_parts().unwrap();
    send_over_udp(header, payload);
}

// Apply feedback received from the decoder
let fb = feedback_codec.decode_bytes(&[1]).unwrap(); // wire byte 1 → High
enc.apply_feedback(fb);
```

### Receiver with bandwidth prediction

```rust
use bytes::Bytes;
use std::time::Duration;
use tambur_rs::{
    Config, Decoder, DecoderEvent, FecDatagram, Feedback,
    FeedbackManager, HighBandwidthPredictor,
};

let config = Config::builder().build().unwrap();
let mut dec = Decoder::new(config.clone()).unwrap();
let mut fb_mgr = FeedbackManager::with_current_feedback(
    HighBandwidthPredictor,
    config.clone(),
    Feedback::High,
);
let mut now = Duration::ZERO;

for wire_packet in receive_from_network() {
    let datagram = FecDatagram::from_bytes(wire_packet).unwrap();
    let outcome = dec.receive_datagram(datagram, now);
    for event in outcome.events {
        match event {
            DecoderEvent::FrameRecovered(frame) => {
                playout(frame.payload);
            }
            DecoderEvent::LossReportReady(report) => {
                let fb = fb_mgr.handle_report(report);
                let wire_byte = fb_mgr.encode_wire(fb);
                send_feedback_byte(wire_byte);
            }
        }
    }
    now += Duration::from_millis(1);
}
```


## Wire format

**FEC datagram** — 18-byte big-endian header + stripe payload:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | `seq_num` (u32, monotonic) |
| 4 | 2 | `is_parity` (MSB) + `frame_num` (15 bits) |
| 6 | 8 | `sizes_of_frames_encoding` (u64) |
| 14 | 2 | `pos_in_frame` (u16) |
| 16 | 2 | `stripe_pos_in_frame` (u16) |
| 18 | — | stripe payload bytes |

**Feedback** — single byte: `0` = None, `high_redundancy_byte` = High, `high_redundancy_byte × step` = Low.

## Minimum supported Rust version (MSRV)

Rust 1.89 or later.

## License

Apache License, Version 2.0
