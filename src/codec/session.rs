//! Session-wide frame indices across 15-bit wire wraps.

use crate::datagram::MAX_FRAME_NUM;

/// Number of distinct wire `frame_num` values (`0..=MAX_FRAME_NUM`).
pub(crate) const FRAME_SPACE: u64 = MAX_FRAME_NUM as u64 + 1;

/// Half the wire space — used for forward/backward disambiguation (RTP-style).
pub(crate) const HALF_FRAME_SPACE: u16 = MAX_FRAME_NUM / 2 + 1;

/// Map a monotonic session index to the on-wire 15-bit frame number.
pub(crate) const fn wire_frame_num(session_index: u64) -> u16 {
    (session_index % FRAME_SPACE) as u16
}

/// Lap / epoch component of a session index.
pub(crate) const fn session_epoch(session_index: u64) -> u64 {
    session_index / FRAME_SPACE
}

const fn session_from_parts(epoch: u64, wire: u16) -> u64 {
    epoch * FRAME_SPACE + wire as u64
}

/// True when `wire` is more than `window` session frames behind `last_session`.
pub(crate) fn is_out_of_window_session(session: u64, last_session: u64, window: u16) -> bool {
    last_session.saturating_sub(session) >= u64::from(window)
}

/// True when `old_session` has fallen out of the coding window relative to `new_session`.
pub(crate) fn frame_expired_session(new_session: u64, old_session: u64, memory: u16) -> bool {
    new_session.saturating_sub(old_session) >= u64::from(memory)
}

/// Resolve a wire `frame_num` to a monotonic session index, closest to the last seen index.
#[derive(Debug, Clone, Default)]
pub(crate) struct SessionIndexTracker {
    last_session: Option<u64>,
}

impl SessionIndexTracker {
    pub(crate) fn resolve_wire(&mut self, wire: u16) -> u64 {
        let session = match self.last_session {
            None => u64::from(wire),
            Some(last) => {
                let last_wire = wire_frame_num(last);
                let epoch = session_epoch(last);
                if wire < last_wire && last_wire - wire > HALF_FRAME_SPACE - 1 {
                    session_from_parts(epoch + 1, wire)
                } else if wire > last_wire
                    && wire > HALF_FRAME_SPACE
                    && wire_frame_num(last) == 0
                    && epoch > 0
                {
                    session_from_parts(epoch - 1, wire)
                } else {
                    session_from_parts(epoch, wire)
                }
            }
        };
        self.last_session = Some(session);
        session
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_roundtrip_within_lap() {
        for session in [0u64, 1, 100, MAX_FRAME_NUM as u64] {
            assert_eq!(wire_frame_num(session), session as u16, "session {session}");
        }
    }

    #[test]
    fn wire_wraps_at_frame_space() {
        assert_eq!(wire_frame_num(FRAME_SPACE), 0);
        assert_eq!(wire_frame_num(FRAME_SPACE + 42), 42);
        assert_eq!(
            wire_frame_num(2 * FRAME_SPACE + MAX_FRAME_NUM as u64),
            MAX_FRAME_NUM
        );
    }

    #[test]
    fn tracker_monotonic_through_wrap() {
        let mut t = SessionIndexTracker::default();
        assert_eq!(t.resolve_wire(0), 0);
        assert_eq!(t.resolve_wire(MAX_FRAME_NUM), u64::from(MAX_FRAME_NUM));
        assert_eq!(t.resolve_wire(0), FRAME_SPACE);
        assert_eq!(t.resolve_wire(1), FRAME_SPACE + 1);
        assert_eq!(
            t.resolve_wire(MAX_FRAME_NUM),
            FRAME_SPACE + u64::from(MAX_FRAME_NUM)
        );
    }

    #[test]
    fn tracker_rejects_stale_previous_lap_as_lower_session() {
        let mut t = SessionIndexTracker::default();
        assert_eq!(t.resolve_wire(MAX_FRAME_NUM), u64::from(MAX_FRAME_NUM));
        assert_eq!(t.resolve_wire(0), FRAME_SPACE);
        // Stale packet from previous lap.
        assert_eq!(t.resolve_wire(MAX_FRAME_NUM), u64::from(MAX_FRAME_NUM));
    }

    #[test]
    fn out_of_window_session() {
        assert!(!is_out_of_window_session(10, 15, 7));
        assert!(is_out_of_window_session(8, 15, 7));
        assert!(!is_out_of_window_session(FRAME_SPACE, FRAME_SPACE + 5, 7));
        assert!(is_out_of_window_session(FRAME_SPACE, FRAME_SPACE + 10, 7));
    }
}
