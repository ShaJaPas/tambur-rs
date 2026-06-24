//! Big-endian wire serialization.

pub(crate) fn put_u64(host: u64) -> [u8; 8] {
    host.to_be_bytes()
}

pub(crate) fn get_u64(net: &[u8]) -> Option<u64> {
    if net.len() < 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&net[..8]);
    Some(u64::from_be_bytes(buf))
}
