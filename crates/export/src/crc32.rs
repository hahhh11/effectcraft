//! CRC-32 (IEEE 802.3), the checksum PNG chunks carry.
//!
//! Table built at compile time; no dependencies, and nothing here can fail.

const POLY: u32 = 0xEDB8_8320;

const TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { POLY ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// The CRC-32 of `data`, continued from `crc` (0 for a new checksum).
pub(crate) fn update(crc: u32, data: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in data {
        // The index is masked to eight bits, so it is always inside the table.
        let idx = ((c ^ b as u32) & 0xFF) as usize;
        let Some(&v) = TABLE.get(idx) else { return !c };
        c = v ^ (c >> 8);
    }
    !c
}

/// The CRC-32 of `data`.
#[cfg(test)]
fn crc32(data: &[u8]) -> u32 {
    update(0, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // Continued in two halves: the same checksum as in one go.
        assert_eq!(update(update(0, b"1234"), b"56789"), 0xCBF4_3926);
    }
}
