//! A minimal ZIP writer for the SVGA container: one entry per file, every entry stored, no
//! encryption and no ZIP64 — a container that would need it is refused with an error rather than
//! written wrong.
//!
//! Entries are stored rather than deflated because the SVGA players in the wild read the archive
//! with their own minimal ZIP readers, not zlib-aware ones: a stored entry needs no decompressor
//! at all. The frames are PNGs (already compressed) and `movie.binary` is a zlib stream the
//! players inflate themselves, so deflating again would win a percent at best.
//!
//! Timestamps are the ZIP epoch (1980-01-01 00:00), so the bytes depend only on the content: two
//! exports of the same frames are byte-identical, which is what the round-trip tests check.

use std::io::Write;

use crate::{ExportError, Result, crc32, io};

/// The DOS date of 1980-01-01: year 0 (1980), month 1, day 1.
const DOS_DATE: u16 = 0x0021;
/// The DOS time of 00:00:00.
const DOS_TIME: u16 = 0;

struct Entry {
    name: String,
    method: u16,
    crc: u32,
    compressed: u32,
    uncompressed: u32,
    offset: u32,
}

/// Builds a ZIP archive: [`Zip::add`] per file, then [`Zip::finish`] writes the central directory.
pub(crate) struct Zip<W: Write> {
    out: W,
    entries: Vec<Entry>,
    offset: u64,
}

impl<W: Write> Zip<W> {
    pub(crate) fn new(out: W) -> Zip<W> {
        Zip { out, entries: Vec::new(), offset: 0 }
    }

    /// Add `data` as the entry `name` (ASCII, relative, forward slashes only), stored.
    pub(crate) fn add(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.add_entry(name, data, false)
    }

    /// Add `data` deflated, for an entry of plain text that shrinks a lot (`movie.spec`). Every
    /// reader that opens a ZIP at all can inflate an entry, so this stays compatible.
    pub(crate) fn add_deflated(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.add_entry(name, data, true)
    }

    fn add_entry(&mut self, name: &str, data: &[u8], compress: bool) -> Result<()> {
        if name.is_empty() || name.len() > 0xFFFF {
            return Err(ExportError::Encode(format!("{name:?} is not a usable archive entry name")));
        }
        if !name.bytes().all(|b| b.is_ascii_graphic() || b == b' ' || b == b'/') {
            return Err(ExportError::Encode(format!("{name:?} is not an ASCII archive entry name")));
        }
        if name.starts_with('/') || name.split('/').any(|part| part == ".." || part.is_empty()) {
            return Err(ExportError::Encode(format!("{name:?} is not a relative archive path")));
        }
        let uncompressed =
            u32::try_from(data.len()).map_err(|_| ExportError::Unsupported(format!("{name} is over 4 GiB, more than an SVGA container holds")))?;
        let deflated = if compress { Some(deflate(data)?) } else { None };
        let (method, body): (u16, &[u8]) = match &deflated {
            Some(d) if d.len() < data.len() => (8, d.as_slice()),
            _ => (0, data),
        };
        let compressed = u32::try_from(body.len()).map_err(|_| ExportError::Unsupported(format!("{name} is over 4 GiB, more than an SVGA container holds")))?;
        let offset = u32::try_from(self.offset).map_err(|_| ExportError::Unsupported("an SVGA container over 4 GiB".into()))?;
        let crc = crc32::crc32(data);
        let name_len = u16::try_from(name.len()).map_err(|_| ExportError::Encode(format!("{name:?} is too long")))?;

        let mut h = Vec::with_capacity(30 + name.len());
        h.extend_from_slice(&0x0403_4B50u32.to_le_bytes()); // local file header
        h.extend_from_slice(&20u16.to_le_bytes()); // version needed: 2.0
        h.extend_from_slice(&0x0800u16.to_le_bytes()); // flags: names are UTF-8
        h.extend_from_slice(&method.to_le_bytes());
        h.extend_from_slice(&DOS_TIME.to_le_bytes());
        h.extend_from_slice(&DOS_DATE.to_le_bytes());
        h.extend_from_slice(&crc.to_le_bytes());
        h.extend_from_slice(&compressed.to_le_bytes());
        h.extend_from_slice(&uncompressed.to_le_bytes());
        h.extend_from_slice(&name_len.to_le_bytes());
        h.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        h.extend_from_slice(name.as_bytes());
        self.out.write_all(&h).map_err(io)?;
        self.out.write_all(body).map_err(io)?;
        self.offset = self.offset.saturating_add(h.len() as u64).saturating_add(body.len() as u64);
        self.entries.push(Entry { name: name.to_string(), method, crc, compressed, uncompressed, offset });
        Ok(())
    }

    /// Write the central directory and its end record, and give back the sink.
    pub(crate) fn finish(mut self) -> Result<W> {
        let start = u32::try_from(self.offset).map_err(|_| ExportError::Unsupported("an SVGA container over 4 GiB".into()))?;
        for e in &self.entries {
            let name_len = u16::try_from(e.name.len()).map_err(|_| ExportError::Encode(format!("{:?} is too long", e.name)))?;
            let mut c = Vec::with_capacity(46 + e.name.len());
            c.extend_from_slice(&0x0201_4B50u32.to_le_bytes()); // central directory header
            c.extend_from_slice(&20u16.to_le_bytes()); // version made by
            c.extend_from_slice(&20u16.to_le_bytes()); // version needed
            c.extend_from_slice(&0x0800u16.to_le_bytes());
            c.extend_from_slice(&e.method.to_le_bytes());
            c.extend_from_slice(&DOS_TIME.to_le_bytes());
            c.extend_from_slice(&DOS_DATE.to_le_bytes());
            c.extend_from_slice(&e.crc.to_le_bytes());
            c.extend_from_slice(&e.compressed.to_le_bytes());
            c.extend_from_slice(&e.uncompressed.to_le_bytes());
            c.extend_from_slice(&name_len.to_le_bytes());
            c.extend_from_slice(&0u16.to_le_bytes()); // extra
            c.extend_from_slice(&0u16.to_le_bytes()); // comment
            c.extend_from_slice(&0u16.to_le_bytes()); // disk number
            c.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
            c.extend_from_slice(&0u32.to_le_bytes()); // external attributes
            c.extend_from_slice(&e.offset.to_le_bytes());
            c.extend_from_slice(e.name.as_bytes());
            self.out.write_all(&c).map_err(io)?;
            self.offset = self.offset.saturating_add(c.len() as u64);
        }
        let size = u32::try_from(self.offset.saturating_sub(start as u64)).map_err(|_| ExportError::Unsupported("an SVGA container over 4 GiB".into()))?;
        let count = u16::try_from(self.entries.len()).map_err(|_| ExportError::Unsupported("more than 65535 files in one SVGA container".into()))?;
        let mut eocd = Vec::with_capacity(22);
        eocd.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
        eocd.extend_from_slice(&0u16.to_le_bytes()); // this disk
        eocd.extend_from_slice(&0u16.to_le_bytes()); // disk with the central directory
        eocd.extend_from_slice(&count.to_le_bytes());
        eocd.extend_from_slice(&count.to_le_bytes());
        eocd.extend_from_slice(&size.to_le_bytes());
        eocd.extend_from_slice(&start.to_le_bytes());
        eocd.extend_from_slice(&0u16.to_le_bytes()); // archive comment length
        self.out.write_all(&eocd).map_err(io)?;
        self.out.flush().map_err(io)?;
        Ok(self.out)
    }
}

/// Raw deflate (no zlib header), which is what a ZIP entry of method 8 holds.
fn deflate(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
    enc.write_all(data).map_err(io)?;
    enc.finish().map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(b: &[u8], p: usize) -> u16 {
        u16::from_le_bytes([b[p], b[p + 1]])
    }
    fn u32_at(b: &[u8], p: usize) -> u32 {
        u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
    }

    /// Read an archive back the way a player does: the end record, then the central directory,
    /// then each entry's local header and data.
    fn read_entries(b: &[u8]) -> Vec<(String, Vec<u8>)> {
        let eocd = (0..b.len().saturating_sub(21)).rev().find(|&p| b.get(p..p + 4) == Some(&[0x50, 0x4b, 0x05, 0x06])).expect("end of central directory");
        let count = u16_at(b, eocd + 10) as usize;
        let mut p = u32_at(b, eocd + 16) as usize;
        assert_eq!(count, u16_at(b, eocd + 8) as usize);
        let mut out = Vec::new();
        for _ in 0..count {
            assert_eq!(u32_at(b, p), 0x0201_4B50, "central directory header signature");
            let method = u16_at(b, p + 10);
            let compressed = u32_at(b, p + 20) as usize;
            let name_len = u16_at(b, p + 28) as usize;
            let extra = u16_at(b, p + 30) as usize;
            let comment = u16_at(b, p + 32) as usize;
            let offset = u32_at(b, p + 42) as usize;
            let name = String::from_utf8(b[p + 46..p + 46 + name_len].to_vec()).expect("utf-8 name");
            let local_name = u16_at(b, offset + 26) as usize;
            let local_extra = u16_at(b, offset + 28) as usize;
            let start = offset + 30 + local_name + local_extra;
            let body = &b[start..start + compressed];
            let data = match method {
                0 => body.to_vec(),
                8 => {
                    let mut d = flate2::read::DeflateDecoder::new(body);
                    let mut v = Vec::new();
                    std::io::Read::read_to_end(&mut d, &mut v).expect("inflate");
                    v
                }
                m => panic!("unexpected compression method {m}"),
            };
            out.push((name, data));
            p += 46 + name_len + extra + comment;
        }
        out
    }

    #[test]
    fn entries_round_trip_through_a_reader() {
        let mut zip = Zip::new(Vec::new());
        zip.add("movie.binary", &[1, 2, 3, 4, 5]).expect("entry");
        let text = b"the same bytes twice, which deflate likes a lot".repeat(40);
        zip.add("notes.txt", &text).expect("entry");
        let bytes = zip.finish().expect("finish");
        assert!(bytes.starts_with(b"PK\x03\x04"), "a ZIP local file header");
        let entries = read_entries(&bytes);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], ("movie.binary".to_string(), vec![1, 2, 3, 4, 5]));
        assert_eq!(entries[1], ("notes.txt".to_string(), text));
    }

    /// Every entry is stored: a player's own ZIP reader then needs no inflater, which is what the
    /// SVGA parsers in the wild assume (#SVGA "incorrect header check").
    #[test]
    fn every_entry_is_stored() {
        let noisy: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        let mut zip = Zip::new(Vec::new());
        zip.add("noise.bin", &noisy).expect("entry");
        let bytes = zip.finish().expect("finish");
        let entries = read_entries(&bytes);
        assert_eq!(entries[0].1, noisy);
        // Stored, so the archive is barely bigger than the data itself.
        assert!(bytes.len() < noisy.len() + 200, "{} bytes for {} of noise", bytes.len(), noisy.len());
    }

    #[test]
    fn odd_names_are_refused() {
        let mut zip = Zip::new(Vec::new());
        assert!(zip.add("", b"x").is_err());
        assert!(zip.add("帧.png", b"x").is_err(), "ASCII only, so every player finds the entry");
        assert!(zip.add("../escape", b"x").is_err());
        assert!(zip.add("a//b", b"x").is_err());
    }
}
