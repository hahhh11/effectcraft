//! The PNG pieces the animated writers share: chunks with their CRC, per-row filtering, a zlib
//! stream, and a whole still image (the SVGA container's frame pictures).
//!
//! 8-bit RGB or RGBA only, no interlacing — the frames a render produces. Filtering picks, per
//! row, the filter whose output has the smallest sum of absolute signed values, the standard
//! heuristic every PNG encoder uses (`None`, `Sub`, `Up`, `Average`, `Paeth`); this is what keeps
//! a rendered frame within a few percent of what libpng writes.

use std::io::Write;

use crate::{ExportError, Result, crc32, io};

/// The eight bytes every PNG starts with.
pub(crate) const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// The colour types an 8-bit frame uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Colour {
    /// Three bytes a pixel, no alpha.
    Rgb,
    /// Four bytes a pixel: straight (unpremultiplied) RGB and alpha.
    Rgba,
}

impl Colour {
    /// The PNG colour-type code.
    fn code(self) -> u8 {
        match self {
            Colour::Rgb => 2,
            Colour::Rgba => 6,
        }
    }

    /// Bytes a pixel.
    fn channels(self) -> usize {
        match self {
            Colour::Rgb => 3,
            Colour::Rgba => 4,
        }
    }
}

/// Write one chunk: length, four-byte type, data, then the CRC of type and data.
pub(crate) fn chunk(w: &mut impl Write, kind: &[u8; 4], data: &[u8]) -> Result<()> {
    let len = u32::try_from(data.len()).map_err(|_| ExportError::Encode("a PNG chunk over 4 GiB".into()))?;
    w.write_all(&len.to_be_bytes()).map_err(io)?;
    w.write_all(kind).map_err(io)?;
    w.write_all(data).map_err(io)?;
    let mut crc = crc32::update(0, kind);
    crc = crc32::update(crc, data);
    w.write_all(&crc.to_be_bytes()).map_err(io)
}

/// The 13-byte `IHDR` data for an 8-bit image.
pub(crate) fn ihdr(width: u32, height: u32, colour: Colour) -> Vec<u8> {
    let mut v = Vec::with_capacity(13);
    v.extend_from_slice(&width.to_be_bytes());
    v.extend_from_slice(&height.to_be_bytes());
    // Bit depth 8, colour type, deflate, adaptive filtering, no interlacing.
    v.extend_from_slice(&[8, colour.code(), 0, 0, 0]);
    v
}

/// The Paeth predictor (PNG filter type 4).
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let pa = (p - a as i16).abs();
    let pb = (p - b as i16).abs();
    let pc = (p - c as i16).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// The image's scan lines, each preceded by its filter byte: the filtered bytes every PNG image
/// stream holds before compression. `px` is row-major, top row first, `colour.channels()` bytes a
/// pixel.
pub(crate) fn filter_rows(px: &[u8], width: u32, height: u32, colour: Colour) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(ExportError::Encode(format!("a {width}×{height} image has no pixels")));
    }
    let bpp = colour.channels();
    let stride = (width as usize).checked_mul(bpp).ok_or_else(|| ExportError::Encode(format!("a {width}×{height} image is too wide")))?;
    let need = stride.checked_mul(height as usize).ok_or_else(|| ExportError::Encode(format!("a {width}×{height} image is too large")))?;
    if px.len() < need {
        return Err(ExportError::Encode(format!(
            "a {width}×{height} {} frame needs {need} bytes, not {}",
            if colour == Colour::Rgba { "RGBA" } else { "RGB" },
            px.len()
        )));
    }
    let mut out = Vec::with_capacity(need.saturating_add(height as usize));
    let mut prev = vec![0u8; stride];
    let mut candidate = vec![0u8; stride];
    let mut best = vec![0u8; stride];
    for y in 0..height as usize {
        let start = y.saturating_mul(stride);
        let Some(row) = px.get(start..start.saturating_add(stride)) else {
            return Err(ExportError::Encode("a frame ended before its last row".into()));
        };
        let mut best_filter = 0u8;
        let mut best_score = u64::MAX;
        for filter in 0..5u8 {
            let mut score = 0u64;
            for i in 0..stride {
                let x = row.get(i).copied().unwrap_or(0);
                let a = if i >= bpp { row.get(i - bpp).copied().unwrap_or(0) } else { 0 };
                let b = prev.get(i).copied().unwrap_or(0);
                let c = if i >= bpp { prev.get(i - bpp).copied().unwrap_or(0) } else { 0 };
                let v = match filter {
                    1 => x.wrapping_sub(a),
                    2 => x.wrapping_sub(b),
                    3 => x.wrapping_sub(((a as u16 + b as u16) / 2) as u8),
                    4 => x.wrapping_sub(paeth(a, b, c)),
                    _ => x,
                };
                if let Some(slot) = candidate.get_mut(i) {
                    *slot = v;
                }
                // Bytes are signed residuals: their absolute value is what the heuristic sums.
                score = score.saturating_add((v as i8).unsigned_abs() as u64);
            }
            if score < best_score {
                best_score = score;
                best_filter = filter;
                best.copy_from_slice(&candidate);
            }
        }
        out.push(best_filter);
        out.extend_from_slice(&best);
        prev.copy_from_slice(row);
    }
    Ok(out)
}

/// A zlib stream (deflate with the two-byte header and Adler-32 trailer PNG asks for).
pub(crate) fn zlib(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    enc.write_all(data).map_err(io)?;
    enc.finish().map_err(io)
}

/// A complete still PNG: signature, `IHDR`, one `IDAT`, `IEND`.
pub(crate) fn still(px: &[u8], width: u32, height: u32, colour: Colour) -> Result<Vec<u8>> {
    let mut file = Vec::new();
    file.extend_from_slice(&SIGNATURE);
    chunk(&mut file, b"IHDR", &ihdr(width, height, colour))?;
    chunk(&mut file, b"IDAT", &zlib(&filter_rows(px, width, height, colour)?)?)?;
    chunk(&mut file, b"IEND", &[])?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode an 8-bit PNG back to its pixels (via the `image` crate, which depends on `png`).
    fn decode(data: &[u8]) -> (u32, u32, Vec<u8>) {
        let img = image::load_from_memory_with_format(data, image::ImageFormat::Png).expect("decode");
        let rgba = img.to_rgba8();
        (rgba.width(), rgba.height(), rgba.into_raw())
    }

    #[test]
    fn still_roundtrips_every_filter() {
        // A gradient with alpha: every filter gets chosen for some row.
        let (w, h) = (37u32, 11u32);
        let px: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let x = (i % w) as u8;
                let y = (i / w) as u8;
                [x.wrapping_mul(7), y.wrapping_mul(23), x ^ y, 200u8]
            })
            .collect();
        let png = still(&px, w, h, Colour::Rgba).expect("encode");
        assert_eq!(&png[..8], &SIGNATURE);
        let (dw, dh, back) = decode(&png);
        assert_eq!((dw, dh), (w, h));
        assert_eq!(back, px, "8-bit RGBA comes back byte for byte");
    }

    #[test]
    fn rgb_still_roundtrips() {
        let (w, h) = (5u32, 3u32);
        let px: Vec<u8> = (0..(w * h * 3)).map(|i| (i * 11) as u8).collect();
        let png = still(&px, w, h, Colour::Rgb).expect("encode");
        let (dw, dh, back) = decode(&png);
        assert_eq!((dw, dh), (w, h));
        assert_eq!(back.len(), (w * h * 4) as usize);
        for (i, p) in back.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(&p[..3], px.get(i * 3..i * 3 + 3).unwrap_or(&[0, 0, 0]), "pixel {i}");
            assert_eq!(p[3], 255);
        }
    }

    #[test]
    fn short_input_is_an_error() {
        assert!(still(&[0u8; 3], 2, 2, Colour::Rgba).is_err());
        assert!(still(&[], 0, 0, Colour::Rgba).is_err());
    }

    #[test]
    fn paeth_matches_the_specification() {
        // From the PNG spec's own tie-break rule: the predictor whose distance is smallest wins,
        // and `a`, then `b`, then `c` break ties in that order.
        assert_eq!(paeth(10, 10, 10), 10, "a wins when a, b and c are equal");
        assert_eq!(paeth(0, 100, 0), 100, "b wins");
        assert_eq!(paeth(10, 20, 15), 15, "c wins: p is 15, so its distance is 0");
        assert_eq!(paeth(200, 10, 100), 100, "c wins again");
        assert_eq!(paeth(0, 0, 0), 0);
    }
}
