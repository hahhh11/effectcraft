//! Animated PNG (APNG): a normal PNG stream whose frame 0 is the still image and whose later
//! frames hang off it as `fdAT` chunks, so anything that can read a PNG shows the first frame and
//! APNG-aware viewers (browsers, Quick Look, mpv) play the whole animation with 8-bit alpha.
//!
//! | Chunk | Where | Holds |
//! |---|---|---|
//! | `acTL` | before the first `IDAT` | the frame count and the play count (0: loop forever) |
//! | `fcTL` | before each frame's data | the frame's rectangle, delay and disposal |
//! | `IDAT` | frame 0 | its zlib stream |
//! | `fdAT` | frames 1.. | a sequence number, then that frame's zlib stream |
//!
//! Every frame is a full-canvas replacement (dispose `NONE`, blend `SOURCE`), so a decoder never
//! has to composite one frame onto the last.

use std::io::Write;

use effectcraft_project::Comp;

use crate::pipeline::Cx;
use crate::{ExportError, Result, State, batch_size, io, pngenc};

/// Writes the APNG stream: the header when it is created, one call per frame, then [`Writer::finish`].
pub(crate) struct Writer<'w, W: Write> {
    out: &'w mut W,
    width: u32,
    height: u32,
    frames: u32,
    written: u32,
    /// The sequence number the next `fcTL` / `fdAT` carries (shared by both, in file order).
    seq: u32,
}

impl<'w, W: Write> Writer<'w, W> {
    /// Write the signature, `IHDR` and `acTL`: the animation has `frames` frames and plays
    /// `plays` times (0 loops forever).
    pub(crate) fn start(out: &'w mut W, width: u32, height: u32, colour: pngenc::Colour, frames: u32, plays: u32) -> Result<Writer<'w, W>> {
        if width == 0 || height == 0 {
            return Err(ExportError::Unsupported(format!("APNG frames cannot be {width}×{height}")));
        }
        if frames == 0 {
            return Err(ExportError::Unsupported("an APNG has at least one frame".into()));
        }
        out.write_all(&pngenc::SIGNATURE).map_err(io)?;
        pngenc::chunk(out, b"IHDR", &pngenc::ihdr(width, height, colour))?;
        let mut actl = Vec::with_capacity(8);
        actl.extend_from_slice(&frames.to_be_bytes());
        actl.extend_from_slice(&plays.to_be_bytes());
        pngenc::chunk(out, b"acTL", &actl)?;
        Ok(Writer { out, width, height, frames, written: 0, seq: 0 })
    }

    /// Add one frame from its compressed image stream (see [`frame_data`]), shown for
    /// `delay_num / delay_den` seconds.
    pub(crate) fn frame(&mut self, image: &[u8], delay_num: u16, delay_den: u16) -> Result<()> {
        if self.written >= self.frames {
            return Err(ExportError::Encode(format!("frame {} of an APNG announced as {} frames", self.written + 1, self.frames)));
        }
        let mut fctl = Vec::with_capacity(26);
        fctl.extend_from_slice(&self.seq.to_be_bytes());
        fctl.extend_from_slice(&self.width.to_be_bytes());
        fctl.extend_from_slice(&self.height.to_be_bytes());
        fctl.extend_from_slice(&0u32.to_be_bytes()); // x offset
        fctl.extend_from_slice(&0u32.to_be_bytes()); // y offset
        fctl.extend_from_slice(&delay_num.to_be_bytes());
        fctl.extend_from_slice(&delay_den.to_be_bytes());
        fctl.push(0); // dispose: NONE (each frame is the whole canvas)
        fctl.push(0); // blend: SOURCE
        pngenc::chunk(self.out, b"fcTL", &fctl)?;
        self.seq = self.seq.wrapping_add(1);
        if self.written == 0 {
            pngenc::chunk(self.out, b"IDAT", image)?;
        } else {
            let mut fdat = Vec::with_capacity(image.len() + 4);
            fdat.extend_from_slice(&self.seq.to_be_bytes());
            fdat.extend_from_slice(image);
            pngenc::chunk(self.out, b"fdAT", &fdat)?;
            self.seq = self.seq.wrapping_add(1);
        }
        self.written = self.written.saturating_add(1);
        Ok(())
    }

    /// Write `IEND`. Fails if fewer frames were written than `acTL` announced.
    pub(crate) fn finish(self) -> Result<()> {
        if self.written != self.frames {
            return Err(ExportError::Encode(format!("the APNG announced {} frames but {} were written", self.frames, self.written)));
        }
        pngenc::chunk(self.out, b"IEND", &[])
    }
}

/// The filtered, zlib-compressed image stream of one frame (what [`Writer::frame`] takes).
pub(crate) fn frame_data(px: &[u8], width: u32, height: u32, colour: pngenc::Colour) -> Result<Vec<u8>> {
    let rows = pngenc::filter_rows(px, width, height, colour)?;
    pngenc::zlib(&rows)
}

/// The delay of one frame at `rate` (frames a second, `num / den`), as the `delay_num /
/// delay_den` pair `fcTL` stores: exact where both fit in 16 bits, milliseconds otherwise.
pub(crate) fn delay_for(rate_num: i64, rate_den: i64) -> (u16, u16) {
    let (mut num, mut den) = (rate_num.max(1), rate_den.max(1));
    let g = gcd(num, den);
    num /= g;
    den /= g;
    if let (Ok(n), Ok(d)) = (u16::try_from(num), u16::try_from(den)) {
        // One frame lasts den / num seconds: delay_num = den, delay_den = num.
        return (d.max(1), n.max(1));
    }
    let ms = (1000.0 * den as f64 / num as f64).round();
    let ms = ms.clamp(1.0, 65_535.0) as u16;
    (ms, 1000)
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// Render the job's frames as an APNG at `path`.
pub(crate) async fn apng(cx: &Cx<'_>, comp: &Comp, w: u32, h: u32, st: &mut State<'_>) -> Result<crate::Report> {
    let path = cx.place(cx.path, 1);
    let mut file = cx.create(&path)?;
    // `Cx::pixels_with` always returns four bytes a pixel (RGBA), regardless of
    // `output.channels`; RGB just flattens alpha over the background. Declaring a
    // three-byte RGB image here would mis-stride every row and turn the frames
    // into coloured vertical stripes. Keep the colour type in sync with the data.
    let colour = pngenc::Colour::Rgba;
    let rate = cx.settings.rate(comp);
    let (delay_num, delay_den) = delay_for(rate.num, rate.den);
    let frames = u32::try_from(st.total()).map_err(|_| ExportError::Unsupported("this animation has too many frames for APNG".into()))?;
    let plays = if cx.output.apng_loop { 0 } else { 1 };
    let mut png = Writer::start(&mut file, w, h, colour, frames, plays)?;
    let batch = batch_size();
    let mut i = 0;
    while i < st.total() {
        let end = (i + batch).min(st.total());
        let data: Vec<Result<Vec<u8>>> = cx
            .frames(comp, (i..end).collect(), |_, img| {
                let px = cx.pixels_with(&img, comp, cx.output.channels, w, h, false);
                frame_data(&px, w, h, colour)
            })
            .await;
        for image in data {
            png.frame(&image?, delay_num, delay_den)?;
        }
        st.advance(end - i)?;
        i = end;
    }
    png.finish()?;
    let bytes = file.finish()?;
    Ok(crate::Report { path, frames: 0, width: w, height: h, seconds: 0.0, bytes, audio: false, log: None, overflow: vec![] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_are_exact_for_broadcast_rates() {
        assert_eq!(delay_for(24, 1), (1, 24));
        assert_eq!(delay_for(30, 1), (1, 30));
        // 29.97 = 30000/1001: 1001/30000 of a second a frame.
        assert_eq!(delay_for(30_000, 1001), (1001, 30_000));
        // Beyond 16 bits: milliseconds.
        assert_eq!(delay_for(254_016_000, 1), (1, 1000));
        assert_eq!(delay_for(0, 0), (1, 1));
    }

    #[test]
    fn every_frame_but_the_first_is_fdat() {
        let px = [255u8; 4 * 4 * 4];
        let colour = pngenc::Colour::Rgba;
        let mut out = Vec::new();
        let mut w = Writer::start(&mut out, 4, 4, colour, 3, 0).expect("start");
        for _ in 0..3 {
            w.frame(&frame_data(&px, 4, 4, colour).expect("data"), 1, 30).expect("frame");
        }
        w.finish().expect("finish");
        let count = |kind: &[u8; 4]| out.windows(4).filter(|c| c.starts_with(kind)).count();
        assert_eq!(count(b"acTL"), 1);
        assert_eq!(count(b"fcTL"), 3, "one per frame");
        assert_eq!(count(b"fdAT"), 2, "frames 1 and 2");
        assert_eq!(count(b"IDAT"), 1, "frame 0 only");
        assert_eq!(count(b"IEND"), 1);
        assert!(out.starts_with(&pngenc::SIGNATURE), "PNG signature");
    }

    #[test]
    fn a_short_animation_is_an_error() {
        let mut out = Vec::new();
        let mut w = Writer::start(&mut out, 2, 2, pngenc::Colour::Rgba, 2, 0).expect("start");
        w.frame(&frame_data(&[0u8; 16], 2, 2, pngenc::Colour::Rgba).expect("data"), 1, 24).expect("frame");
        assert!(w.finish().is_err(), "one frame short of acTL");
    }
}
