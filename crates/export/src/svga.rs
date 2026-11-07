//! SVGA (`.svga`): a ZIP container holding `movie.binary` — a protobuf `MovieEntity` — and one
//! PNG per animation frame, which is what the SVGA players in Chinese apps (and the web
//! `svgaplayerweb`) read.
//!
//! The format animates *sprites*: a sprite has one image and a list of key frames whose alpha,
//! transform and layout the player interpolates, and the key frames are spread evenly over the
//! movie's length. That is why a composited render — which has no vector structure left to keep —
//! becomes one sprite per frame, each with exactly one key frame per movie frame and alpha 1 on
//! its own frame only: at any moment one sprite is opaque and the rest are clear. The cost is
//! `frames²` key frames in `movie.binary`, which deflate shrinks to almost nothing (they are
//! nearly all zero) but which bounds the export to [`MAX_FRAMES`].
//!
//! Writes:
//!
//! | Entry | Holds |
//! |---|---|
//! | `frame_00000.png` … | one 8-bit RGBA (or RGB) PNG a frame, the sprite's `imageKey` |
//! | `movie.binary` | `MovieEntity{version, params, sprites}` (protobuf), zlib-compressed: the players inflate it themselves |

use effectcraft_project::Comp;
use effectcraft_project::render_queue::Channels;

use crate::pipeline::Cx;
use crate::{ExportError, Result, State, batch_size, pngenc, zip};

/// The `MovieEntity.version` players check for; 2.x supports ZIP images as entries.
const VERSION: &str = "2.0.0";

/// The most frames one container is written with: `movie.binary` needs one key frame per frame
/// per sprite, so the sprites' size grows with the square of the length (3000 frames is 9 M key
/// frames, ~63 MB before deflate).
pub(crate) const MAX_FRAMES: u32 = 3000;

/// Protobuf wire type of a varint field.
const VARINT: u8 = 0;
/// Protobuf wire type of a length-delimited field (strings, bytes, nested messages).
const LEN: u8 = 2;
/// Protobuf wire type of a 32-bit field (floats).
const FIXED32: u8 = 5;

fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn tag(out: &mut Vec<u8>, field: u32, wire: u8) {
    varint(out, ((field as u64) << 3) | wire as u64);
}

fn float_field(out: &mut Vec<u8>, field: u32, v: f32) {
    tag(out, field, FIXED32);
    out.extend_from_slice(&v.to_le_bytes());
}

fn int_field(out: &mut Vec<u8>, field: u32, v: u64) {
    tag(out, field, VARINT);
    varint(out, v);
}

fn bytes_field(out: &mut Vec<u8>, field: u32, data: &[u8]) {
    tag(out, field, LEN);
    varint(out, data.len() as u64);
    out.extend_from_slice(data);
}

fn string_field(out: &mut Vec<u8>, field: u32, s: &str) {
    bytes_field(out, field, s.as_bytes());
}

/// `movie.binary`: the canvas, the frame rate, and one sprite per frame (`images` are their
/// `imageKey`s, in order).
fn movie(width: u32, height: u32, fps: u32, images: &[String]) -> Vec<u8> {
    let mut params = Vec::new();
    float_field(&mut params, 1, width as f32); // viewBoxWidth
    float_field(&mut params, 2, height as f32); // viewBoxHeight
    int_field(&mut params, 3, fps.max(1) as u64); // fps
    int_field(&mut params, 4, images.len() as u64); // frames

    // FrameEntity{alpha: 1.0} and FrameEntity{alpha: 0.0}: field 2 of SpriteEntity, length 5,
    // then field 1 (alpha) as a fixed32.
    const OPAQUE: [u8; 7] = [0x12, 0x05, 0x0D, 0x00, 0x00, 0x80, 0x3F];
    const CLEAR: [u8; 7] = [0x12, 0x05, 0x0D, 0x00, 0x00, 0x00, 0x00];

    let keyframes = images.len().saturating_mul(OPAQUE.len());
    let mut body = Vec::with_capacity(64 + keyframes);
    string_field(&mut body, 1, VERSION);
    bytes_field(&mut body, 2, &params);
    let mut sprite = Vec::with_capacity(16 + keyframes);
    for (i, key) in images.iter().enumerate() {
        sprite.clear();
        string_field(&mut sprite, 1, key);
        for j in 0..images.len() {
            sprite.extend_from_slice(if i == j { &OPAQUE } else { &CLEAR });
        }
        bytes_field(&mut body, 4, &sprite);
    }
    body
}

/// The frame rate SVGA stores: a whole number of frames a second (a fractional rate is rounded,
/// the container has no field for 29.97).
fn fps_of(num: i64, den: i64) -> u32 {
    let den = den.max(1);
    let fps = (num as f64 / den as f64).round();
    if !fps.is_finite() || fps < 1.0 { 1 } else { (fps.min(120.0)) as u32 }
}

/// Render the job's frames into an SVGA container at `path`.
pub(crate) async fn svga(cx: &Cx<'_>, comp: &Comp, w: u32, h: u32, st: &mut State<'_>) -> Result<crate::Report> {
    let frames = u32::try_from(st.total()).map_err(|_| ExportError::Unsupported("this animation has too many frames for SVGA".into()))?;
    if frames == 0 {
        return Err(ExportError::Unsupported("an SVGA container needs at least one frame".into()));
    }
    if frames > MAX_FRAMES {
        return Err(ExportError::Unsupported(format!(
            "SVGA frame-by-frame export holds up to {MAX_FRAMES} frames (this render has {frames}): shorten the work area or render a movie"
        )));
    }
    let colour = if cx.output.channels == Channels::Rgba { pngenc::Colour::Rgba } else { pngenc::Colour::Rgb };
    let rate = cx.settings.rate(comp);
    let fps = fps_of(rate.num, rate.den);
    let path = cx.place(cx.path, 1);
    let file = cx.create(&path)?;
    let mut archive = zip::Zip::new(file);
    let mut images: Vec<String> = Vec::with_capacity(frames as usize);
    let batch = batch_size();
    let mut i = 0;
    while i < st.total() {
        let end = (i + batch).min(st.total());
        let pngs: Vec<Result<Vec<u8>>> = cx
            .frames(comp, (i..end).collect(), |_, img| {
                let px = cx.pixels_with(&img, comp, cx.output.channels, w, h, false);
                pngenc::still(&px, w, h, colour)
            })
            .await;
        for (k, png) in pngs.into_iter().enumerate() {
            let name = format!("frame_{:05}.png", i.saturating_add(k as u64));
            archive.add(&name, &png?)?;
            images.push(name);
        }
        st.advance(end - i)?;
        i = end;
    }
    // The players inflate `movie.binary` themselves (`pako.Inflate` in svgaplayerweb, which
    // throws "incorrect header check" on a bare protobuf), so its bytes are a zlib stream and the
    // archive entry that carries it is stored as it is.
    let proto = movie(w, h, fps, &images);
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(9));
    std::io::Write::write_all(&mut encoder, &proto).map_err(crate::io)?;
    let packed = encoder.finish().map_err(crate::io)?;
    archive.add("movie.binary", &packed)?;
    let file = archive.finish()?;
    let bytes = file.finish()?;
    Ok(crate::Report { path, frames: 0, width: w, height: h, seconds: 0.0, bytes, audio: false, log: None, overflow: vec![] })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a protobuf message into `(field, wire type, value)` triples, where a length-delimited
    /// value is its bytes.
    fn fields(mut b: &[u8]) -> Vec<(u32, u8, Vec<u8>)> {
        let mut out = Vec::new();
        let read_varint = |b: &mut &[u8]| -> u64 {
            let mut v = 0u64;
            let mut shift = 0;
            loop {
                let byte = b[0];
                *b = &b[1..];
                v |= ((byte & 0x7F) as u64) << shift;
                if byte & 0x80 == 0 {
                    return v;
                }
                shift += 7;
            }
        };
        while !b.is_empty() {
            let key = read_varint(&mut b);
            let (field, wire) = ((key >> 3) as u32, (key & 7) as u8);
            let value = match wire {
                0 => {
                    let start = b;
                    let _ = read_varint(&mut b);
                    let consumed = start.len().saturating_sub(b.len());
                    start[..consumed].to_vec()
                }
                2 => {
                    let n = read_varint(&mut b) as usize;
                    let v = b[..n].to_vec();
                    b = &b[n..];
                    v
                }
                5 => {
                    let v = b[..4].to_vec();
                    b = &b[4..];
                    v
                }
                w => panic!("unexpected wire type {w}"),
            };
            out.push((field, wire, value));
        }
        out
    }

    /// A varint's value from its own bytes.
    fn varint_value(v: &[u8]) -> u64 {
        let mut out = 0u64;
        let mut shift = 0;
        for byte in v {
            out |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        out
    }

    fn f32_of(v: &[u8]) -> f32 {
        f32::from_le_bytes([v[0], v[1], v[2], v[3]])
    }

    #[test]
    fn movie_describes_the_canvas_and_one_sprite_a_frame() {
        let images: Vec<String> = (0..3).map(|i| format!("frame_{i:05}.png")).collect();
        let m = movie(640, 360, 25, &images);
        let top = fields(&m);
        assert_eq!(top[0].0, 1, "version");
        assert_eq!(top[0].2, VERSION.as_bytes());
        let params = fields(&top[1].2);
        assert_eq!(f32_of(&params[0].2), 640.0);
        assert_eq!(f32_of(&params[1].2), 360.0);
        assert_eq!(varint_value(&params[2].2), 25);
        assert_eq!(varint_value(&params[3].2), 3, "frames");

        let sprites: Vec<_> = top.iter().filter(|f| f.0 == 4).collect();
        assert_eq!(sprites.len(), 3, "one sprite a frame");
        for (i, sprite) in sprites.iter().enumerate() {
            let s = fields(&sprite.2);
            assert_eq!(s[0].2, images[i].as_bytes().to_vec(), "the sprite's image");
            let frames: Vec<_> = s.iter().filter(|f| f.0 == 2).collect();
            assert_eq!(frames.len(), 3, "one key frame per movie frame");
            for (j, kf) in frames.iter().enumerate() {
                let alpha = f32_of(&fields(&kf.2)[0].2);
                assert_eq!(alpha, if i == j { 1.0 } else { 0.0 }, "sprite {i}, key frame {j}");
            }
        }
    }

    #[test]
    fn odd_frame_rates_round_to_whole_numbers() {
        assert_eq!(fps_of(30, 1), 30);
        assert_eq!(fps_of(30_000, 1001), 30);
        assert_eq!(fps_of(24_000, 1001), 24);
        assert_eq!(fps_of(0, 0), 1);
        assert_eq!(fps_of(600, 1), 120);
    }
}
