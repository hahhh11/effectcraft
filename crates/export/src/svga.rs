//! SVGA (`.svga`) 2.0: the whole file is one zlib stream wrapping a protobuf `MovieEntity`.
//!
//! This is the form every 2.x player reads: `svgaplayerweb` inflates a non-ZIP file and decodes
//! it (`load_viaProto`), the duiba share-page parser does exactly the same, and the native iOS /
//! Android SDKs inflate the file and call `MovieEntity.parseFrom`. Neither of them opens a ZIP.
//! The 1.x form — a ZIP holding `movie.spec` plus one PNG per frame — is read only by players
//! that ship JSZip, which is why an earlier ZIP-based export failed on those pages.
//!
//! A 2.0 `MovieEntity` carries its pictures **inside** the protobuf: `images` is a map from an
//! image key to the PNG's bytes. The rest is the canvas, frame rate and one sprite per frame.
//! A composited render has no vector structure left to keep, so it becomes one sprite per frame,
//! each with one key frame per movie frame and alpha 1 on its own frame only: at any moment one
//! sprite is opaque and the rest are clear. That costs `frames²` key frames, which bounds the
//! export to [`MAX_FRAMES`].
//!
//! | Protobuf field | Holds |
//! |---|---|
//! | `1 version` | `"2.0.0"` |
//! | `2 params` | `MovieParams{viewBoxWidth, viewBoxHeight, fps, frames}` |
//! | `3 images` | map of image key → PNG bytes, one per frame |
//! | `4 sprites` | `SpriteEntity{imageKey, frames}` — alpha 1 on its own frame |

use std::io::Write as _;

use effectcraft_project::Comp;
use effectcraft_project::render_queue::Channels;

use crate::pipeline::Cx;
use crate::{ExportError, Result, State, batch_size, pngenc};

/// The `MovieEntity.version` players check for.
const VERSION: &str = "2.0.0";

/// The most frames one file is written with: every sprite has one key frame per movie frame, so
/// the protobuf grows with the square of the length (600 frames is 360 k key frames, ~11 MB
/// before zlib; the embedded PNGs are much larger).
pub(crate) const MAX_FRAMES: u32 = 600;

/// Protobuf wire type of a varint field.
const VARINT: u8 = 0;
/// Protobuf wire type of a length-delimited field (strings, bytes, nested messages, map entries).
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

/// A `Layout` covering the whole canvas at the origin.
fn layout(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(20);
    float_field(&mut out, 1, 0.0); // x
    float_field(&mut out, 2, 0.0); // y
    float_field(&mut out, 3, width as f32);
    float_field(&mut out, 4, height as f32);
    out
}

/// One `FrameEntity`: its alpha and a layout over the whole canvas. `transform` is left out —
/// every player defaults it to the identity matrix — which keeps the `frames²` block small.
fn keyframe(alpha: f32, width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    float_field(&mut out, 1, alpha);
    bytes_field(&mut out, 2, &layout(width, height));
    out
}

/// The protobuf `MovieEntity`: canvas, frame rate, the pictures themselves and one sprite per
/// frame. `images` is `(key, PNG bytes)` so no ZIP or sidecar files are needed.
fn movie(width: u32, height: u32, fps: u32, images: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut params = Vec::with_capacity(18);
    float_field(&mut params, 1, width as f32); // viewBoxWidth
    float_field(&mut params, 2, height as f32); // viewBoxHeight
    int_field(&mut params, 3, fps.max(1) as u64);
    int_field(&mut params, 4, images.len() as u64); // frames

    let n = images.len();
    let clear = keyframe(0.0, width, height);
    let opaque = keyframe(1.0, width, height);
    let mut out = Vec::with_capacity(64 + images.iter().map(|(_, p)| p.len() + 32).sum::<usize>() + n * n * (clear.len() + 4));
    string_field(&mut out, 1, VERSION);
    bytes_field(&mut out, 2, &params);
    for (key, png) in images {
        // A protobuf map is a repeated message of {1: key, 2: value}.
        let mut entry = Vec::with_capacity(key.len() + png.len() + 16);
        string_field(&mut entry, 1, key);
        bytes_field(&mut entry, 2, png);
        bytes_field(&mut out, 3, &entry);
    }
    for (i, (key, _)) in images.iter().enumerate() {
        let mut sprite = Vec::with_capacity(16 + n * (clear.len() + 4));
        string_field(&mut sprite, 1, key);
        for j in 0..n {
            bytes_field(&mut sprite, 2, if i == j { &opaque } else { &clear });
        }
        bytes_field(&mut out, 4, &sprite);
    }
    out
}

/// The frame rate SVGA stores: a whole number of frames a second (a fractional rate is rounded,
/// the format has no field for 29.97).
fn fps_of(num: i64, den: i64) -> u32 {
    let den = den.max(1);
    let fps = (num as f64 / den as f64).round();
    if !fps.is_finite() || fps < 1.0 { 1 } else { (fps.min(120.0)) as u32 }
}

/// Render the job's frames into a zlib-compressed SVGA 2.0 file at `path`.
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
    let batch = batch_size();
    // The pictures are embedded in the protobuf, so they are held until the movie is assembled.
    let mut images: Vec<(String, Vec<u8>)> = Vec::with_capacity(frames as usize);
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
            images.push((format!("frame_{:05}", i.saturating_add(k as u64)), png?));
        }
        st.advance(end - i)?;
        i = end;
    }
    let proto = movie(w, h, fps, &images);
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(9));
    encoder.write_all(&proto).map_err(crate::io)?;
    let packed = encoder.finish().map_err(crate::io)?;
    let mut file = cx.create(&path)?;
    file.write_all(&packed).map_err(crate::io)?;
    let bytes = file.finish()?;
    Ok(crate::Report { path, frames: 0, width: w, height: h, seconds: 0.0, bytes, audio: false, log: None, overflow: vec![] })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a protobuf message into `(field, wire type, value)` triples, where a length-delimited
    /// value is its bytes.
    fn fields(mut b: &[u8]) -> Vec<(u32, u8, Vec<u8>)> {
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
        let mut out = Vec::new();
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

    fn sample_images(n: usize) -> Vec<(String, Vec<u8>)> {
        (0..n).map(|i| (format!("frame_{i:05}"), vec![0x89, b'P', b'N', b'G', i as u8])).collect()
    }

    #[test]
    fn movie_carries_its_pictures_and_one_sprite_a_frame() {
        let images = sample_images(3);
        let m = movie(640, 360, 25, &images);
        let top = fields(&m);
        assert_eq!(top[0].0, 1, "version");
        assert_eq!(top[0].2, VERSION.as_bytes());
        let params = fields(&top[1].2);
        assert_eq!(f32_of(&params[0].2), 640.0, "viewBoxWidth");
        assert_eq!(f32_of(&params[1].2), 360.0, "viewBoxHeight");
        assert_eq!(varint_value(&params[2].2), 25, "fps");
        assert_eq!(varint_value(&params[3].2), 3, "frames");

        // Field 3 is the map: each entry is {1: key, 2: PNG bytes}.
        let map: Vec<_> = top.iter().filter(|f| f.0 == 3).collect();
        assert_eq!(map.len(), 3, "one picture a frame, embedded");
        for (i, entry) in map.iter().enumerate() {
            let e = fields(&entry.2);
            assert_eq!(e[0].0, 1, "map key");
            assert_eq!(e[0].2, images[i].0.as_bytes().to_vec());
            assert_eq!(e[1].0, 2, "map value");
            assert_eq!(e[1].2, images[i].1, "the PNG bytes themselves");
        }

        let sprites: Vec<_> = top.iter().filter(|f| f.0 == 4).collect();
        assert_eq!(sprites.len(), 3, "one sprite a frame");
        for (i, sprite) in sprites.iter().enumerate() {
            let s = fields(&sprite.2);
            assert_eq!(s[0].2, images[i].0.as_bytes().to_vec(), "the sprite's imageKey");
            let frames: Vec<_> = s.iter().filter(|f| f.0 == 2).collect();
            assert_eq!(frames.len(), 3, "one key frame per movie frame");
            for (j, kf) in frames.iter().enumerate() {
                let k = fields(&kf.2);
                assert_eq!(f32_of(&k[0].2), if i == j { 1.0 } else { 0.0 }, "sprite {i}, key frame {j}");
                let l = fields(&k[1].2);
                assert_eq!(f32_of(&l[2].2), 640.0, "layout.width");
                assert_eq!(f32_of(&l[3].2), 360.0, "layout.height");
            }
        }
    }

    #[test]
    fn odd_frame_rates_round_to_whole_numbers() {
        assert_eq!(fps_of(30, 1), 30);
        assert_eq!(fps_of(30_000, 1001), 30, "29.97 rounds to 30");
        assert_eq!(fps_of(24_000, 1001), 24);
        assert_eq!(fps_of(0, 0), 1, "never zero");
        assert_eq!(fps_of(600, 1), 120, "clamped");
    }

    #[test]
    fn varints_encode_large_values() {
        for n in [0u64, 1, 127, 128, 300, u32::MAX as u64] {
            let mut b = Vec::new();
            varint(&mut b, n);
            assert_eq!(varint_value(&b), n);
        }
    }
}
