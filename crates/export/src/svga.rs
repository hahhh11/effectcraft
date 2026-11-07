//! SVGA (`.svga`): a ZIP container holding `movie.spec` — the 1.5 JSON description — and one PNG
//! per animation frame.
//!
//! `movie.spec` is what the players read out of a ZIP: `svgaplayerweb` parses it and never looks
//! inside a container for `movie.binary` (the 2.0 protobuf; it reads that only from bare, non-ZIP
//! files), and the native SDKs read 1.5 as well, so one layout serves every player.
//!
//! The format animates *sprites*: a sprite has one image and a list of key frames whose alpha,
//! transform and layout the player draws per movie frame (`sprite.frames[currentFrame]`). That is
//! why a composited render — which has no vector structure left to keep — becomes one sprite per
//! frame, each with one key frame per movie frame and alpha 1 on its own frame only: at any
//! moment one sprite is opaque and the rest are clear. The cost is `frames²` key frames, which
//! bounds the export to [`MAX_FRAMES`]; the JSON is repetitive, so the entry is deflated.
//!
//! | Entry | Holds |
//! |---|---|
//! | `frame_00000.png` … | one 8-bit RGBA (or RGB) PNG a frame, the sprite's image |
//! | `movie.spec` | `{params, images, sprites}` (JSON, deflated) |

use effectcraft_project::Comp;
use effectcraft_project::render_queue::Channels;

use crate::pipeline::Cx;
use crate::{ExportError, Result, State, batch_size, pngenc, zip};

/// The most frames one container is written with: `movie.spec` needs one key frame per movie
/// frame per sprite, so the JSON grows with the square of the length (600 frames is 360 k key
/// frames, ~50 MB before deflate).
pub(crate) const MAX_FRAMES: u32 = 600;

/// The frame rate SVGA stores: a whole number of frames a second (a fractional rate is rounded,
/// the format has no field for 29.97).
fn fps_of(num: i64, den: i64) -> u32 {
    let den = den.max(1);
    let fps = (num as f64 / den as f64).round();
    if !fps.is_finite() || fps < 1.0 { 1 } else { (fps.min(120.0)) as u32 }
}

/// One key frame of a sprite: opaque on its own frame, clear on every other one, laid out over
/// the whole canvas with an identity transform.
fn keyframe(out: &mut String, opaque: bool, w: u32, h: u32) {
    out.push_str(if opaque { r#"{"alpha":1,"# } else { r#"{"alpha":0,"# });
    out.push_str(r#""transform":{"a":1,"b":0,"c":0,"d":1,"tx":0,"ty":0},"layout":{"x":0,"y":0,"width":"#);
    push_num(out, w);
    out.push_str(r#","height":"#);
    push_num(out, h);
    out.push_str("}}");
}

fn push_num(out: &mut String, n: u32) {
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    let mut v = n;
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    out.push_str(std::str::from_utf8(&buf[i..]).unwrap_or_default());
}

/// `movie.spec`: the canvas, the frame rate, the image map (key → entry base name) and one sprite
/// per frame. Keys are the same in both places, which is what the players look up.
fn spec(w: u32, h: u32, fps: u32, images: &[String]) -> String {
    let n = images.len();
    let mut s = String::with_capacity(256 + n * 64 + n * n * 140);
    s.push_str(r#"{"params":{"viewBoxWidth":"#);
    push_num(&mut s, w);
    s.push_str(r#","viewBoxHeight":"#);
    push_num(&mut s, h);
    s.push_str(r#","fps":"#);
    push_num(&mut s, fps);
    s.push_str(r#","frames":"#);
    push_num(&mut s, n as u32);
    s.push_str(r#"},"images":{"#);
    for (i, key) in images.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        s.push_str(key);
        s.push_str(r#"":""#);
        s.push_str(key);
        s.push('"');
    }
    s.push_str(r#"},"sprites":["#);
    for (i, key) in images.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(r#"{"imageKey":""#);
        s.push_str(key);
        s.push_str(r#"","frames":["#);
        for j in 0..n {
            if j > 0 {
                s.push(',');
            }
            keyframe(&mut s, i == j, w, h);
        }
        s.push_str("]}");
    }
    s.push_str("]}");
    s
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
            let base = format!("frame_{:05}", i.saturating_add(k as u64));
            archive.add(&format!("{base}.png"), &png?)?;
            images.push(base);
        }
        st.advance(end - i)?;
        i = end;
    }
    archive.add_deflated("movie.spec", spec(w, h, fps, &images).as_bytes())?;
    let file = archive.finish()?;
    let bytes = file.finish()?;
    Ok(crate::Report { path, frames: 0, width: w, height: h, seconds: 0.0, bytes, audio: false, log: None, overflow: vec![] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_describes_the_canvas_and_one_sprite_a_frame() {
        let images: Vec<String> = (0..3).map(|i| format!("frame_{i:05}")).collect();
        let s = spec(640, 360, 25, &images);
        let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
        let params = &v["params"];
        assert_eq!(params["viewBoxWidth"], 640.0);
        assert_eq!(params["viewBoxHeight"], 360.0);
        assert_eq!(params["fps"], 25);
        assert_eq!(params["frames"], 3);
        // The image map keys are what the sprites point at, and the player appends ".png".
        assert_eq!(v["images"]["frame_00000"], "frame_00000");
        let sprites = v["sprites"].as_array().expect("sprites");
        assert_eq!(sprites.len(), 3, "one sprite a frame");
        for (i, sprite) in sprites.iter().enumerate() {
            assert_eq!(sprite["imageKey"], images[i]);
            let frames = sprite["frames"].as_array().expect("frames");
            assert_eq!(frames.len(), 3, "one key frame per movie frame");
            for (j, kf) in frames.iter().enumerate() {
                assert_eq!(kf["alpha"], if i == j { 1.0 } else { 0.0 }, "sprite {i}, frame {j}");
                assert_eq!(kf["layout"]["width"], 640.0);
                assert_eq!(kf["layout"]["height"], 360.0);
                assert_eq!(kf["transform"]["a"], 1.0);
                assert_eq!(kf["transform"]["tx"], 0.0);
            }
        }
    }

    #[test]
    fn the_frame_rate_is_whole() {
        assert_eq!(fps_of(30, 1), 30);
        assert_eq!(fps_of(30000, 1001), 30, "29.97 rounds to 30");
        assert_eq!(fps_of(24000, 1001), 24);
        assert_eq!(fps_of(0, 0), 1, "never zero");
    }

    #[test]
    fn numbers_survive_the_hand_written_json() {
        for n in [0u32, 7, 480, 1080, 4096, 65535, u32::MAX] {
            let mut s = String::new();
            push_num(&mut s, n);
            assert_eq!(s, n.to_string());
        }
    }
}
