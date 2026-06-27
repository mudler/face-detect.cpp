#!/usr/bin/env python3
r"""face_carousel_square - the 1:1 social/mobile cut of the face-detect.cpp reel.

Same live capability story as face_carousel.py (the one binary runs the whole
InsightFace pipeline on ~10 public-domain paintings, real CLI output parsed live), re-laid for
a square 1080x1080 frame: the face crop stacks on top and the recognition card
(RECOGNIZED + live 512-d tick + age / gender / detect) sits below it, with the
same header + "n / 10" progress. It ends on a square LocalAI card.

It reuses face_carousel.py for the CLI gather + cache, palette, easing, fonts and
the recognition panel, so the numbers stay identical to the 16:9 cut. The square
re-lay (header, stacked face/reco, end card) lives here. No em-dashes.

  python3 face_carousel_square.py
  python3 face_carousel_square.py --no-cache   # re-run the CLI from scratch

Outputs face_carousel_square.mp4 + face_carousel_square.gif into ../media/.
"""
import argparse, pickle, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw

import face_carousel as fc
from face_carousel import (
    BG, CARD, LINE, INK, DIM, DIMMER, TEAL, TEAL_D, GOLD,
    font, fit_font, clamp, ease_out, ease_in_out, ease_out_back,
    rounded, pill, draw_reco_panel, THUMB, gather,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent

SW = 1080            # square side
MARGIN = 60
RIGHT = SW - MARGIN  # 1020


# ---------------------------------------------------------------- header
def draw_header_sq(cv, idx, n):
    d = ImageDraw.Draw(cv)
    bx = MARGIN
    fb = font(32, True)
    d.text((bx, 32), "face-detect", font=fb, fill=INK)
    bw = d.textlength("face-detect", font=fb)
    d.text((bx + bw, 32), ".cpp", font=fb, fill=TEAL)
    bw2 = d.textlength(".cpp", font=fb)
    d.text((bx + bw + bw2 + 14, 42), "·  recognizing the masters",
           font=font(17, False), fill=DIM)
    # progress dots + n/10 (top right)
    dr = 6; gap = 17
    total = n * gap
    sx = RIGHT - total
    cy = 52
    for i in range(n):
        cxp = sx + i * gap
        col = TEAL_D if i < idx else (TEAL if i == idx else LINE)
        rr = dr + (2 if i == idx else 0)
        d.ellipse([cxp - rr, cy - rr, cxp + rr, cy + rr], fill=col)
    lbl = f"{idx + 1} / {n}"
    flbl = font(15, True, mono=True)
    lw = d.textlength(lbl, font=flbl)
    d.text((sx - lw - 16, cy - 9), lbl, font=flbl, fill=DIM)
    d.line([bx, 100, RIGHT, 100], fill=LINE, width=1)


# ---------------------------------------------------------------- face card
def draw_face_card_sq(cv, subj, cx, cy, TS, alpha, box_t, lm_t, scan_t):
    """Stacked face thumbnail card, scaled to TS px, animated box + landmarks."""
    sc = TS / THUMB
    pad = 24
    card_w = TS + 2 * pad
    card_h = TS + 96
    layer = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    rounded(ld, [0, 0, card_w - 1, card_h - 1], 18, fill=CARD + (255,),
            outline=LINE + (255,), width=1)
    tx, ty = pad, pad
    thumb = subj["thumb"].resize((TS, TS), Image.LANCZOS)
    layer.paste(thumb, (tx, ty))
    ld = ImageDraw.Draw(layer)
    rounded(ld, [tx - 1, ty - 1, tx + TS, ty + TS], 8,
            outline=LINE + (255,), width=1)

    # scanning sweep while box draws
    if 0.0 < scan_t < 1.0:
        sy = ty + int(scan_t * TS)
        glow = Image.new("RGBA", (TS, 28), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for k in range(28):
            a = int(70 * (1 - abs(k - 14) / 14))
            gd.line([0, k, TS, k], fill=TEAL + (a,))
        layer.alpha_composite(glow, (tx, max(ty, sy - 14)))
        ld = ImageDraw.Draw(layer)

    bx = subj["box"]
    bcx = (bx[0] + bx[2]) / 2 * sc + tx
    bcy = (bx[1] + bx[3]) / 2 * sc + ty
    bw = (bx[2] - bx[0]) * sc; bh = (bx[3] - bx[1]) * sc
    bt = ease_out(box_t)
    hw = bw / 2 * bt; hh = bh / 2 * bt
    x1, y1 = bcx - hw, bcy - hh
    x2, y2 = bcx + hw, bcy + hh
    if box_t > 0.02:
        ba = int(255 * clamp(box_t * 1.5))
        ld.rectangle([x1, y1, x2, y2], outline=TEAL + (ba,), width=3)
        cl = 20
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            ld.line([px, py, px + dx * cl, py], fill=TEAL + (ba,), width=5)
            ld.line([px, py, px, py + dy * cl], fill=TEAL + (ba,), width=5)
        if box_t > 0.6:
            fb = font(17, True)
            pa = int(255 * clamp((box_t - 0.6) / 0.4))
            txt = f"face  {subj['score'] * 100:.0f}%"
            tw = ld.textlength(txt, font=fb)
            asc, desc = fb.getmetrics(); th = asc + desc
            by = max(ty + 2, y1 - th - 12)
            rounded(ld, [x1, by, x1 + tw + 20, by + th + 8], (th + 8) // 2,
                    fill=TEAL + (pa,))
            ld.text((x1 + 10, by + 4), txt, font=fb, fill=BG + (pa,))

    # 5 landmarks pop in sequence
    for i, (lx, ly) in enumerate(subj["lms"]):
        seg = lm_t * 5 - i
        if seg <= 0:
            continue
        s = ease_out_back(clamp(seg))
        r = 8 * s
        px, py = lx * sc + tx, ly * sc + ty
        a = int(255 * clamp(seg))
        ld.ellipse([px - r, py - r, px + r, py + r],
                   fill=GOLD + (a,), outline=BG + (a,), width=2)

    # caption under thumbnail: painting title + artist, then the model line.
    ftt = fit_font(ld, subj["title"], TS, 18, bold=True)
    ld.text((tx, ty + TS + 18), subj["title"], font=ftt, fill=INK + (235,))
    faa = fit_font(ld, f"{subj['artist']}  ·  buffalo_l GGUF", TS, 14, bold=False)
    ld.text((tx, ty + TS + 48), f"{subj['artist']}  ·  buffalo_l GGUF",
            font=faa, fill=DIMMER + (220,))

    if alpha < 1.0:
        a = layer.split()[3].point(lambda v: int(v * alpha))
        layer.putalpha(a)
    cv.alpha_composite(layer, (cx, cy))


def draw_brandline_sq(cv):
    d = ImageDraw.Draw(cv)
    fb = font(13, False)
    d.text((MARGIN, 1042), "github.com/mudler/face-detect.cpp", font=fb, fill=DIMMER)
    t = "Brought to you by the LocalAI team  ·  localai.io"
    tw = d.textlength(t, font=fb)
    d.text((RIGHT - tw, 1042), t, font=fb, fill=DIMMER)


# ---------------------------------------------------------------- frame
TS_FACE = 400
CARD_Y = 124


def render_square_frame(subjects, idx, phase_name, p, hold_frame=0):
    cv = Image.new("RGBA", (SW, SW), BG + (255,))
    draw_header_sq(cv, idx, len(subjects))
    subj = subjects[idx]

    card_w = TS_FACE + 48
    card_h = TS_FACE + 96
    card_x = (SW - card_w) // 2

    rx = MARGIN
    rw = SW - 2 * MARGIN
    ry = CARD_Y + card_h + 34

    alpha = 1.0; xoff = 0; box_t = 1; lm_t = 1; scan_t = -1; reveal_t = 1
    if phase_name == "in":
        alpha = ease_out(p)
        xoff = int((1 - ease_out(p)) * 70)
        box_t = 0; lm_t = 0; reveal_t = 0
    elif phase_name == "scan":
        scan_t = p
        box_t = ease_out(clamp(p * 1.3))
        lm_t = clamp(p * 1.2 - 0.5)
        reveal_t = 0
    elif phase_name == "reveal":
        box_t = 1; lm_t = 1
        reveal_t = p
    elif phase_name == "hold":
        box_t = 1; lm_t = 1; reveal_t = 1
    elif phase_name == "out":
        alpha = 1 - ease_in_out(p)
        xoff = -int(ease_in_out(p) * 70)
        reveal_t = 1

    draw_face_card_sq(cv, subj, card_x + xoff, CARD_Y, TS_FACE,
                      alpha, box_t, lm_t, scan_t)
    if reveal_t > 0:
        panel_alpha = alpha
        tmp = Image.new("RGBA", (SW, SW), (0, 0, 0, 0))
        draw_reco_panel(tmp, subj, rx, ry, rw, reveal_t, hold_frame)
        if panel_alpha < 1.0:
            al = tmp.split()[3].point(lambda v: int(v * panel_alpha))
            tmp.putalpha(al)
        cv.alpha_composite(tmp)

    draw_brandline_sq(cv)
    return cv.convert("RGB")


# ---------------------------------------------------------------- end card
def _fit_font(d, text, max_w, start, bold):
    sz = start
    while sz > 14:
        f = font(sz, bold)
        if d.textlength(text, font=f) <= max_w:
            return f
        sz -= 1
    return font(14, bold)


def draw_end_card_sq(subjects, t):
    cv = Image.new("RGBA", (SW, SW), BG + (255,))
    d = ImageDraw.Draw(cv)
    a = ease_out(clamp(t * 1.4))
    A = int(255 * a)

    # strip of all recognized faces near the top
    n = len(subjects)
    sw = 88; gap = 10
    total = n * sw + (n - 1) * gap
    sx = (SW - total) // 2
    sy = 120
    for i, s in enumerate(subjects):
        st = clamp(t * 2.2 - i * 0.05)
        if st <= 0:
            continue
        sa = int(255 * ease_out(st))
        th = s["thumb"].resize((sw, sw), Image.LANCZOS).convert("RGBA")
        if sa < 255:
            al = th.split()[3].point(lambda v: int(v * sa / 255))
            th.putalpha(al)
        xx = sx + i * (sw + gap)
        cv.alpha_composite(th, (xx, sy))
        dd = ImageDraw.Draw(cv)
        rounded(dd, [xx, sy, xx + sw - 1, sy + sw - 1], 8,
                outline=TEAL_D + (sa,), width=1)
    d = ImageDraw.Draw(cv)
    cap = f"recognized {n} / {n} faces on public-domain paintings, real CLI output"
    cw = d.textlength(cap, font=font(16, False))
    d.text(((SW - cw) // 2, sy + sw + 16), cap, font=font(16, False), fill=DIM + (A,))

    # logo, vertically centred in the lower block
    logo = Image.open(ROOT / "assets" / "localai_logo.png").convert("RGBA")
    lh = 88; lw = int(logo.width * lh / logo.height)
    logo = logo.resize((lw, lh), Image.LANCZOS)
    if A < 255:
        al = logo.split()[3].point(lambda v: int(v * A / 255))
        logo.putalpha(al)
    ly = 460
    cv.alpha_composite(logo, ((SW - lw) // 2, ly))
    d = ImageDraw.Draw(cv)

    t1 = "from the LocalAI team  ·  localai.io"
    t1w = d.textlength(t1, font=font(17, False))
    d.text(((SW - t1w) // 2, ly + lh + 14), t1, font=font(17, False), fill=DIM + (A,))

    # headline (teal), auto-fit to width
    t2 = "Face recognition on the masters, one binary, no Python"
    fh = _fit_font(d, t2, SW - 2 * MARGIN, 34, True)
    t2w = d.textlength(t2, font=fh)
    th2 = clamp(t * 1.3 - 0.2); A2 = int(255 * ease_out(th2))
    hy = ly + lh + 54
    d.text(((SW - t2w) // 2, hy), t2, font=fh, fill=TEAL + (A2,))

    # subline on two readable rows
    subs = ["detect, align, recognize, age + gender on classic paintings.",
            "bit-exact vs InsightFace, one binary, no Python"]
    fs = font(16, False)
    th3 = clamp(t * 1.3 - 0.35); A3 = int(255 * ease_out(th3))
    sy3 = hy + 50
    for i, line in enumerate(subs):
        lwd = d.textlength(line, font=fs)
        d.text(((SW - lwd) // 2, sy3 + i * 28), line, font=fs, fill=DIM + (A3,))

    # four links in two centred rows near the bottom
    th4 = clamp(t * 1.3 - 0.5); A4 = int(255 * ease_out(th4))
    row1 = ["localai.io", "github.com/mudler/LocalAI"]
    row2 = ["github.com/mudler/face-detect.cpp", "huggingface.co/mudler/face-detect-gguf"]
    fl = font(16, True, mono=True)

    def draw_link_row(items, yy):
        widths = [d.textlength(s, font=fl) for s in items]
        sep = 34
        tw = sum(widths) + sep * (len(items) - 1)
        xx = (SW - tw) // 2
        for i, s in enumerate(items):
            d.text((xx, yy), s, font=fl, fill=TEAL + (A4,))
            if i < len(items) - 1:
                dotx = xx + widths[i] + sep / 2 - 2
                d.ellipse([dotx, yy + 9, dotx + 4, yy + 13], fill=DIMMER + (A4,))
            xx += widths[i] + sep

    ybase = sy3 + 2 * 28 + 34
    draw_link_row(row1, ybase)
    draw_link_row(row2, ybase + 32)

    return cv.convert("RGB")


# ---------------------------------------------------------------- build
def build_frames(subjects, fps, outdir):
    frames = []
    F_IN = max(4, int(fps * 0.32))
    F_SCAN = max(5, int(fps * 0.62))
    F_REVEAL = max(5, int(fps * 0.42))
    F_HOLD = max(8, int(fps * 0.78))
    F_OUT = max(4, int(fps * 0.30))
    n = len(subjects)
    hold_counter = 0
    for idx in range(n):
        for f in range(F_IN):
            frames.append(render_square_frame(subjects, idx, "in", f / (F_IN - 1)))
        for f in range(F_SCAN):
            frames.append(render_square_frame(subjects, idx, "scan", f / (F_SCAN - 1)))
        for f in range(F_REVEAL):
            frames.append(render_square_frame(subjects, idx, "reveal", f / (F_REVEAL - 1), hold_counter))
            hold_counter += 1
        for f in range(F_HOLD):
            frames.append(render_square_frame(subjects, idx, "hold", 1.0, hold_counter))
            hold_counter += 1
        if idx < n - 1:
            for f in range(F_OUT):
                frames.append(render_square_frame(subjects, idx, "out", f / (F_OUT - 1), hold_counter))
        hold_counter += 3

    F_END_IN = max(10, int(fps * 0.9))
    F_END_HOLD = int(fps * 2.4)
    for f in range(F_END_IN):
        frames.append(draw_end_card_sq(subjects, f / (F_END_IN - 1)))
    last = draw_end_card_sq(subjects, 1.0)
    for f in range(F_END_HOLD):
        frames.append(last)

    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("frame_*.png"):
        old.unlink()
    for i, fr in enumerate(frames):
        fr.save(outdir / f"frame_{i:04d}.png")
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=str(ROOT / "build-cpu/examples/cli/facedetect-cli"))
    ap.add_argument("--model", default=str(ROOT / "models/buffalo_l-f16.gguf"))
    ap.add_argument("--fixtures", default=str(ROOT / "tests/fixtures"))
    ap.add_argument("--out", default=str(HERE.parent.parent / "media"))
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--gif-fps", type=int, default=13)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    cache = Path("/tmp/face_carousel_cache.pkl")
    if cache.exists() and not a.no_cache:
        with open(cache, "rb") as fh:
            subjects = pickle.load(fh)
        print(f"loaded {len(subjects)} subjects from cache")
    else:
        print("gathering real CLI outputs ...")
        subjects = gather(a.cli, a.model, a.fixtures)
        with open(cache, "wb") as fh:
            pickle.dump(subjects, fh)
        print(f"gathered {len(subjects)} subjects")

    framedir = Path(tempfile.mkdtemp(prefix="face_carousel_sq_frames_"))
    nf = build_frames(subjects, a.fps, framedir)
    print(f"rendered {nf} frames -> {framedir}")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    mp4 = outdir / "face_carousel_paintings_square.mp4"
    gif = outdir / "face_carousel_paintings_square.gif"
    pal = framedir / "palette.png"

    subprocess.run(["ffmpeg", "-y", "-framerate", str(a.fps), "-i",
                    str(framedir / "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vf = f"fps={a.gif_fps},scale=720:720:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(a.fps), "-i",
                    str(framedir / "frame_%04d.png"),
                    "-vf", f"{vf},palettegen=stats_mode=diff", str(pal)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(a.fps), "-i",
                    str(framedir / "frame_%04d.png"), "-i", str(pal),
                    "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    str(gif)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("wrote", mp4, "(", mp4.stat().st_size // 1024, "KB )")
    print("wrote", gif, "(", gif.stat().st_size // 1024, "KB )")
    insp = outdir / "_carousel_square_frames"
    insp.mkdir(exist_ok=True)
    for tag, i in [("early", int(nf * 0.04)), ("mid", int(nf * 0.13)), ("end", nf - 30)]:
        shutil.copy(framedir / f"frame_{i:04d}.png", insp / f"{tag}_{i:04d}.png")
    print("inspection frames in", insp)
    print("framedir", framedir)


if __name__ == "__main__":
    main()
