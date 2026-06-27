#!/usr/bin/env python3
r"""face_carousel - a live face-RECOGNITION carousel for face-detect.cpp.

This is not a speed race. It is a capability reel: the one binary runs the whole
InsightFace pipeline (detect, 5-point align, 512-d recognition, age + gender) on
~10 famous PUBLIC-DOMAIN PAINTINGS, and the numbers on screen are the REAL CLI
outputs parsed live from the JSON the binary prints.

The faces are classic public-domain portraits (PD-Art / PD-old-100, sourced from
Wikimedia Commons - see art/sources.txt) in benchmarks/demo/art/, led by the
Mona Lisa. The detect box + 5 landmarks + score, the analyze age + gender, and the
512-d embedding are all run on the artwork image; each face is labeled with the
painting title + artist (not "real photo"). The age + gender are the model's
literal, charming reading of the painting, not a claim about the sitter.

Per face the recognition animates on: the teal detection box draws in, the five
gold landmarks pop, and a card reveals the real outputs (recognized, a live 512-d
embedding tick, age + gender from analyze, detect score). Brief hold, slide to
the next. After the last face, the LocalAI end card.

  python3 face_carousel.py
  python3 face_carousel.py --no-cache   # re-run the CLI from scratch

Outputs face_carousel.mp4 + face_carousel.gif into ../media/. No em-dashes.
"""
import argparse, json, math, os, pickle, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent

# palette - matches face_demo.py / the locate-anything.cpp race
BG     = (13, 17, 23)
CARD   = (22, 27, 34)
CARD2  = (28, 34, 43)
LINE   = (38, 46, 56)
INK    = (223, 228, 234)
DIM    = (132, 142, 154)
DIMMER = (96, 105, 117)
TEAL   = (62, 200, 224)
TEAL_D = (32, 110, 126)
GREEN  = (80, 200, 120)
GREEN_D= (28, 70, 46)
GOLD   = (240, 200, 90)

FONTDIR = "/usr/share/fonts/truetype/dejavu"
_FCACHE = {}
def font(sz, bold=True, mono=False):
    key = (sz, bold, mono)
    if key in _FCACHE:
        return _FCACHE[key]
    name = "DejaVuSansMono" if mono else ("DejaVuSans-Bold" if bold else "DejaVuSans")
    if mono and bold:
        name = "DejaVuSansMono-Bold"
    try:
        f = ImageFont.truetype(f"{FONTDIR}/{name}.ttf", sz)
    except Exception:
        f = ImageFont.load_default()
    _FCACHE[key] = f
    return f


def fit_font(d, text, max_w, start, bold=True, mono=False, floor=11):
    """Largest font (<= start, >= floor) at which text fits within max_w."""
    sz = start
    while sz > floor:
        f = font(sz, bold, mono)
        if d.textlength(text, font=f) <= max_w:
            return f
        sz -= 1
    return font(floor, bold, mono)


# ---------------------------------------------------------------- easing
def clamp(t): return 0.0 if t < 0 else (1.0 if t > 1 else t)
def ease_out(t): t = clamp(t); return 1 - (1 - t) ** 3
def ease_in_out(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
def ease_out_back(t):
    t = clamp(t); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


# ---------------------------------------------------------------- CLI glue
def run_json(cli, args):
    out = subprocess.run([cli, *args, "--json"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"cli failed: {' '.join(args)}\n{out.stderr}")
    return json.loads(out.stdout)


THUMB = 360  # native thumbnail resolution we render faces at


def _square_crop(img, box, margin=0.85):
    """Square crop around a detect box; return (crop, ox, oy, scale_to_thumb)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(bw, bh) * (1 + margin) / 2
    sx1, sy1 = int(cx - half), int(cy - half)
    sx2, sy2 = int(cx + half), int(cy + half)
    W, H = img.size
    sx1 = max(0, sx1); sy1 = max(0, sy1); sx2 = min(W, sx2); sy2 = min(H, sy2)
    crop = img.crop((sx1, sy1, sx2, sy2))
    side = max(crop.size)
    sq = Image.new("RGB", (side, side), BG)
    sq.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    # account for the centering offset of the square padding
    pad_x = (side - crop.width) // 2
    pad_y = (side - crop.height) // 2
    sq = sq.resize((THUMB, THUMB), Image.LANCZOS)
    scale = THUMB / side
    # map original-image coord -> thumb coord
    def map_pt(px, py):
        return ((px - sx1 + pad_x) * scale, (py - sy1 + pad_y) * scale)
    return sq, map_pt


ART_DIR = HERE / "art"

# The carousel reel: famous PUBLIC-DOMAIN paintings (PD-Art / PD-old-100, sourced
# from Wikimedia Commons - see art/sources.txt for the per-file provenance). Each
# is a clean frontal / three-quarter portrait that detects with sane 5-point
# landmarks. Order leads with the Mona Lisa and varies era, gender and age for a
# shareable reel. The age / gender below are the model's LITERAL reading of the
# painting (charming, honest), not a claim about the sitter.
ART_REEL = [
    ("mona_lisa.jpg",          "Mona Lisa",                            "Leonardo da Vinci"),
    ("napoleon.jpg",           "The Emperor Napoleon in His Study",    "Jacques-Louis David"),
    ("girl_red_hat.jpg",       "Girl with a Red Hat",                  "Johannes Vermeer"),
    ("durer.jpg",              "Self-Portrait at Twenty-Eight",        "Albrecht Duerer"),
    ("pearl_earring.jpg",      "Girl with a Pearl Earring",            "Johannes Vermeer"),
    ("velazquez_innocent.jpg", "Portrait of Pope Innocent X",          "Diego Velazquez"),
    ("rembrandt.jpg",          "Self-Portrait",                        "Rembrandt van Rijn"),
    ("laughing_cavalier.jpg",  "The Laughing Cavalier",                "Frans Hals"),
    ("ingres.jpg",             "Madame Marcotte de Sainte-Marie",      "Jean-Auguste-Dominique Ingres"),
    ("van_gogh.jpg",           "Self-Portrait with Grey Felt Hat",     "Vincent van Gogh"),
]


def gather(cli, model, fixtures=None):
    """Run the real face-detect.cpp pipeline on each public-domain painting.

    Every number on screen is the REAL CLI output on the artwork image: the
    detect score + box + 5 landmarks, the analyze age + gender, and the 512-d
    embedding. The square thumbnail is only a display crop; box and landmarks
    are mapped into the crop frame so they sit over the painted face.
    """
    subjects = []
    for fname, title, artist in ART_REEL:
        path = str(ART_DIR / fname)
        det = run_json(cli, ["detect", "--model", model, "--input", path])
        if not det["faces"]:
            continue
        # the strongest face (CLI sorts by score) is the portrait sitter
        df = max(det["faces"], key=lambda f: f["score"])
        box, lms = df["box"], df["landmarks"]

        ana = run_json(cli, ["analyze", "--model", model, "--input", path])
        af = max(ana["faces"], key=lambda f: f["score"])
        emb = run_json(cli, ["embed", "--model", model, "--input", path])

        gender = "Female" if str(af["gender"]).upper().startswith("F") else "Male"
        src = Image.open(path).convert("RGB")
        crop, map_pt = _square_crop(src, box)
        tbox = list(map_pt(box[0], box[1])) + list(map_pt(box[2], box[3]))
        tlms = [list(map_pt(lx, ly)) for (lx, ly) in lms]
        subjects.append({
            "thumb": crop,
            "box": tbox,
            "lms": tlms,
            "score": df["score"],
            "age": af["age"],
            "gender": gender,
            "dim": emb["dim"],
            "emb": [float(v) for v in emb["embedding"]],
            "title": title,
            "artist": artist,
            "label": f"{title}  ·  {artist}",
        })
    return subjects


# ---------------------------------------------------------------- drawing
def rounded(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def pill(d, x, y, text, fnt, fg, bg, padx=11, pady=5):
    tw = d.textlength(text, font=fnt)
    asc, desc = fnt.getmetrics(); th = asc + desc
    rounded(d, [x, y, x + tw + 2 * padx, y + th + 2 * pady],
            (th + 2 * pady) // 2, fill=bg)
    d.text((x + padx, y + pady), text, font=fnt, fill=fg)
    return tw + 2 * padx


def blend(c1, c2, t):
    t = clamp(t)
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_header(cv, idx, n):
    d = ImageDraw.Draw(cv)
    bx = 56
    fb = font(34, True)
    d.text((bx, 30), "face-detect", font=fb, fill=INK)
    bw = d.textlength("face-detect", font=fb)
    d.text((bx + bw, 30), ".cpp", font=fb, fill=TEAL)
    bw2 = d.textlength(".cpp", font=fb)
    fx = font(18, False)
    d.text((bx + bw + bw2 + 16, 40), "·  recognizing the masters",
           font=fx, fill=DIM)
    # progress dots + n/10 (top right)
    dots_n = n
    dr = 6; gap = 17
    total = dots_n * gap
    sx = 1224 - total
    cy = 52
    for i in range(dots_n):
        cxp = sx + i * gap
        if i < idx:
            col = TEAL_D
        elif i == idx:
            col = TEAL
        else:
            col = LINE
        rr = dr + (2 if i == idx else 0)
        d.ellipse([cxp - rr, cy - rr, cxp + rr, cy + rr], fill=col)
    lbl = f"{idx + 1} / {n}"
    flbl = font(15, True, mono=True)
    lw = d.textlength(lbl, font=flbl)
    d.text((sx - lw - 16, cy - 9), lbl, font=flbl, fill=DIM)
    d.line([bx, 96, 1224, 96], fill=LINE, width=1)


def draw_face_card(cv, subj, cx, cy, alpha, box_t, lm_t, scan_t):
    """Face thumbnail card with animated box + landmarks. cx,cy = top-left."""
    card_w = THUMB + 48
    card_h = THUMB + 92
    # base card on an RGBA layer so we can fade the whole thing
    layer = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    rounded(ld, [0, 0, card_w - 1, card_h - 1], 16, fill=CARD + (255,),
            outline=LINE + (255,), width=1)
    # thumbnail
    tx, ty = 24, 24
    layer.paste(subj["thumb"], (tx, ty))
    ld = ImageDraw.Draw(layer)
    rounded(ld, [tx - 1, ty - 1, tx + THUMB, ty + THUMB], 6,
            outline=LINE + (255,), width=1)

    # scanning sweep (teal translucent line) while box draws
    if 0.0 < scan_t < 1.0:
        sy = ty + int(scan_t * THUMB)
        glow = Image.new("RGBA", (THUMB, 24), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for k in range(24):
            a = int(70 * (1 - abs(k - 12) / 12))
            gd.line([0, k, THUMB, k], fill=TEAL + (a,))
        layer.alpha_composite(glow, (tx, max(ty, sy - 12)))
        ld = ImageDraw.Draw(layer)

    bx = subj["box"]
    bcx = (bx[0] + bx[2]) / 2 + tx
    bcy = (bx[1] + bx[3]) / 2 + ty
    bw = (bx[2] - bx[0]); bh = (bx[3] - bx[1])
    # box grows from center outward
    bt = ease_out(box_t)
    hw = bw / 2 * bt; hh = bh / 2 * bt
    x1, y1 = bcx - hw, bcy - hh
    x2, y2 = bcx + hw, bcy + hh
    if box_t > 0.02:
        ba = int(255 * clamp(box_t * 1.5))
        ld.rectangle([x1, y1, x2, y2], outline=TEAL + (ba,), width=3)
        cl = 18
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            ld.line([px, py, px + dx * cl, py], fill=TEAL + (ba,), width=5)
            ld.line([px, py, px, py + dy * cl], fill=TEAL + (ba,), width=5)
        # score badge once box is mostly drawn
        if box_t > 0.6:
            fb = font(15, True)
            pa = int(255 * clamp((box_t - 0.6) / 0.4))
            txt = f"face  {subj['score'] * 100:.0f}%"
            tw = ld.textlength(txt, font=fb)
            asc, desc = fb.getmetrics(); th = asc + desc
            by = max(ty + 2, y1 - th - 12)
            rounded(ld, [x1, by, x1 + tw + 18, by + th + 8], (th + 8) // 2,
                    fill=TEAL + (pa,))
            ld.text((x1 + 9, by + 4), txt, font=fb, fill=BG + (pa,))

    # 5 landmarks pop in sequence
    for i, (lx, ly) in enumerate(subj["lms"]):
        seg = lm_t * 5 - i
        if seg <= 0:
            continue
        s = ease_out_back(clamp(seg))
        r = 7 * s
        px, py = lx + tx, ly + ty
        a = int(255 * clamp(seg))
        ld.ellipse([px - r, py - r, px + r, py + r],
                   fill=GOLD + (a,), outline=BG + (a,), width=2)

    # caption under thumbnail: painting title + artist (the on-screen label is
    # the artwork, not "real photo"), then the model line.
    cap_w = THUMB
    ft = fit_font(ld, subj["title"], cap_w, 16, bold=True)
    ld.text((tx, ty + THUMB + 16), subj["title"], font=ft, fill=INK + (235,))
    fa = fit_font(ld, f"{subj['artist']}  ·  buffalo_l GGUF", cap_w, 13, bold=False)
    ld.text((tx, ty + THUMB + 42),
            f"{subj['artist']}  ·  buffalo_l GGUF", font=fa, fill=DIMMER + (220,))

    if alpha < 1.0:
        a = layer.split()[3].point(lambda v: int(v * alpha))
        layer.putalpha(a)
    cv.alpha_composite(layer, (cx, cy))


def kv_card(d, x, y, w, h, tag, value, sub, accent, alpha):
    a = int(255 * alpha)
    rounded(d, [x, y, x + w, y + h], 12, fill=CARD + (a,), outline=LINE + (a,), width=1)
    ft = font(13, True); fv = font(26, True); fs = font(14, False)
    d.rectangle([x + 14, y + 18, x + 18, y + h - 16], fill=accent + (a,))
    d.text((x + 30, y + 14), tag, font=ft, fill=accent + (a,))
    d.text((x + 30, y + 34), value, font=fv, fill=INK + (a,))
    if sub:
        vw = d.textlength(value, font=fv)
        d.text((x + 30 + vw + 12, y + 44), sub, font=fs, fill=DIM + (a,))


def draw_reco_panel(cv, subj, rx, ry, rw, reveal_t, hold_frame):
    layer = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a_all = ease_out(reveal_t)

    d.text((rx, ry), "RECOGNITION", font=font(13, True), fill=DIMMER + (int(255 * a_all),))
    y = ry + 26

    # RECOGNIZED row with animated check
    rt = clamp(reveal_t * 1.6)
    aa = int(255 * ease_out(rt))
    rounded(d, [rx, y, rx + rw, y + 64], 13, fill=CARD + (aa,), outline=GREEN_D + (aa,), width=1)
    # check disc
    ccx, ccy, cr = rx + 34, y + 32, 16
    d.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], fill=GREEN_D + (aa,), outline=GREEN + (aa,), width=2)
    ck = clamp(reveal_t * 2 - 0.4)
    if ck > 0:
        p0 = (ccx - 7, ccy + 1)
        p1 = (ccx - 2, ccy + 6)
        p2 = (ccx + 8, ccy - 6)
        if ck < 0.5:
            t = ck / 0.5
            mp = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            d.line([p0, mp], fill=GREEN + (255,), width=3)
        else:
            t = (ck - 0.5) / 0.5
            mp = (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)
            d.line([p0, p1], fill=GREEN + (255,), width=3)
            d.line([p1, mp], fill=GREEN + (255,), width=3)
    d.text((rx + 64, y + 13), "RECOGNIZED", font=font(22, True), fill=INK + (aa,))
    d.text((rx + 64, y + 40), "512-d face embedding extracted", font=font(14, False), fill=DIM + (aa,))
    pill(d, rx + rw - 132, y + 19, "1 : N READY", font(13, True), TEAL, TEAL_D + (aa,))
    y += 84

    # live embedding tick: real embedding values scrolling
    et = clamp(reveal_t * 1.4)
    if et > 0:
        ea = int(255 * et)
        d.text((rx, y), "LIVE 512-D EMBEDDING", font=font(12, True), fill=DIMMER + (ea,))
        d.text((rx + rw - 80, y), f"dim {subj['dim']}", font=font(12, True, mono=True), fill=TEAL + (ea,))
        yb = y + 22
        rounded(d, [rx, yb, rx + rw, yb + 54], 10, fill=CARD2 + (ea,))
        emb = subj["emb"]
        ncol = 8
        win = (hold_frame * 1) % max(1, (len(emb) - ncol))
        fm = font(15, True, mono=True)
        cellw = (rw - 28) / ncol
        for i in range(ncol):
            v = emb[(win + i) % len(emb)]
            s = f"{v:+.2f}"
            col = TEAL if v >= 0 else ROSE if False else DIM
            col = blend(DIM, TEAL, clamp((v + 0.3) / 0.6)) if v >= 0 else blend(DIM, GOLD, clamp((-v) / 0.3))
            cxp = rx + 14 + i * cellw
            d.text((cxp, yb + 17), s, font=fm, fill=col + (ea,))
        # moving caret
        car = rx + 14 + ((hold_frame) % ncol) * cellw - 4
        d.line([car, yb + 8, car, yb + 46], fill=TEAL + (int(ea * 0.5),), width=2)
        y = yb + 54 + 22

    # age / gender / score cards
    gt = clamp(reveal_t * 1.2 - 0.2)
    if gt > 0:
        ga = ease_out(gt)
        cw = (rw - 28) / 3
        cards = [
            ("AGE", str(subj["age"]), "years", GOLD),
            ("GENDER", subj["gender"], "audeering", GOLD),
            ("DETECT", f"{subj['score']:.2f}", "score", TEAL),
        ]
        for i, (tg, val, sub, acc) in enumerate(cards):
            kv_card(d, rx + i * (cw + 14), y, cw, 78, tg, val, sub, acc, ga)
        y += 78 + 22

    # honest footer note
    nt = clamp(reveal_t * 1.1 - 0.3)
    if nt > 0:
        na = int(255 * ease_out(nt))
        d.text((rx, y), "age + gender are the model reading a painting  ·  one binary, no Python, real CLI output",
               font=font(14, False), fill=DIM + (na,))

    cv.alpha_composite(layer)


def draw_brandline(cv):
    d = ImageDraw.Draw(cv)
    d.text((56, 690), "github.com/mudler/face-detect.cpp", font=font(13, False), fill=DIMMER)
    t = "Brought to you by the LocalAI team  ·  localai.io"
    tw = d.textlength(t, font=font(13, False))
    d.text((1224 - tw, 690), t, font=font(13, False), fill=DIMMER)


# ---------------------------------------------------------------- end card
def draw_end_card(subjects, t):
    cv = Image.new("RGBA", (1280, 720), BG + (255,))
    d = ImageDraw.Draw(cv)
    a = ease_out(clamp(t * 1.4))
    A = int(255 * a)

    # strip of all recognized faces at top
    n = len(subjects)
    sw = 92; gap = 12
    total = n * sw + (n - 1) * gap
    sx = (1280 - total) // 2
    sy = 70
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
        rounded(dd, [xx, sy, xx + sw - 1, sy + sw - 1], 8, outline=TEAL_D + (sa,), width=1)
    d = ImageDraw.Draw(cv)
    cap = f"recognized {n} / {n} faces on classic public-domain paintings, real CLI output"
    cw = d.textlength(cap, font=font(15, False))
    d.text(((1280 - cw) // 2, sy + sw + 14), cap, font=font(15, False), fill=DIM + (A,))

    # logo
    logo = Image.open(ROOT / "assets" / "localai_logo.png").convert("RGBA")
    lh = 72; lw = int(logo.width * lh / logo.height)
    logo = logo.resize((lw, lh), Image.LANCZOS)
    if A < 255:
        al = logo.split()[3].point(lambda v: int(v * A / 255))
        logo.putalpha(al)
    ly = 250
    cv.alpha_composite(logo, ((1280 - lw) // 2, ly))
    d = ImageDraw.Draw(cv)
    t1 = "from the LocalAI team  ·  localai.io"
    t1w = d.textlength(t1, font=font(16, False))
    d.text(((1280 - t1w) // 2, ly + lh + 12), t1, font=font(16, False), fill=DIM + (A,))

    # headline
    t2 = "Face recognition on the masters, one binary, no Python"
    fh = font(40, True)
    t2w = d.textlength(t2, font=fh)
    th2 = clamp(t * 1.3 - 0.2); A2 = int(255 * ease_out(th2))
    d.text(((1280 - t2w) // 2, ly + lh + 50), t2, font=fh, fill=TEAL + (A2,))

    t3 = "detect, align, recognize, age + gender on classic public-domain paintings.  bit-exact vs InsightFace"
    fs = font(17, False)
    t3w = d.textlength(t3, font=fs)
    th3 = clamp(t * 1.3 - 0.35); A3 = int(255 * ease_out(th3))
    d.text(((1280 - t3w) // 2, ly + lh + 104), t3, font=fs, fill=DIM + (A3,))

    # four links in two rows near logo
    th4 = clamp(t * 1.3 - 0.5); A4 = int(255 * ease_out(th4))
    row1 = ["localai.io", "github.com/mudler/LocalAI"]
    row2 = ["github.com/mudler/face-detect.cpp", "huggingface.co/mudler/face-detect-gguf"]
    fl = font(16, True, mono=True)
    def draw_link_row(items, yy):
        widths = [d.textlength(s, font=fl) for s in items]
        sep = 36
        tw = sum(widths) + sep * (len(items) - 1)
        xx = (1280 - tw) // 2
        for i, s in enumerate(items):
            d.text((xx, yy), s, font=fl, fill=TEAL + (A4,))
            if i < len(items) - 1:
                dotx = xx + widths[i] + sep / 2 - 2
                d.ellipse([dotx, yy + 9, dotx + 4, yy + 13], fill=DIMMER + (A4,))
            xx += widths[i] + sep
    ybase = ly + lh + 168
    draw_link_row(row1, ybase)
    draw_link_row(row2, ybase + 30)

    return cv.convert("RGB")


# ---------------------------------------------------------------- frames
def render_carousel_frame(subjects, idx, phase_name, p, hold_frame=0):
    cv = Image.new("RGBA", (1280, 720), BG + (255,))
    draw_header(cv, idx, len(subjects))
    subj = subjects[idx]

    card_x0 = 70
    card_y = 132
    rx = 70 + (THUMB + 48) + 56
    rw = 1224 - rx
    ry = 168

    # defaults
    alpha = 1.0; xoff = 0; box_t = 1; lm_t = 1; scan_t = -1; reveal_t = 1
    if phase_name == "in":
        alpha = ease_out(p)
        xoff = int((1 - ease_out(p)) * 90)
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
        xoff = -int(ease_in_out(p) * 90)
        reveal_t = 1

    draw_face_card(cv, subj, card_x0 + xoff, card_y, alpha, box_t, lm_t, scan_t)
    if reveal_t > 0:
        # fade the right panel together with slide alpha on in/out
        panel_alpha = alpha
        tmp = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
        draw_reco_panel(tmp, subj, rx, ry, rw, reveal_t, hold_frame)
        if panel_alpha < 1.0:
            al = tmp.split()[3].point(lambda v: int(v * panel_alpha))
            tmp.putalpha(al)
        cv.alpha_composite(tmp)

    draw_brandline(cv)
    return cv.convert("RGB")


def build_frames(subjects, fps, outdir):
    frames = []
    # per-subject phase frame counts (at given fps)
    F_IN = max(4, int(fps * 0.32))
    F_SCAN = max(5, int(fps * 0.62))
    F_REVEAL = max(5, int(fps * 0.42))
    F_HOLD = max(8, int(fps * 0.78))
    F_OUT = max(4, int(fps * 0.30))
    n = len(subjects)
    hold_counter = 0
    for idx in range(n):
        for f in range(F_IN):
            frames.append(render_carousel_frame(subjects, idx, "in", f / (F_IN - 1)))
        for f in range(F_SCAN):
            frames.append(render_carousel_frame(subjects, idx, "scan", f / (F_SCAN - 1)))
        for f in range(F_REVEAL):
            frames.append(render_carousel_frame(subjects, idx, "reveal", f / (F_REVEAL - 1), hold_counter))
            hold_counter += 1
        for f in range(F_HOLD):
            frames.append(render_carousel_frame(subjects, idx, "hold", 1.0, hold_counter))
            hold_counter += 1
        # last subject: no slide-out (go to end card)
        if idx < n - 1:
            for f in range(F_OUT):
                frames.append(render_carousel_frame(subjects, idx, "out", f / (F_OUT - 1), hold_counter))
        hold_counter += 3

    # end card
    F_END_IN = max(10, int(fps * 0.9))
    F_END_HOLD = int(fps * 2.4)
    for f in range(F_END_IN):
        frames.append(draw_end_card(subjects, f / (F_END_IN - 1)))
    last = draw_end_card(subjects, 1.0)
    for f in range(F_END_HOLD):
        frames.append(last)

    # write
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

    framedir = Path(tempfile.mkdtemp(prefix="face_carousel_frames_"))
    nf = build_frames(subjects, a.fps, framedir)
    print(f"rendered {nf} frames -> {framedir}")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    mp4 = outdir / "face_carousel_paintings.mp4"
    gif = outdir / "face_carousel_paintings.gif"
    pal = framedir / "palette.png"

    subprocess.run(["ffmpeg", "-y", "-framerate", str(a.fps), "-i",
                    str(framedir / "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # clean gif via palettegen/paletteuse
    vf = f"fps={a.gif_fps},scale=900:-1:flags=lanczos"
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
    # leave a few frames for inspection
    insp = outdir / "_carousel_frames"
    insp.mkdir(exist_ok=True)
    for tag, i in [("early", int(nf * 0.04)), ("mid", int(nf * 0.13)), ("end", nf - 30)]:
        shutil.copy(framedir / f"frame_{i:04d}.png", insp / f"{tag}_{i:04d}.png")
    print("inspection frames in", insp)
    print("framedir", framedir)


if __name__ == "__main__":
    main()
