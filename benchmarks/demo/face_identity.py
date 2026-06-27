#!/usr/bin/env python3
r"""face_identity - a 1:N identity-matching reel for face-detect.cpp.

This is not a speed race. It is a recognition-capability reel: the one binary
decides WHO a face is. A PROBE face sits on the left; a LINEUP of candidate
faces sits on the right. For every candidate the binary computes the REAL
cosine distance to the probe (facedetect-cli verify), and the numbers on screen
are the literal CLI output parsed live. A scan sweeps the lineup, each candidate
shows its real distance, then the single true match (lowest distance, below the
0.35 threshold) lights up GREEN "MATCH - same person" while the rest stay dim /
red "no match".

DATA (all public-domain U.S. federal photos, see tests/fixtures/PROVENANCE.md):
  - Probe: face_a.jpg (one identity).
  - The true match: face_b.jpg (same identity, second photo session).
  - Distractors: face_c.jpg (a different person) plus several distinct faces
    cropped from multi.jpg (a real group photo; detect gives ~11 face boxes).

Every distance is the real output of:
  FACEDETECT_DEVICE=cpu facedetect-cli verify --model <m> --a <probe> --b <cand>
which prints {"distance":..,"verified":..}. Multi.jpg crops are written to temp
jpegs and verified against the probe the same way.

  python3 face_identity.py
  python3 face_identity.py --no-cache    # re-run the CLI from scratch
  python3 face_identity.py --square      # also render the 1:1 square cut

Outputs face_identity.mp4 + face_identity.gif into ../media/. No em-dashes.
"""
import argparse, json, os, pickle, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# palette - matches face_demo.py / face_carousel.py / the locate-anything.cpp race
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
RED    = (224, 96, 96)
RED_D  = (74, 30, 34)

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


def fit_font(d, text, max_w, start, bold=True, mono=False, floor=10):
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
def blend(c1, c2, t):
    t = clamp(t)
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ---------------------------------------------------------------- CLI glue
def run_json(cli, args, env):
    out = subprocess.run([cli, *args], capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"cli failed: {' '.join(args)}\n{out.stderr}")
    return json.loads(out.stdout)


def _square_crop(img, box, margin=0.85, side_px=360):
    """Square crop around a detect box, returned at side_px resolution."""
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
    return sq.resize((side_px, side_px), Image.LANCZOS)


def _strongest(det):
    return max(det["faces"], key=lambda f: f["score"])


# Lineup composition. The single true match is face_b (same identity as the
# probe). Distractors are face_c (a different person) plus distinct faces cropped
# from the group photo multi.jpg. MULTI_PICKS are indices into the multi.jpg
# faces sorted by box area (descending); index 0 is skipped on purpose because it
# is the same identity as the probe inside the group shot (a second true match),
# which would defeat the "exactly one match" gate.
MULTI_PICKS = [1, 2, 3, 6]


def gather(cli, model, fixtures):
    env = dict(os.environ, FACEDETECT_DEVICE="cpu")
    fx = Path(fixtures)
    probe_path = str(fx / "face_a.jpg")
    tmpdir = Path(tempfile.mkdtemp(prefix="face_identity_crops_"))

    # probe thumbnail (display only) from its strongest face
    pdet = run_json(cli, ["detect", "--model", model, "--input", probe_path, "--json"], env)
    probe_thumb = _square_crop(Image.open(probe_path).convert("RGB"), _strongest(pdet)["box"])

    def verify(cand_path):
        v = run_json(cli, ["verify", "--model", model, "--a", probe_path, "--b", cand_path], env)
        return float(v["distance"]), bool(v["verified"])

    cands = []

    # the true match + the named distractor, straight from the fixtures
    for fname, tag in [("face_b.jpg", "match"), ("face_c.jpg", "distractor")]:
        p = str(fx / fname)
        det = run_json(cli, ["detect", "--model", model, "--input", p, "--json"], env)
        thumb = _square_crop(Image.open(p).convert("RGB"), _strongest(det)["box"])
        dist, ver = verify(p)
        cands.append({"thumb": thumb, "dist": dist, "verified": ver, "src": fname})

    # distractor crops from the real group photo
    multi_path = str(fx / "multi.jpg")
    mdet = run_json(cli, ["detect", "--model", model, "--input", multi_path, "--json"], env)
    mimg = Image.open(multi_path).convert("RGB")
    faces = sorted(mdet["faces"], key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
    for k in MULTI_PICKS:
        f = faces[k]
        thumb = _square_crop(mimg, f["box"])
        # write the display crop to a temp jpg and verify THAT exact image, so the
        # number on screen is the real distance of the face we show.
        cp = tmpdir / f"multi_{k}.jpg"
        thumb.save(cp, quality=95)
        dist, ver = verify(str(cp))
        cands.append({"thumb": thumb, "dist": dist, "verified": ver, "src": f"multi.jpg#{k}"})

    # GATE: exactly one verified match, and it is the lowest distance.
    matches = [c for c in cands if c["verified"]]
    if len(matches) != 1:
        raise RuntimeError(f"GATE FAILED: expected exactly 1 match, got {len(matches)}: "
                           + ", ".join(f"{c['src']}={c['dist']:.4f}/{c['verified']}" for c in cands))
    lowest = min(cands, key=lambda c: c["dist"])
    if not lowest["verified"]:
        raise RuntimeError("GATE FAILED: lowest-distance candidate is not the verified match")

    # display order: a fixed shuffle so the match is neither first nor last
    order = [0, 1, 2, 3, 4, 5]  # match=0 currently; move it to slot 3
    by_src = {c["src"]: c for c in cands}
    seq = ["multi.jpg#1", "face_c.jpg", "multi.jpg#2", "face_b.jpg", "multi.jpg#3", "multi.jpg#6"]
    cands = [by_src[s] for s in seq if s in by_src]
    return {"probe_thumb": probe_thumb, "cands": cands,
            "match_idx": next(i for i, c in enumerate(cands) if c["verified"]),
            "threshold": 0.35}


# ---------------------------------------------------------------- drawing prims
def rounded(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def pill(d, x, y, text, fnt, fg, bg, padx=11, pady=5):
    tw = d.textlength(text, font=fnt)
    asc, desc = fnt.getmetrics(); th = asc + desc
    rounded(d, [x, y, x + tw + 2 * padx, y + th + 2 * pady], (th + 2 * pady) // 2, fill=bg)
    d.text((x + padx, y + pady), text, font=fnt, fill=fg)
    return tw + 2 * padx


def draw_header(cv):
    d = ImageDraw.Draw(cv)
    bx = 56
    fb = font(34, True)
    d.text((bx, 30), "face-detect", font=fb, fill=INK)
    bw = d.textlength("face-detect", font=fb)
    d.text((bx + bw, 30), ".cpp", font=fb, fill=TEAL)
    bw2 = d.textlength(".cpp", font=fb)
    fx = font(18, False)
    d.text((bx + bw + bw2 + 16, 40), "·  identity match: find the same person", font=fx, fill=DIM)
    pill(d, 1224 - 150, 36, "1 : N  IDENTIFY", font(13, True), TEAL, TEAL_D)
    d.line([bx, 96, 1224, 96], fill=LINE, width=1)


def draw_brandline(cv):
    d = ImageDraw.Draw(cv)
    d.text((56, 690), "github.com/mudler/face-detect.cpp", font=font(13, False), fill=DIMMER)
    t = "Brought to you by the LocalAI team  ·  localai.io"
    tw = d.textlength(t, font=font(13, False))
    d.text((1224 - tw, 690), t, font=font(13, False), fill=DIMMER)


# ---------------------------------------------------------------- layout
PROBE_T = 300          # probe thumb size
PROBE_X = 56
PROBE_Y = 150
GRID_T  = 180          # candidate thumb size
GRID_COLS = 3
GRID_CARD_W = GRID_T + 24
GRID_CARD_H = GRID_T + 78
GRID_GX = 24
GRID_GY = 20
GRID_REGION_X = 440
GRID_REGION_W = 1224 - GRID_REGION_X
_grid_w = GRID_COLS * GRID_CARD_W + (GRID_COLS - 1) * GRID_GX
GRID_X0 = GRID_REGION_X + (GRID_REGION_W - _grid_w) // 2
GRID_Y0 = 132


def cell_xy(i):
    col = i % GRID_COLS
    row = i // GRID_COLS
    x = GRID_X0 + col * (GRID_CARD_W + GRID_GX)
    y = GRID_Y0 + row * (GRID_CARD_H + GRID_GY)
    return x, y


def probe_center():
    return (PROBE_X + 24 + PROBE_T / 2, PROBE_Y + 24 + PROBE_T / 2)


def cell_center(i):
    x, y = cell_xy(i)
    return (x + 12 + GRID_T / 2, y + 12 + GRID_T / 2)


def draw_probe(cv, state, alpha=1.0):
    layer = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cw, ch = PROBE_T + 48, PROBE_T + 96
    x, y = PROBE_X, PROBE_Y
    rounded(d, [x, y, x + cw, y + ch], 16, fill=CARD + (255,), outline=TEAL_D + (255,), width=2)
    layer.paste(state["probe_thumb"].resize((PROBE_T, PROBE_T), Image.LANCZOS), (x + 24, y + 24))
    d = ImageDraw.Draw(layer)
    rounded(d, [x + 23, y + 23, x + 24 + PROBE_T, y + 24 + PROBE_T], 6, outline=TEAL_D + (255,), width=1)
    pill(d, x + 24, y + 24 + 10, "PROBE", font(15, True), BG, TEAL)
    d.text((x + 24, y + 24 + PROBE_T + 16), "query face", font=font(19, True), fill=INK)
    d.text((x + 24, y + 24 + PROBE_T + 44), "buffalo_l GGUF  ·  512-d arcface", font=font(13, False), fill=DIMMER)
    if alpha < 1.0:
        a = layer.split()[3].point(lambda v: int(v * alpha))
        layer.putalpha(a)
    cv.alpha_composite(layer)


def draw_candidate(cv, state, i, computed_t, scanning, reveal_t, alpha=1.0):
    """One lineup candidate. computed_t: distance reveal 0..1. scanning: sweep on.
    reveal_t: 0..1 final verdict colorize."""
    c = state["cands"][i]
    is_match = (i == state["match_idx"])
    layer = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = cell_xy(i)
    tx, ty = x + 12, y + 12

    # verdict-driven accent
    if reveal_t > 0:
        if is_match:
            accent = blend(LINE, GREEN, ease_out(reveal_t))
            ow = 1 + int(2 * ease_out(reveal_t))
        else:
            accent = blend(LINE, RED_D, ease_out(reveal_t))
            ow = 1
    else:
        accent = TEAL if scanning else LINE
        ow = 2 if scanning else 1
    rounded(d, [x, y, x + GRID_CARD_W, y + GRID_CARD_H], 14, fill=CARD + (255,), outline=accent + (255,), width=ow)

    # thumbnail; non-match dims a touch on reveal
    thumb = c["thumb"].resize((GRID_T, GRID_T), Image.LANCZOS)
    if reveal_t > 0 and not is_match:
        from PIL import ImageEnhance
        thumb = ImageEnhance.Brightness(thumb).enhance(1 - 0.42 * ease_out(reveal_t))
    layer.paste(thumb, (tx, ty))
    d = ImageDraw.Draw(layer)
    rounded(d, [tx - 1, ty - 1, tx + GRID_T, ty + GRID_T], 6, outline=accent + (255,), width=1)

    # scanning sweep
    if scanning:
        glow = Image.new("RGBA", (GRID_T, 22), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for k in range(22):
            a = int(80 * (1 - abs(k - 11) / 11))
            gd.line([0, k, GRID_T, k], fill=TEAL + (a,))
        sy = ty + int((computed_t) * GRID_T)
        layer.alpha_composite(glow, (tx, max(ty, min(ty + GRID_T - 22, sy - 11))))
        d = ImageDraw.Draw(layer)

    # distance readout row under the thumbnail
    ry = ty + GRID_T + 12
    if computed_t > 0.02:
        ca = int(255 * clamp(computed_t * 1.4))
        dist = c["dist"]
        # distance bar (lower is closer): fill scales inverse to distance vs ~1.2 cap
        bar_x0, bar_x1 = tx, tx + GRID_T
        rounded(d, [bar_x0, ry, bar_x1, ry + 8], 4, fill=CARD2 + (ca,))
        frac = clamp(1 - dist / 1.2)
        if reveal_t > 0:
            barc = GREEN if is_match else RED
        else:
            barc = TEAL
        fillw = int((bar_x1 - bar_x0) * frac * clamp(computed_t))
        if fillw > 4:
            rounded(d, [bar_x0, ry, bar_x0 + fillw, ry + 8], 4, fill=barc + (ca,))
        # numeric distance
        fm = font(20, True, mono=True)
        dtxt = f"d {dist:.2f}"
        d.text((tx, ry + 16), dtxt, font=fm, fill=(INK if reveal_t == 0 else (GREEN if is_match else DIM)) + (ca,))
        # verdict at right
        if reveal_t > 0:
            va = int(255 * ease_out(reveal_t))
            if is_match:
                vt = "MATCH"; vc = GREEN; vb = GREEN_D
            else:
                vt = "no match"; vc = RED; vb = RED_D
            fv = font(14, True)
            vw = d.textlength(vt, font=fv)
            pill(d, tx + GRID_T - vw - 22, ry + 13, vt, fv, vc + (va,), vb + (va,))
        else:
            # during scan, show verified/under-threshold hint subtly
            fs = font(13, False)
            ht = "verified" if c["verified"] else "below thr" if False else ""

    if alpha < 1.0:
        a = layer.split()[3].point(lambda v: int(v * alpha))
        layer.putalpha(a)
    cv.alpha_composite(layer)


def draw_connector(cv, state, active_i, t):
    """Teal connector from probe to the candidate currently being scanned."""
    if active_i is None:
        return
    layer = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    px, py = probe_center()
    cx, cy = cell_center(active_i)
    # animate a dot travelling along the line
    tt = ease_in_out(t)
    dx, dy = px + (cx - px) * tt, py + (cy - py) * tt
    d.line([px, py, dx, dy], fill=TEAL + (140,), width=2)
    r = 6
    d.ellipse([dx - r, dy - r, dx + r, dy + r], fill=TEAL + (220,))
    d.ellipse([dx - r - 4, dy - r - 4, dx + r + 4, dy + r + 4], outline=TEAL + (90,), width=2)
    cv.alpha_composite(layer)


def draw_match_check(cv, state, t):
    """A green check disc over the matched candidate on reveal."""
    i = state["match_idx"]
    x, y = cell_xy(i)
    layer = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * ease_out(clamp(t * 1.4)))
    ccx, ccy, cr = x + GRID_CARD_W - 30, y + 30, int(20 * ease_out_back(clamp(t * 1.6)))
    if cr < 2:
        return
    d.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], fill=GREEN_D + (a,), outline=GREEN + (a,), width=3)
    ck = clamp(t * 2 - 0.5)
    if ck > 0:
        p0 = (ccx - 8, ccy + 1); p1 = (ccx - 2, ccy + 7); p2 = (ccx + 9, ccy - 7)
        if ck < 0.5:
            tt = ck / 0.5
            mp = (p0[0] + (p1[0] - p0[0]) * tt, p0[1] + (p1[1] - p0[1]) * tt)
            d.line([p0, mp], fill=GREEN + (255,), width=4)
        else:
            tt = (ck - 0.5) / 0.5
            mp = (p1[0] + (p2[0] - p1[0]) * tt, p1[1] + (p2[1] - p1[1]) * tt)
            d.line([p0, p1], fill=GREEN + (255,), width=4)
            d.line([p1, mp], fill=GREEN + (255,), width=4)
    cv.alpha_composite(layer)


def draw_verdict_banner(cv, state, t):
    """Bottom banner once decided: the green MATCH summary line."""
    c = state["cands"][state["match_idx"]]
    d = ImageDraw.Draw(cv)
    a = int(255 * ease_out(t))
    by = 632
    bx0, bx1 = PROBE_X, 1224
    rounded(d, [bx0, by, bx1, by + 44], 12, fill=CARD + (a,), outline=GREEN_D + (a,), width=1)
    d.rectangle([bx0 + 14, by + 12, bx0 + 18, by + 32], fill=GREEN + (a,))
    head = "MATCH  ·  same person"
    d.text((bx0 + 32, by + 11), head, font=font(20, True), fill=GREEN + (a,))
    hw = d.textlength(head, font=font(20, True))
    detail = f"d = {c['dist']:.2f}   verified   ·   threshold {state['threshold']:.2f}   ·   1 match in {len(state['cands'])} candidates"
    d.text((bx0 + 32 + hw + 24, by + 15), detail, font=font(15, False), fill=DIM + (a,))


# ---------------------------------------------------------------- end card
def draw_end_card(state, t, size=(1280, 720)):
    W, H = size
    cv = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(cv)
    a = ease_out(clamp(t * 1.4)); A = int(255 * a)

    # probe + arrow + matched face strip
    sw = 96
    probe = state["probe_thumb"].resize((sw, sw), Image.LANCZOS).convert("RGBA")
    match = state["cands"][state["match_idx"]]["thumb"].resize((sw, sw), Image.LANCZOS).convert("RGBA")
    gap = 70
    total = sw * 2 + gap
    sx = (W - total) // 2
    sy = 64
    st = clamp(t * 2.0)
    sa = int(255 * ease_out(st))
    for img, xx, lbl, col in [(probe, sx, "probe", TEAL), (match, sx + sw + gap, "match", GREEN)]:
        im = img.copy()
        if sa < 255:
            al = im.split()[3].point(lambda v: int(v * sa / 255)); im.putalpha(al)
        cv.alpha_composite(im, (xx, sy))
        dd = ImageDraw.Draw(cv)
        rounded(dd, [xx, sy, xx + sw - 1, sy + sw - 1], 8, outline=col + (sa,), width=2)
        lw = dd.textlength(lbl, font=font(13, True))
        dd.text((xx + (sw - lw) / 2, sy + sw + 8), lbl, font=font(13, True), fill=col + (sa,))
    # equals/arrow with distance between them
    d = ImageDraw.Draw(cv)
    mx, my = sx + sw + gap // 2, sy + sw // 2
    d.line([sx + sw + 8, my, sx + sw + gap - 8, my], fill=GREEN + (sa,), width=3)
    dist = state["cands"][state["match_idx"]]["dist"]
    dtxt = f"d={dist:.2f}"
    dw = d.textlength(dtxt, font=font(14, True, mono=True))
    d.text((mx - dw / 2, my - 30), dtxt, font=font(14, True, mono=True), fill=GREEN + (sa,))

    # logo
    logo = Image.open(ROOT / "assets" / "localai_logo.png").convert("RGBA")
    lh = 72; lw = int(logo.width * lh / logo.height)
    logo = logo.resize((lw, lh), Image.LANCZOS)
    if A < 255:
        al = logo.split()[3].point(lambda v: int(v * A / 255)); logo.putalpha(al)
    ly = 252
    cv.alpha_composite(logo, ((W - lw) // 2, ly))
    d = ImageDraw.Draw(cv)
    t1 = "from the LocalAI team  ·  localai.io"
    t1w = d.textlength(t1, font=font(16, False))
    d.text(((W - t1w) // 2, ly + lh + 12), t1, font=font(16, False), fill=DIM + (A,))

    # headline
    t2 = "Face recognition: 1-to-N identity matching, one binary, no Python"
    fh = font(38, True)
    t2w = d.textlength(t2, font=fh)
    if t2w > W - 80:
        fh = fit_font(d, t2, W - 80, 38, bold=True)
        t2w = d.textlength(t2, font=fh)
    th2 = clamp(t * 1.3 - 0.2); A2 = int(255 * ease_out(th2))
    d.text(((W - t2w) // 2, ly + lh + 50), t2, font=fh, fill=TEAL + (A2,))

    t3 = "real arcface cosine distance per candidate, decided against a 0.35 threshold.  real CLI output"
    fs3 = font(16, False)
    t3w = d.textlength(t3, font=fs3)
    if t3w > W - 80:
        fs3 = fit_font(d, t3, W - 80, 16, bold=False)
        t3w = d.textlength(t3, font=fs3)
    th3 = clamp(t * 1.3 - 0.35); A3 = int(255 * ease_out(th3))
    d.text(((W - t3w) // 2, ly + lh + 100), t3, font=fs3, fill=DIM + (A3,))

    # four links in two rows
    th4 = clamp(t * 1.3 - 0.5); A4 = int(255 * ease_out(th4))
    row1 = ["localai.io", "github.com/mudler/LocalAI"]
    row2 = ["github.com/mudler/face-detect.cpp", "huggingface.co/mudler/face-detect-gguf"]
    fl = font(16, True, mono=True)
    def draw_link_row(items, yy):
        widths = [d.textlength(s, font=fl) for s in items]
        sep = 36
        tw = sum(widths) + sep * (len(items) - 1)
        xx = (W - tw) // 2
        for i, s in enumerate(items):
            d.text((xx, yy), s, font=fl, fill=TEAL + (A4,))
            if i < len(items) - 1:
                dotx = xx + widths[i] + sep / 2 - 2
                d.ellipse([dotx, yy + 9, dotx + 4, yy + 13], fill=DIMMER + (A4,))
            xx += widths[i] + sep
    ybase = ly + lh + 156
    draw_link_row(row1, ybase)
    draw_link_row(row2, ybase + 30)
    return cv.convert("RGB")


# ---------------------------------------------------------------- frame compose
def render_frame(state, phase, p, scan_pos=0.0):
    cv = Image.new("RGBA", (1280, 720), BG + (255,))
    draw_header(cv)
    n = len(state["cands"])

    # phase defaults
    probe_a = 1.0
    cand_alpha = [1.0] * n
    computed = [0.0] * n
    scanning = [False] * n
    reveal_t = 0.0
    active = None

    if phase == "intro":
        probe_a = ease_out(clamp(p * 1.4))
        for i in range(n):
            ci = clamp(p * 2.4 - i * 0.16)
            cand_alpha[i] = ease_out(ci)
    elif phase == "scan":
        # scan_pos in [0, n]; integer part = fully computed, current = fractional
        for i in range(n):
            computed[i] = clamp(scan_pos - i)
        cur = int(scan_pos)
        if cur < n and (scan_pos - cur) > 0.001:
            scanning[cur] = True
            active = cur
    elif phase == "decide":
        for i in range(n):
            computed[i] = 1.0
        reveal_t = p
    elif phase == "hold":
        for i in range(n):
            computed[i] = 1.0
        reveal_t = 1.0

    draw_probe(cv, state, probe_a)
    for i in range(n):
        draw_candidate(cv, state, i, computed[i], scanning[i], reveal_t, cand_alpha[i])
    if phase == "scan" and active is not None:
        draw_connector(cv, state, active, scan_pos - int(scan_pos))
    if reveal_t > 0:
        draw_match_check(cv, state, reveal_t)
        draw_verdict_banner(cv, state, clamp(reveal_t * 1.3))

    draw_brandline(cv)
    return cv.convert("RGB")


def build_frames(state, fps, outdir, square=False):
    frames = []
    n = len(state["cands"])
    F_INTRO = max(8, int(fps * 0.7))
    F_PER = max(7, int(fps * 0.42))      # frames per candidate scan
    F_GAP = max(2, int(fps * 0.08))      # tiny pause after all scanned
    F_DECIDE = max(8, int(fps * 0.55))
    F_HOLD = int(fps * 2.4)

    for f in range(F_INTRO):
        frames.append(render_frame(state, "intro", f / (F_INTRO - 1)))
    # scan: march scan_pos from 0 to n
    total_scan = F_PER * n
    for f in range(total_scan):
        scan_pos = (f + 1) / F_PER
        scan_pos = min(scan_pos, n)
        frames.append(render_frame(state, "scan", 0, scan_pos))
    for f in range(F_GAP):
        frames.append(render_frame(state, "scan", 0, float(n)))
    for f in range(F_DECIDE):
        frames.append(render_frame(state, "decide", f / (F_DECIDE - 1)))
    for f in range(F_HOLD):
        frames.append(render_frame(state, "hold", 1.0))

    # end card
    F_END_IN = max(10, int(fps * 0.9))
    F_END_HOLD = int(fps * 2.6)
    for f in range(F_END_IN):
        frames.append(draw_end_card(state, f / (F_END_IN - 1)))
    last = draw_end_card(state, 1.0)
    for f in range(F_END_HOLD):
        frames.append(last)

    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("frame_*.png"):
        old.unlink()
    for i, fr in enumerate(frames):
        fr.save(outdir / f"frame_{i:04d}.png")
    return len(frames)


# ---------------------------------------------------------------- square variant
def render_frame_square(state, phase, p, scan_pos=0.0):
    """1:1 1080x1080: probe on top, 2x3 lineup below. Reuses the same state."""
    S = 1080
    cv = Image.new("RGBA", (S, S), BG + (255,))
    d = ImageDraw.Draw(cv)
    # header
    fb = font(40, True)
    d.text((48, 36), "face-detect", font=fb, fill=INK)
    bw = d.textlength("face-detect", font=fb)
    d.text((48 + bw, 36), ".cpp", font=fb, fill=TEAL)
    d.text((48, 88), "identity match  ·  find the same person", font=font(19, False), fill=DIM)
    pill(d, S - 184, 44, "1 : N  IDENTIFY", font(14, True), TEAL, TEAL_D)
    d.line([48, 128, S - 48, 128], fill=LINE, width=1)

    n = len(state["cands"])
    # probe centered top
    PT = 200
    px0 = (S - (PT + 48)) // 2
    py0 = 150
    probe_a = ease_out(clamp(p * 1.4)) if phase == "intro" else 1.0
    pl = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(pl)
    rounded(pd, [px0, py0, px0 + PT + 48, py0 + PT + 70], 16, fill=CARD + (255,), outline=TEAL_D + (255,), width=2)
    pl.paste(state["probe_thumb"].resize((PT, PT), Image.LANCZOS), (px0 + 24, py0 + 24))
    pd = ImageDraw.Draw(pl)
    pill(pd, px0 + 24, py0 + 34, "PROBE", font(15, True), BG, TEAL)
    pd.text((px0 + 24, py0 + 24 + PT + 12), "query face  ·  buffalo_l GGUF", font=font(14, False), fill=DIMMER)
    if probe_a < 1.0:
        al = pl.split()[3].point(lambda v: int(v * probe_a)); pl.putalpha(al)
    cv.alpha_composite(pl)

    # lineup grid 3 cols x 2 rows
    GT = 200
    CW, CH = GT + 24, GT + 70
    gx, gy = 28, 24
    gridw = 3 * CW + 2 * gx
    gx0 = (S - gridw) // 2
    gy0 = py0 + PT + 110

    def sq_cell(i):
        col = i % 3; row = i // 3
        return gx0 + col * (CW + gx), gy0 + row * (CH + gy)

    # compute phase fields (mirror render_frame)
    computed = [0.0] * n; scanning = [False] * n; reveal_t = 0.0; active = None; cand_a = [1.0] * n
    if phase == "intro":
        for i in range(n):
            cand_a[i] = ease_out(clamp(p * 2.4 - i * 0.16))
    elif phase == "scan":
        for i in range(n):
            computed[i] = clamp(scan_pos - i)
        cur = int(scan_pos)
        if cur < n and (scan_pos - cur) > 0.001:
            scanning[cur] = True; active = cur
    elif phase == "decide":
        computed = [1.0] * n; reveal_t = p
    elif phase == "hold":
        computed = [1.0] * n; reveal_t = 1.0

    for i in range(n):
        c = state["cands"][i]; is_match = (i == state["match_idx"])
        x, y = sq_cell(i); tx, ty = x + 12, y + 12
        layer = Image.new("RGBA", cv.size, (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
        if reveal_t > 0:
            accent = blend(LINE, GREEN, ease_out(reveal_t)) if is_match else blend(LINE, RED_D, ease_out(reveal_t))
            ow = 1 + int(2 * ease_out(reveal_t)) if is_match else 1
        else:
            accent = TEAL if scanning[i] else LINE; ow = 2 if scanning[i] else 1
        rounded(ld, [x, y, x + CW, y + CH], 14, fill=CARD + (255,), outline=accent + (255,), width=ow)
        thumb = c["thumb"].resize((GT, GT), Image.LANCZOS)
        if reveal_t > 0 and not is_match:
            from PIL import ImageEnhance
            thumb = ImageEnhance.Brightness(thumb).enhance(1 - 0.42 * ease_out(reveal_t))
        layer.paste(thumb, (tx, ty)); ld = ImageDraw.Draw(layer)
        rounded(ld, [tx - 1, ty - 1, tx + GT, ty + GT], 6, outline=accent + (255,), width=1)
        if scanning[i]:
            glow = Image.new("RGBA", (GT, 22), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
            for k in range(22):
                aa = int(80 * (1 - abs(k - 11) / 11)); gd.line([0, k, GT, k], fill=TEAL + (aa,))
            sy = ty + int(computed[i] * GT)
            layer.alpha_composite(glow, (tx, max(ty, min(ty + GT - 22, sy - 11)))); ld = ImageDraw.Draw(layer)
        if computed[i] > 0.02:
            ca = int(255 * clamp(computed[i] * 1.4)); dist = c["dist"]
            bx0, bx1 = tx, tx + GT
            rounded(ld, [bx0, ty + GT + 12, bx1, ty + GT + 20], 4, fill=CARD2 + (ca,))
            frac = clamp(1 - dist / 1.2); barc = (GREEN if is_match else RED) if reveal_t > 0 else TEAL
            fw = int((bx1 - bx0) * frac * clamp(computed[i]))
            if fw > 4:
                rounded(ld, [bx0, ty + GT + 12, bx0 + fw, ty + GT + 20], 4, fill=barc + (ca,))
            ld.text((tx, ty + GT + 28), f"d {dist:.2f}", font=font(21, True, mono=True),
                    fill=(INK if reveal_t == 0 else (GREEN if is_match else DIM)) + (ca,))
            if reveal_t > 0:
                va = int(255 * ease_out(reveal_t))
                vt, vc, vb = ("MATCH", GREEN, GREEN_D) if is_match else ("no match", RED, RED_D)
                fv = font(14, True); vw = ld.textlength(vt, font=fv)
                pill(ld, tx + GT - vw - 22, ty + GT + 25, vt, fv, vc + (va,), vb + (va,))
        if cand_a[i] < 1.0:
            al = layer.split()[3].point(lambda v: int(v * cand_a[i])); layer.putalpha(al)
        cv.alpha_composite(layer)
        if reveal_t > 0 and is_match:
            # check disc
            cl = Image.new("RGBA", cv.size, (0, 0, 0, 0)); cd = ImageDraw.Draw(cl)
            a = int(255 * ease_out(clamp(reveal_t * 1.4)))
            ccx, ccy, cr = x + CW - 30, y + 30, int(20 * ease_out_back(clamp(reveal_t * 1.6)))
            if cr >= 2:
                cd.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], fill=GREEN_D + (a,), outline=GREEN + (a,), width=3)
                ck = clamp(reveal_t * 2 - 0.5)
                if ck > 0:
                    p0 = (ccx - 8, ccy + 1); p1 = (ccx - 2, ccy + 7); p2 = (ccx + 9, ccy - 7)
                    if ck < 0.5:
                        tt = ck / 0.5; mp = (p0[0] + (p1[0] - p0[0]) * tt, p0[1] + (p1[1] - p0[1]) * tt)
                        cd.line([p0, mp], fill=GREEN + (255,), width=4)
                    else:
                        tt = (ck - 0.5) / 0.5; mp = (p1[0] + (p2[0] - p1[0]) * tt, p1[1] + (p2[1] - p1[1]) * tt)
                        cd.line([p0, p1], fill=GREEN + (255,), width=4); cd.line([p1, mp], fill=GREEN + (255,), width=4)
                cv.alpha_composite(cl)

    # connector
    if phase == "scan" and active is not None:
        cl = Image.new("RGBA", cv.size, (0, 0, 0, 0)); cd = ImageDraw.Draw(cl)
        pcx, pcy = px0 + 24 + PT / 2, py0 + 24 + PT / 2
        x, y = sq_cell(active); ccx, ccy = x + 12 + GT / 2, y + 12 + GT / 2
        tt = ease_in_out(scan_pos - int(scan_pos))
        dx, dy = pcx + (ccx - pcx) * tt, pcy + (ccy - pcy) * tt
        cd.line([pcx, pcy, dx, dy], fill=TEAL + (130,), width=2)
        cd.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=TEAL + (220,))
        cv.alpha_composite(cl)

    # verdict banner
    if reveal_t > 0:
        c = state["cands"][state["match_idx"]]; a = int(255 * ease_out(clamp(reveal_t * 1.3)))
        d = ImageDraw.Draw(cv)
        by = gy0 + 2 * CH + gy + 16
        rounded(d, [48, by, S - 48, by + 56], 12, fill=CARD + (a,), outline=GREEN_D + (a,), width=1)
        d.rectangle([62, by + 16, 66, by + 40], fill=GREEN + (a,))
        d.text((80, by + 12), "MATCH  ·  same person", font=font(22, True), fill=GREEN + (a,))
        d.text((80, by + 38), f"d = {c['dist']:.2f}   verified   ·   threshold {state['threshold']:.2f}   ·   1 match in {n}",
               font=font(14, False), fill=DIM + (a,))

    # brandline
    d = ImageDraw.Draw(cv)
    d.text((48, S - 40), "github.com/mudler/face-detect.cpp", font=font(13, False), fill=DIMMER)
    t = "Brought to you by the LocalAI team  ·  localai.io"
    tw = d.textlength(t, font=font(13, False))
    d.text((S - 48 - tw, S - 40), t, font=font(13, False), fill=DIMMER)
    return cv.convert("RGB")


def build_frames_square(state, fps, outdir):
    frames = []
    n = len(state["cands"])
    F_INTRO = max(8, int(fps * 0.7)); F_PER = max(7, int(fps * 0.42))
    F_GAP = max(2, int(fps * 0.08)); F_DECIDE = max(8, int(fps * 0.55)); F_HOLD = int(fps * 2.4)
    for f in range(F_INTRO):
        frames.append(render_frame_square(state, "intro", f / (F_INTRO - 1)))
    for f in range(F_PER * n):
        frames.append(render_frame_square(state, "scan", 0, min((f + 1) / F_PER, n)))
    for f in range(F_GAP):
        frames.append(render_frame_square(state, "scan", 0, float(n)))
    for f in range(F_DECIDE):
        frames.append(render_frame_square(state, "decide", f / (F_DECIDE - 1)))
    for f in range(F_HOLD):
        frames.append(render_frame_square(state, "hold", 1.0))
    F_END_IN = max(10, int(fps * 0.9)); F_END_HOLD = int(fps * 2.6)
    for f in range(F_END_IN):
        frames.append(draw_end_card(state, f / (F_END_IN - 1), size=(1080, 1080)))
    last = draw_end_card(state, 1.0, size=(1080, 1080))
    for f in range(F_END_HOLD):
        frames.append(last)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("frame_*.png"):
        old.unlink()
    for i, fr in enumerate(frames):
        fr.save(outdir / f"frame_{i:04d}.png")
    return len(frames)


# ---------------------------------------------------------------- encode
def encode(framedir, fps, gif_fps, mp4, gif, scale=900):
    pal = framedir / "palette.png"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(framedir / "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vf = f"fps={gif_fps},scale={scale}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(framedir / "frame_%04d.png"),
                    "-vf", f"{vf},palettegen=stats_mode=diff", str(pal)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(framedir / "frame_%04d.png"),
                    "-i", str(pal), "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    str(gif)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=str(ROOT / "build-cpu/examples/cli/facedetect-cli"))
    ap.add_argument("--model", default=str(ROOT / "models/buffalo_l-f16.gguf"))
    ap.add_argument("--fixtures", default=str(ROOT / "tests/fixtures"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks/media"))
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--gif-fps", type=int, default=13)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--square", action="store_true")
    a = ap.parse_args()

    cache = Path("/tmp/face_identity_cache.pkl")
    if cache.exists() and not a.no_cache:
        with open(cache, "rb") as fh:
            state = pickle.load(fh)
        print(f"loaded state from cache ({len(state['cands'])} candidates)")
    else:
        print("gathering real CLI outputs ...")
        state = gather(a.cli, a.model, a.fixtures)
        with open(cache, "wb") as fh:
            pickle.dump(state, fh)
    # report the distance table
    print("\nprobe = face_a.jpg   threshold =", state["threshold"])
    for i, c in enumerate(state["cands"]):
        tag = "  <-- MATCH" if i == state["match_idx"] else ""
        print(f"  [{i}] {c['src']:<16} d={c['dist']:.4f}  verified={c['verified']}{tag}")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    framedir = Path(tempfile.mkdtemp(prefix="face_identity_frames_"))
    nf = build_frames(state, a.fps, framedir)
    print(f"\nrendered {nf} frames (16:9) -> {framedir}")
    encode(framedir, a.fps, a.gif_fps, outdir / "face_identity.mp4", outdir / "face_identity.gif", scale=900)
    print("wrote", outdir / "face_identity.mp4", "and .gif")

    if a.square:
        sdir = Path(tempfile.mkdtemp(prefix="face_identity_sq_"))
        nfs = build_frames_square(state, a.fps, sdir)
        print(f"rendered {nfs} frames (1:1) -> {sdir}")
        encode(sdir, a.fps, a.gif_fps, outdir / "face_identity_square.mp4", outdir / "face_identity_square.gif", scale=720)
        print("wrote", outdir / "face_identity_square.mp4", "and .gif")

    # leave inspection frames
    insp = outdir / "_identity_frames"
    insp.mkdir(exist_ok=True)
    for tag, i in [("scan", int(nf * 0.30)), ("decide", int(nf * 0.62)), ("end", nf - 30)]:
        shutil.copy(framedir / f"frame_{i:04d}.png", insp / f"{tag}_{i:04d}.png")
    print("inspection frames in", insp)


if __name__ == "__main__":
    main()
