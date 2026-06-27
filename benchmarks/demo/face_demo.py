#!/usr/bin/env python3
r"""face_demo - compose the face-detect.cpp HERO panel with Pillow.

On CPU we do not win a speed race, so the hero is CAPABILITY + bit-exactness:
one binary runs the whole InsightFace pipeline (detect, 5-point align, 512-d
recognition, age/gender) on a real photo, and the numbers on screen are the
REAL measured CLI outputs, parsed live from the JSON the binary prints.

It runs the CLI for real (detect / analyze / verify SAME / verify DIFFERENT),
parses the JSON, and draws a single polished still on the dark #0d1117 palette
matching the locate-anything.cpp demo look. No em-dashes in rendered text.

  python3 face_demo.py
  python3 face_demo.py --cli ../../build-cpu/examples/cli/facedetect-cli \
      --model ../../models/buffalo_l-f16.gguf --fixtures ../../tests/fixtures \
      --out ../media/face_demo.png
"""
import argparse, json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

# palette - matches locate-anything.cpp / the LocalAI race
BG     = (13, 17, 23)       # #0d1117
CARD   = (22, 27, 34)       # #161b22
CARD2  = (28, 34, 43)
LINE   = (38, 46, 56)
INK    = (223, 228, 234)
DIM    = (132, 142, 154)
DIMMER = (96, 105, 117)
TEAL   = (62, 200, 224)     # accent
TEAL_D = (32, 110, 126)
GREEN  = (80, 200, 120)
GREEN_D= (28, 70, 46)
ROSE   = (232, 110, 120)
ROSE_D = (78, 38, 44)
GOLD   = (240, 200, 90)

FONTDIR = "/usr/share/fonts/truetype/dejavu"
def font(sz, bold=True, mono=False):
    name = "DejaVuSansMono" if mono else ("DejaVuSans-Bold" if bold else "DejaVuSans")
    if mono and bold:
        name = "DejaVuSansMono-Bold"
    try:
        return ImageFont.truetype(f"{FONTDIR}/{name}.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def run_json(cli, args):
    out = subprocess.run([cli, *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def gather(cli, model, fixtures):
    fx = Path(fixtures)
    a = str(fx / "face_a.jpg"); b = str(fx / "face_b.jpg"); c = str(fx / "face_c.jpg")
    det = run_json(cli, ["detect", "--model", model, "--input", a, "--json"])
    ana = run_json(cli, ["analyze", "--model", model, "--input", a, "--json"])
    emb = run_json(cli, ["embed", "--model", model, "--input", a, "--json"])
    vab = run_json(cli, ["verify", "--model", model, "--a", a, "--b", b])
    vac = run_json(cli, ["verify", "--model", model, "--a", a, "--b", c])
    return {"img": a, "det": det, "ana": ana, "emb": emb, "vab": vab, "vac": vac}


def rounded(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def pill(d, x, y, text, fnt, fg, bg, padx=10, pady=5):
    tw = d.textlength(text, font=fnt)
    asc, desc = fnt.getmetrics(); th = asc + desc
    rounded(d, [x, y, x + tw + 2 * padx, y + th + 2 * pady], (th + 2 * pady) // 2, fill=bg)
    d.text((x + padx, y + pady), text, font=fnt, fill=fg)
    return tw + 2 * padx


def draw_photo(cv, data, rect):
    """Left pane: real photo, teal face box, 5 SCRFD landmark dots, score badge."""
    d = ImageDraw.Draw(cv)
    ox, oy, pw, ph = rect
    img = Image.open(data["img"]).convert("RGB")
    s = min(pw / img.width, ph / img.height)
    iw, ih = int(img.width * s), int(img.height * s)
    img = img.resize((iw, ih), Image.LANCZOS)
    ix, iy = ox + (pw - iw) // 2, oy + (ph - ih) // 2
    # soft frame shadow
    rounded(d, [ix - 3, iy - 3, ix + iw + 3, iy + ih + 3], 10, outline=LINE, width=2)
    cv.paste(img, (ix, iy))
    d = ImageDraw.Draw(cv)
    face = data["det"]["faces"][0]
    x1, y1, x2, y2 = face["box"]
    bx = [ix + x1 * s, iy + y1 * s, ix + x2 * s, iy + y2 * s]
    # teal detection box with corner ticks
    d.rectangle(bx, outline=TEAL, width=3)
    cl = 16
    for (cx, cy, dx, dy) in [(bx[0], bx[1], 1, 1), (bx[2], bx[1], -1, 1),
                             (bx[0], bx[3], 1, -1), (bx[2], bx[3], -1, -1)]:
        d.line([cx, cy, cx + dx * cl, cy], fill=TEAL, width=5)
        d.line([cx, cy, cx, cy + dy * cl], fill=TEAL, width=5)
    # score badge above box
    fb = font(15, True)
    pill(d, bx[0], bx[1] - 30, f"face  {face['score']*100:.0f}%", fb, BG, TEAL, padx=9, pady=4)
    # 5 landmark dots
    for (lx, ly) in face["landmarks"]:
        px, py = ix + lx * s, iy + ly * s
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=GOLD, outline=BG, width=2)
    # caption under photo
    fc = font(14, False)
    cap = "tests/fixtures/face_a.jpg  (real photo)"
    d.text((ix, iy + ih + 12), cap, font=fc, fill=DIMMER)


def kv_row(d, x, y, w, tag, value, sub=None, accent=TEAL):
    """A capability row: accent tab, TAG (caps), value, optional sub."""
    ft = font(13, True); fv = font(21, True); fs = font(13, False)
    d.rectangle([x, y + 3, x + 4, y + 34], fill=accent)
    d.text((x + 16, y + 1), tag, font=ft, fill=accent)
    d.text((x + 16, y + 18), value, font=fv, fill=INK)
    if sub:
        vw = d.textlength(value, font=fv)
        d.text((x + 16 + vw + 12, y + 24), sub, font=fs, fill=DIM)


def verify_row(d, x, y, w, label, dist, same):
    fl = font(15, False); fd = font(20, True, mono=True); fp = font(13, True)
    accent = GREEN if same else ROSE
    abg = GREEN_D if same else ROSE_D
    rounded(d, [x, y, x + w, y + 46], 9, fill=CARD2)
    d.text((x + 14, y + 13), label, font=fl, fill=INK)
    # distance value (monospace, right-ish)
    dtxt = f"{dist:.2f}"
    dw = d.textlength(dtxt, font=fd)
    dx = x + w - 150
    d.text((dx - dw, y + 12), dtxt, font=fd, fill=accent)
    d.text((dx - dw - d.textlength("dist ", font=font(13, False)), y + 17), "dist ", font=font(13, False), fill=DIMMER)
    # verdict pill
    verdict = "SAME" if same else "DIFFERENT"
    pill(d, x + w - 110, y + 11, verdict, fp, accent, abg, padx=12, pady=5)


def render(data, out):
    W, H = 1280, 720
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)

    # ---- title strip ----
    f_brand = font(38, True)
    bx = 48
    d.text((bx, 30), "face-detect", font=f_brand, fill=INK)
    bw = d.textlength("face-detect", font=f_brand)
    d.text((bx + bw, 30), ".cpp", font=f_brand, fill=TEAL)
    sub = "detect, align, recognize, age + gender. one binary, no Python, bit-exact vs InsightFace"
    d.text((bx + 2, 78), sub, font=font(16, False), fill=DIM)
    d.line([bx, 116, W - 48, 116], fill=LINE, width=1)

    # ---- left photo pane ----
    pad = 48
    pane_w = 470
    draw_photo(cv, data, (pad, 140, pane_w, 500))
    d = ImageDraw.Draw(cv)

    # ---- right capability column ----
    rx = pad + pane_w + 40
    rw = W - rx - pad
    ry = 140
    d.text((rx, ry), "ONE PHOTO, ONE BINARY, FULL INSIGHTFACE PIPELINE",
           font=font(13, True), fill=DIMMER)
    ry += 30

    face = data["det"]["faces"][0]
    ana = data["ana"]["faces"][0]
    gender = "Male" if ana["gender"].upper().startswith("M") else "Female"
    dim = data["emb"]["dim"]

    # capability cards grid (2 columns x 2 rows)
    gh = 64; gw = (rw - 18) // 2
    cells = [
        ("DETECT", "1 face", f"score {face['score']:.2f}", TEAL),
        ("LANDMARKS", "5-point", "SCRFD align", TEAL),
        ("RECOGNIZE", f"{dim}-d", "embedding", TEAL),
        ("AGE / GENDER", f"{ana['age']}, {gender}", "audeering", GOLD),
    ]
    for i, (tag, val, sub, acc) in enumerate(cells):
        col = i % 2; row = i // 2
        cx = rx + col * (gw + 18)
        cy = ry + row * (gh + 14)
        rounded(d, [cx, cy, cx + gw, cy + gh], 11, fill=CARD)
        kv_row(d, cx + 10, cy + 13, gw - 20, tag, val, sub, acc)
    ry += 2 * gh + 14 + 24

    # ---- verify / 1:N identity ----
    d.text((rx, ry), "VERIFY  (cosine distance, lower = same identity)",
           font=font(13, True), fill=DIMMER)
    ry += 26
    verify_row(d, rx, ry, rw, "vs portrait B  (same person)",
               data["vab"]["distance"], data["vab"]["verified"])
    ry += 56
    verify_row(d, rx, ry, rw, "vs portrait C  (different person)",
               data["vac"]["distance"], data["vac"]["verified"])
    ry += 56

    note = "no cloud, no Python runtime. GGUF weights, parity verified vs InsightFace ONNX."
    d.text((rx, ry + 6), note, font=font(13, False), fill=DIM)

    # ---- footer: logo + links ----
    logo = Image.open(HERE.parent.parent / "assets" / "localai_logo.png").convert("RGBA")
    lh = 54; lw = int(logo.width * lh / logo.height)
    logo = logo.resize((lw, lh), Image.LANCZOS)
    logo_x = W - pad - lw
    logo_y = H - 64
    cv.paste(logo, (logo_x, logo_y), logo)

    # brand tagline on the far left, vertically centred with the logo
    fl_tag = font(13, False)
    d.text((bx, logo_y + (lh - sum(fl_tag.getmetrics())) // 2),
           "Brought to you by the LocalAI team  ·  localai.io", font=fl_tag, fill=DIMMER)

    # four required links in two readable rows, right-aligned just left of the logo.
    # The umbrella LocalAI links (localai.io + github.com/mudler/LocalAI) are accented
    # teal; the project repo + HF weights sit beneath in ink/dim. No crowding: the
    # block hugs the logo on the right, the tagline anchors the left.
    fl = font(14, False)
    x_right = logo_x - 22
    rows = [
        [("localai.io", TEAL), ("  ·  ", DIMMER), ("github.com/mudler/LocalAI", TEAL)],
        [("github.com/mudler/face-detect.cpp", INK), ("  ·  ", DIMMER),
         ("huggingface.co/mudler/face-detect-gguf", DIM)],
    ]
    asc, desc = fl.getmetrics(); lh_txt = asc + desc
    gap = 6
    block_h = 2 * lh_txt + gap
    ry0 = logo_y + (lh - block_h) // 2
    for ri, segs in enumerate(rows):
        total = sum(d.textlength(t, font=fl) for t, _ in segs)
        sx = x_right - total
        sy = ry0 + ri * (lh_txt + gap)
        for text, col in segs:
            d.text((sx, sy), text, font=fl, fill=col)
            sx += d.textlength(text, font=fl)

    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=str(HERE.parent.parent / "build-cpu/examples/cli/facedetect-cli"))
    ap.add_argument("--model", default=str(HERE.parent.parent / "models/buffalo_l-f16.gguf"))
    ap.add_argument("--fixtures", default=str(HERE.parent.parent / "tests/fixtures"))
    ap.add_argument("--out", default=str(HERE.parent / "media" / "face_demo.png"))
    a = ap.parse_args()
    data = gather(a.cli, a.model, a.fixtures)
    render(data, a.out)


if __name__ == "__main__":
    main()
