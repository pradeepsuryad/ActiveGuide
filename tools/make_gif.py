"""Convert the demo videos into GIFs the README can actually play.

Run:  python tools/make_gif.py --all
      python tools/make_gif.py logs/phase3.mp4 --out docs/phase3.gif
      python tools/make_gif.py --all --max-mb 3

Why this exists
---------------
GitHub does not play a committed `.mp4` inline in a README -- the file renders
as a download link, so the one artefact that shows the controller *moving* is
the one nobody sees. A GIF plays. The demos keep writing MP4 because it is
20-40x smaller and is the right format for looking at a run properly; this
converts a copy for the page.

Keeping GIFs small
------------------
A GIF is intra-frame compressed with a 256-colour palette, so cost scales with
frames x pixels and there is no motion compensation to save you. Three knobs,
in the order worth turning:

  --step    keep every Nth frame. Cheapest by far, and these runs are slow
            sweeps where consecutive frames differ very little.
  --width   downscale. Quadratic in cost, but text in the render gets mushy.
  --colors  shrink the palette. The scenes are a few flat-shaded solids on a
            dark grid, so far fewer than 256 colours are needed.

``--max-mb`` retries with progressively larger ``--step`` until the file fits,
so the README never picks up a 30 MB asset by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The standard set, with per-clip framing. Phase 3 is three panels side by side
# and so needs more width than the two-panel Phase 2 clips to stay legible.
STANDARD = [
    ("logs/phase3.mp4", "docs/phase3.gif", 900),
    ("logs/phase2_panda.mp4", "docs/phase2_panda.gif", 720),
    ("logs/phase2_psm.mp4", "docs/phase2_psm.gif", 720),
]


def read_frames(path: Path):
    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("needs imageio: pip install -e \".[media]\"")
    try:
        reader = imageio.get_reader(str(path))
    except Exception as exc:                       # noqa: BLE001 - report and stop
        sys.exit(f"cannot read {path}: {exc}\n"
                 "reading mp4 needs imageio-ffmpeg: pip install -e \".[media]\"")
    with reader:
        return [f for f in reader]


def build(frames, out: Path, width: int, fps: int, step: int, colors: int) -> float:
    from PIL import Image

    keep = frames[::step]
    if len(keep) < 2:
        sys.exit("not enough frames after subsampling; lower --step")

    imgs = []
    for f in keep:
        im = Image.fromarray(f)
        if im.width != width:
            h = round(im.height * width / im.width)
            im = im.resize((width, h), Image.LANCZOS)
        # One adaptive palette per frame, then let the GIF writer share what it
        # can. ADAPTIVE beats the default web palette badly on these renders,
        # which are mostly smooth greys and one saturated red.
        imgs.append(im.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT))

    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=round(1000 / fps), loop=0, optimize=True, disposal=2)
    return out.stat().st_size / 1e6


def convert(src: Path, out: Path, width: int, fps: int, step: int,
            colors: int, max_mb: float) -> None:
    if not src.exists():
        print(f"[skip] {src.relative_to(ROOT)} not found -- run the demo first")
        return

    frames = read_frames(src)
    print(f"{src.relative_to(ROOT)}: {len(frames)} frames "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")

    while True:
        mb = build(frames, out, width, fps, step, colors)
        fits = max_mb <= 0 or mb <= max_mb
        print(f"  -> {out.relative_to(ROOT)}  {mb:.2f} MB  "
              f"(step {step}, {len(frames[::step])} frames, {width}px, {colors} colours)"
              + ("" if fits else "  too big, retrying"))
        if fits or step >= 16:
            if not fits:
                print("  ! still over budget at step 16; lower --width or --colors")
            return
        step += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="input .mp4 (omit with --all)")
    ap.add_argument("--out", help="output .gif")
    ap.add_argument("--all", action="store_true", help="convert the standard set")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--step", type=int, default=3, help="keep every Nth frame")
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--max-mb", type=float, default=4.0,
                    help="raise --step until the GIF fits (0 disables)")
    args = ap.parse_args()

    if args.all:
        for src, out, width in STANDARD:
            convert(ROOT / src, ROOT / out, width, args.fps, args.step,
                    args.colors, args.max_mb)
        return

    if not args.src:
        ap.error("give an input file or --all")
    src = Path(args.src)
    out = Path(args.out) if args.out else ROOT / "docs" / (src.stem + ".gif")
    convert(src if src.is_absolute() else ROOT / src,
            out if out.is_absolute() else ROOT / out,
            args.width, args.fps, args.step, args.colors, args.max_mb)


if __name__ == "__main__":
    main()
