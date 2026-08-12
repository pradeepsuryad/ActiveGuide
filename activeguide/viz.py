"""Caption bars for the side-by-side demo videos.

A comparison video is only worth anything if you can tell which panel is which.
The demos render two or three controllers on the same task next to each other,
and without a label the clip shows three near-identical arms doing subtly
different things -- which is exactly the information the comparison is meant to
carry.

Kept apart from the demos because both Phase 2 and Phase 3 compose panels this
way, and because it is presentation code: nothing here feeds a metric.
"""

from __future__ import annotations

import numpy as np

BAR_BG = (20, 20, 27)
BAR_FG = (238, 238, 244)


def _font(size: int):
    """A readable font, falling back through what is likely to exist."""
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                 "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1 scales it
    except TypeError:                                 # older Pillow: fixed size
        return ImageFont.load_default()


def caption_bar(width: int, text: str, height: int = 30, size: int = 17,
                bg=BAR_BG, fg=BAR_FG) -> np.ndarray:
    """Render one caption strip, ``(height, width, 3)`` uint8.

    Built once per panel rather than per frame: the text never changes within a
    run, and drawing it onto several hundred frames individually is pure waste.
    """
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(im)
    font = _font(size)
    try:
        w = draw.textlength(text, font=font)
    except AttributeError:                            # very old Pillow
        w = font.getsize(text)[0]
    draw.text(((width - w) / 2.0, (height - size) / 2.0 - 1), text,
              font=font, fill=fg)
    return np.asarray(im, dtype=np.uint8)


def stack_labeled(panels, labels, **kw) -> list[np.ndarray]:
    """Caption each panel and lay them out left to right, frame by frame.

    ``panels`` is one frame list per controller. Runs finish at different times
    -- that is itself a result -- so shorter ones hold on their final frame
    rather than being cut, and the clip stays synchronised to the slowest.
    """
    panels = [list(p) for p in panels]
    if not panels or any(not p for p in panels):
        return []
    if len(panels) != len(labels):
        raise ValueError(f"{len(panels)} panels but {len(labels)} labels")

    n = max(len(p) for p in panels)
    panels = [p + [p[-1]] * (n - len(p)) for p in panels]
    bars = [caption_bar(p[0].shape[1], t, **kw) for p, t in zip(panels, labels)]

    return [np.hstack([np.vstack([bar, frame])
                       for bar, frame in zip(bars, column)])
            for column in zip(*panels)]
