"""Tests for the comparison-video caption bars.

Presentation code, but the padding rule carries meaning: runs finish at
different times -- that is a result in Phase 2 and Phase 3 alike -- so a shorter
run must hold on its last frame rather than be truncated, or the clip would
quietly hide the controller that took longest.
"""

import unittest

import numpy as np

from activeguide.viz import caption_bar, stack_labeled

H, W = 32, 48


def frames(n, value):
    return [np.full((H, W, 3), value, dtype=np.uint8) for _ in range(n)]


class TestCaptionBar(unittest.TestCase):
    def test_shape_and_dtype(self):
        bar = caption_bar(W, "hello", height=20)
        self.assertEqual(bar.shape, (20, W, 3))
        self.assertEqual(bar.dtype, np.uint8)

    def test_text_is_actually_drawn(self):
        """A silently-failing font would produce a plain bar and no label."""
        blank = caption_bar(200, " ", height=24)
        text = caption_bar(200, "joint-space QP", height=24)
        self.assertFalse(np.array_equal(blank, text))

    def test_empty_text_is_uniform(self):
        bar = caption_bar(60, "", height=16)
        self.assertEqual(len(np.unique(bar.reshape(-1, 3), axis=0)), 1)


class TestStackLabeled(unittest.TestCase):
    def test_lays_panels_out_side_by_side(self):
        out = stack_labeled([frames(4, 10), frames(4, 20)], ["a", "b"],
                            height=12)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0].shape, (H + 12, 2 * W, 3))

    def test_shorter_run_holds_its_last_frame(self):
        short, long_ = frames(2, 10), frames(5, 20)
        short[-1][:] = 99
        out = stack_labeled([short, long_], ["a", "b"], height=10)

        self.assertEqual(len(out), 5)                  # synced to the slowest
        for frame in out[2:]:                          # after the short run ends
            left = frame[10:, :W]
            self.assertTrue(np.all(left == 99), "short run should freeze, not cut")

    def test_three_panels(self):
        out = stack_labeled([frames(3, i) for i in (1, 2, 3)],
                            ["a", "b", "c"], height=8)
        self.assertEqual(out[0].shape, (H + 8, 3 * W, 3))

    def test_rejects_mismatched_labels(self):
        with self.assertRaises(ValueError):
            stack_labeled([frames(2, 1), frames(2, 2)], ["only one"])

    def test_empty_input_is_not_an_error(self):
        """The demos call this whenever recording is on; a run that produced no
        frames should degrade to 'no video', not crash the whole demo."""
        self.assertEqual(stack_labeled([], []), [])
        self.assertEqual(stack_labeled([frames(2, 1), []], ["a", "b"]), [])


if __name__ == "__main__":
    unittest.main()
