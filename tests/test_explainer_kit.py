"""The explainer film's geometry helpers (skipped where Manim, which only the film needs, is absent)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HAS_MANIM = importlib.util.find_spec("manim") is not None


@unittest.skipUnless(HAS_MANIM, "the explainer's dependencies are not installed")
class ArrowEndsTest(unittest.TestCase):
    """The overlap check samples arrows between their reported ends, so the ends must follow
    every way the film moves an arrow: turning with a cubelet, applying a matrix, shifting."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "explainer"))
        import kit
        self.kit = kit

    def assert_ends(self, rod, start, end):
        np.testing.assert_allclose(rod.get_start(), start, atol=1e-9)
        np.testing.assert_allclose(rod.get_end(), end, atol=1e-9)

    def test_arrow_ends_follow_rotation_and_shift(self):
        arrow = self.kit.SolidArrow((0, 0, 0), (1, 0, 0), "#FF0000")
        arrow.rotate(np.pi / 2, axis=(0, 0, 1), about_point=(0, 0, 0)).shift((0, 2, 0))
        self.assert_ends(arrow, (0, 2, 0), (0, 3, 0))

    def test_piercing_arrow_ends_follow_a_matrix(self):
        quarter_turn = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        pierce = self.kit.piercing_arrow((1, 1, 1), (1.5, 1, 1), (2, 1, 1), "#00FF00")
        pierce.apply_matrix(quarter_turn, about_point=(0, 0, 0))
        inside, outlined = pierce
        self.assert_ends(inside, quarter_turn @ (1, 1, 1), quarter_turn @ (1.5, 1, 1))
        for arrow in outlined:  # The black outline and the arrow it outlines.
            self.assert_ends(arrow, quarter_turn @ (1.5, 1, 1), quarter_turn @ (2, 1, 1))



@unittest.skipUnless(HAS_MANIM, "the explainer's dependencies are not installed")
class CaptionTest(unittest.TestCase):
    """A spoken tuple ("one, zero, zero") stays on one caption, wherever the line must break."""

    LINES = [
        "A center, like this one, sits at zero, one, zero. And an edge, like this one, at zero, "
        "one, one.",
        "The first is one, zero, zero: that's e x, so green.",
        "The coordinates of every cubelet are minus one, zero, or one.",
        "One, minus one, one. Front, left, top: exactly where the corner went.",
    ]

    def test_no_caption_breaks_inside_a_tuple(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "explainer"))
        import re
        import build
        number = re.compile(build.SPOKEN_NUMBER + r",?$")
        for line in self.LINES:
            chunks = build.caption_chunks(line)
            self.assertEqual(" ".join(chunks), line)
            for before, after in zip(chunks, chunks[1:]):
                self.assertFalse(
                    number.search(before) and re.match(build.SPOKEN_NUMBER, after),
                    f"{before!r} | {after!r}")
            self.assertTrue(all(len(chunk) <= build.CAPTION_WIDTH for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
