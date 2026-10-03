"""The explainer film's geometry helpers (skipped where Manim, which only the film needs, is absent)."""

import ast
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np

HAS_MANIM = importlib.util.find_spec("manim") is not None
EXPLAINER = Path(__file__).resolve().parent.parent / "explainer"
sys.path.insert(0, str(EXPLAINER))


@unittest.skipUnless(HAS_MANIM, "the explainer's dependencies are not installed")
class ArrowEndsTest(unittest.TestCase):
    """The overlap check samples arrows between their reported ends, so the ends must follow
    every way the film moves an arrow: turning with a cubelet, applying a matrix, shifting."""

    def setUp(self):
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
    """No caption breaks inside a spoken tuple ("minus one, zero, or one"), in any line of the
    film or in lines long enough to force breaks around one."""

    # Lines where breaking at any word would split the tuple, at "or | one" and "minus | one".
    FORCED = [
        # "xx" pads the line so that the caption's even split falls right after "or".
        "xx the minus one, zero, or one and stays there for good.",
        "The tip ends up at minus one, zero and rests there.",
    ]

    def test_no_caption_breaks_inside_a_tuple(self):
        import build
        film = [node.value for node in ast.walk(ast.parse((EXPLAINER / "scenes.py").read_text()))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        lines = [line for line in film + self.FORCED if re.search(build.SPOKEN_TUPLE, line)]
        self.assertGreater(len(lines), len(self.FORCED))  # The film's own tuples are covered.
        for line in lines:
            chunks = build.caption_chunks(line)
            self.assertEqual(" ".join(chunks), line)
            self.assertTrue(all(len(chunk) <= build.CAPTION_WIDTH for chunk in chunks), chunks)
            breaks = [sum(len(c) + 1 for c in chunks[:k]) - 1 for k in range(1, len(chunks))]
            for span in re.finditer(build.SPOKEN_TUPLE, line):
                inside = [b for b in breaks if span.start() < b < span.end()]
                self.assertFalse(inside, f"{span[0]!r} split in {chunks}")


if __name__ == "__main__":
    unittest.main()
