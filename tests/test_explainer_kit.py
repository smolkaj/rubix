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


if __name__ == "__main__":
    unittest.main()
