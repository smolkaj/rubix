import unittest
import os
import tempfile
import numpy as np

from rubix import (
    solved_cube,
    moves,
    unit_vectors,
    color_names,
    apply_move_to_cube,
    is_cube_solved,
    solve,
    norm1,
    rotation_matrix,
    inverse_move,
    NUM_CUBELETS,
)


class TestRubixCube(unittest.TestCase):
    def test_solved_cube_invariants(self):
        """Verify structural properties and L1 norm classifications of the solved cube."""
        self.assertEqual(len(solved_cube), NUM_CUBELETS)
        self.assertEqual(NUM_CUBELETS, 26)
        self.assertTrue(is_cube_solved(solved_cube))
        self.assertEqual(len(moves), 12)
        self.assertEqual(len(unit_vectors), 6)
        self.assertTrue(all(v in color_names for v in unit_vectors))

        # Check L1 norm classifications: 6 centers, 12 edges, 8 corners
        centers = [c for c, _ in solved_cube if norm1(c) == 1]
        edges = [c for c, _ in solved_cube if norm1(c) == 2]
        corners = [c for c, _ in solved_cube if norm1(c) == 3]
        interiors = [c for c, _ in solved_cube if norm1(c) == 0]

        self.assertEqual(len(centers), 6)
        self.assertEqual(len(edges), 12)
        self.assertEqual(len(corners), 8)
        self.assertEqual(len(interiors), 0)

        # Coordinate domain invariant: {-1, 0, 1}^3
        for c, r in solved_cube:
            self.assertTrue(all(x in (-1, 0, 1) for x in c))
            self.assertEqual(r, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def test_rotation_matrix_properties(self):
        """Verify that all rotation matrices are orthogonal and orientation-preserving (det(M) = 1 and M^T M = I)."""
        identity = np.eye(3)
        for move in moves:
            v, _ = move
            M = rotation_matrix(move)
            # Orthogonality: M^T @ M = I
            self.assertTrue(np.allclose(M.T @ M, identity), f"Move {move} is not orthogonal")
            # Orientation preserving: det(M) = +1 (SO(3))
            self.assertTrue(np.isclose(np.linalg.det(M), 1.0), f"det(M) != 1 for move {move}")
            # Rotational axis preservation: M @ v = v
            self.assertTrue(np.allclose(M @ np.array(v), np.array(v)), f"Axis not fixed for move {move}")

    def test_single_move_order_4(self):
        """Every 90-degree face turn must have order 4 (cycle of 4 returns to identity)."""
        for move in moves:
            cube = solved_cube
            states = [cube]
            for _ in range(4):
                cube = apply_move_to_cube(move, cube)
                states.append(cube)
            # Distinct intermediate states
            self.assertEqual(len(set(states[:4])), 4, f"Move {move} did not produce 4 distinct states")
            # 4th rotation returns to initial state
            self.assertEqual(cube, solved_cube, f"Move {move} x 4 did not return to solved state")

    def test_inverse_move_cancellation(self):
        """Applying a move and its inverse should be the identity."""
        for move in moves:
            inv = inverse_move(move)
            self.assertIn(inv, moves)
            cube = apply_move_to_cube(move, solved_cube)
            self.assertNotEqual(cube, solved_cube)
            cube_restored = apply_move_to_cube(inv, cube)
            self.assertEqual(cube_restored, solved_cube)

    def test_sexy_move_order_6(self):
        """The 'sexy move' (R U R' U') repeated 6 times returns the cube to its original state."""
        # Find R (Right: +y) and U (Up/Top: +z)
        r_cw = ((0, 1, 0), 1)
        r_ccw = ((0, 1, 0), -1)
        u_cw = ((0, 0, 1), 1)
        u_ccw = ((0, 0, 1), -1)

        sexy_move = [r_cw, u_cw, r_ccw, u_ccw]

        cube = solved_cube
        for iteration in range(1, 7):
            for m in sexy_move:
                cube = apply_move_to_cube(m, cube)
            if iteration < 6:
                self.assertFalse(
                    is_cube_solved(cube),
                    f"Cube solved prematurely at iteration {iteration}",
                )

        self.assertEqual(cube, solved_cube)
        self.assertTrue(is_cube_solved(cube))

    def test_solve_simple_scramble(self):
        """Solver should successfully solve a shallowly scrambled cube."""
        # Apply 2 known moves
        scramble_moves = [((1, 0, 0), 1), ((0, 1, 0), -1)]
        cube = solved_cube
        for m in scramble_moves:
            cube = apply_move_to_cube(m, cube)
        self.assertFalse(is_cube_solved(cube))

        # Solve
        solution = solve(cube)
        for m in solution:
            cube = apply_move_to_cube(m, cube)

        self.assertTrue(is_cube_solved(cube))

    def test_shuffle_reproducibility(self):
        """Shuffle with the same seed must produce identical cube states."""
        from rubix import shuffle
        cube_a = shuffle(solved_cube, iterations=20, seed=42)
        cube_b = shuffle(solved_cube, iterations=20, seed=42)
        cube_c = shuffle(solved_cube, iterations=20, seed=99)
        self.assertEqual(cube_a, cube_b)
        self.assertNotEqual(cube_a, cube_c)
        self.assertFalse(is_cube_solved(cube_a))

    def test_descriptions(self):
        """Verify describe_position, describe_move, and describe_cubelet_type."""
        from rubix import describe_position, describe_move, describe_cubelet_type
        self.assertEqual(describe_position((1, 0, 0)), "front")
        self.assertEqual(describe_position((0, 1, 1)), "top-right")
        self.assertEqual(describe_position((-1, -1, -1)), "bottom-left-back")
        self.assertEqual(describe_cubelet_type((0, 0, 1)), "center")
        self.assertEqual(describe_cubelet_type((1, 1, 0)), "edge")
        self.assertEqual(describe_cubelet_type((1, 1, 1)), "corner")
        self.assertEqual(
            describe_move(((1, 0, 0), 1)),
            "clockwise rotation of front slice",
        )
        self.assertEqual(
            describe_move(((0, 0, -1), -1)),
            "counterclockwise rotation of bottom slice",
        )

    def test_headless_gui_render(self):
        """Verify that rubix_gui renders a frame headlessly without error."""
        import rubix_gui

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "preview.png")
            result = rubix_gui.render_frame_to_image(solved_cube, out_path)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_opposite_face_moves_commute(self):
        """Opposite face moves act on disjoint slices and commute: m1 * m2 == m2 * m1."""
        for m1 in moves:
            v1, _ = m1
            opposite_v = tuple(-x for x in v1)
            for d2 in [-1, 1]:
                m2 = (opposite_v, d2)
                # Apply m1 then m2
                s1 = apply_move_to_cube(m2, apply_move_to_cube(m1, solved_cube))
                # Apply m2 then m1
                s2 = apply_move_to_cube(m1, apply_move_to_cube(m2, solved_cube))
                self.assertEqual(s1, s2, f"Opposite moves {m1} and {m2} did not commute")

    def test_min_moves_to_position(self):
        """Verify min_moves_to_position ignores piece orientation."""
        from rubix import min_moves_to_position, min_moves_to_solved, position
        corner = (1, 1, -1)
        # Identity rotation: 0 moves to solved and position
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertEqual(min_moves_to_position(corner, identity), 0)
        self.assertEqual(min_moves_to_solved(corner, identity), 0)

        # Find a rotation where corner is in home place but twisted
        from rubix import shuffle
        for seed in range(50):
            cube = shuffle(solved_cube, iterations=20, seed=seed)
            for c, r in cube:
                if c == corner and position(c, r) == corner and r != identity:
                    # Corner is positioned but twisted
                    self.assertEqual(min_moves_to_position(c, r), 0)
                    self.assertGreater(min_moves_to_solved(c, r), 0)
                    return


if __name__ == "__main__":
    unittest.main()
