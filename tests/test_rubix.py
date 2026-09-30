import unittest

from rubix import (
    solved_cube,
    moves,
    unit_vectors,
    color_names,
    apply_move_to_cube,
    is_cube_solved,
    solve,
    NUM_CUBELETS,
)


class TestRubixCube(unittest.TestCase):
    def test_solved_cube_invariants(self):
        """Verify structural properties of the solved cube."""
        self.assertEqual(len(solved_cube), NUM_CUBELETS)
        self.assertEqual(NUM_CUBELETS, 26)
        self.assertTrue(is_cube_solved(solved_cube))
        self.assertEqual(len(moves), 12)
        self.assertEqual(len(unit_vectors), 6)
        self.assertTrue(all(v in color_names for v in unit_vectors))

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
        for v in unit_vectors:
            move_cw = (v, 1)
            move_ccw = (v, -1)
            cube = apply_move_to_cube(move_cw, solved_cube)
            self.assertNotEqual(cube, solved_cube)
            cube_restored = apply_move_to_cube(move_ccw, cube)
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


if __name__ == "__main__":
    unittest.main()
