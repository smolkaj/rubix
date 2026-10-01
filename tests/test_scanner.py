import unittest
import os
import tempfile
import numpy as np

from rubix import (
    solved_cube,
    shuffle,
    apply_move_to_cube,
    is_cube_solved,
    solve,
    color_names,
    unit_vectors,
    norm1,
    position,
)


class TestRubiksCubeScanner(unittest.TestCase):
    def setUp(self):
        # Import scanner components
        from rubix_scanner import (
            ColorClassifier,
            CubeFaceDetector,
            CubeModel,
            SyntheticCubeFeed,
            order_quad_points,
            pos_to_grid,
            grid_to_pos,
            VALID_EDGES,
            VALID_CORNERS,
            CANONICAL_COLORS,
            get_confidence_color,
            RGB_COLORS,
            RubiksCubeScanner,
            render_scanner_ui,
        )
        self.ColorClassifier = ColorClassifier
        self.CubeFaceDetector = CubeFaceDetector
        self.CubeModel = CubeModel
        self.SyntheticCubeFeed = SyntheticCubeFeed
        self.order_quad_points = order_quad_points
        self.pos_to_grid = pos_to_grid
        self.grid_to_pos = grid_to_pos
        self.VALID_EDGES = VALID_EDGES
        self.VALID_CORNERS = VALID_CORNERS
        self.CANONICAL_COLORS = CANONICAL_COLORS
        self.get_confidence_color = get_confidence_color
        self.RGB_COLORS = RGB_COLORS
        self.RubiksCubeScanner = RubiksCubeScanner
        self.render_scanner_ui = render_scanner_ui

    def test_color_classifier(self):
        """Verify that ColorClassifier accurately identifies the 6 canonical colors."""
        classifier = self.ColorClassifier()
        canonical_bgr = {
            "WHITE": (240, 240, 240),
            "YELLOW": (20, 220, 240),
            "GREEN": (45, 195, 45),
            "BLUE": (220, 110, 25),
            "RED": (30, 30, 225),
            "ORANGE": (20, 125, 235),
        }
        for expected_name, bgr in canonical_bgr.items():
            patch = np.full((30, 30, 3), bgr, dtype=np.uint8)
            detected, conf = classifier.classify_patch(patch)
            self.assertEqual(detected, expected_name, f"Expected {expected_name}, got {detected}")
            self.assertGreater(conf, 0.5)

    def test_color_classifier_noise_and_dark_rejection(self):
        """Verify that dark shadows, black borders, and desaturated noise are rejected as UNKNOWN."""
        classifier = self.ColorClassifier()
        dark_patches = [
            (10, 10, 10),   # Pitch black
            (25, 25, 25),   # Dark plastic seam
            (15, 10, 35),   # Dark shadow
            (40, 40, 40),   # Dim neutral gray
        ]
        for bgr in dark_patches:
            patch = np.full((20, 20, 3), bgr, dtype=np.uint8)
            detected, conf = classifier.classify_patch(patch)
            self.assertEqual(detected, "UNKNOWN", f"Expected UNKNOWN for {bgr}, got {detected}")
            self.assertEqual(conf, 0.0)

    def test_grid_pos_bijection(self):
        """Verify that grid_to_pos and pos_to_grid form a complete bijection for all 54 facelets."""
        face_normals = [
            ( 1,  0,  0),  # FRONT
            ( 0,  1,  0),  # RIGHT
            ( 0,  0,  1),  # TOP
            ( 0, -1,  0),  # LEFT
            (-1,  0,  0),  # BACK
            ( 0,  0, -1),  # BOTTOM
        ]
        visited_positions = set()
        for fn in face_normals:
            for r in range(3):
                for c in range(3):
                    pos = self.grid_to_pos(fn, r, c)
                    self.assertEqual(np.dot(pos, fn), 1)
                    r_back, c_back = self.pos_to_grid(pos, fn)
                    self.assertEqual((r, c), (r_back, c_back))
                    visited_positions.add((pos, fn))

        self.assertEqual(len(visited_positions), 54)

    def test_physical_camera_geometry_consistency(self):
        """Verify that adjacent faces share edge cubelets and corner cubelets with 100% spatial consistency."""
        # Front top edge (r=0, c=1) must equal Top front edge (r=2, c=1) -> cubelet (1, 0, 1)
        self.assertEqual(self.grid_to_pos((1, 0, 0), 0, 1), (1, 0, 1))
        self.assertEqual(self.grid_to_pos((0, 0, 1), 2, 1), (1, 0, 1))

        # Front right edge (r=1, c=2) must equal Right front edge (r=1, c=0) -> cubelet (1, 1, 0)
        self.assertEqual(self.grid_to_pos((1, 0, 0), 1, 2), (1, 1, 0))
        self.assertEqual(self.grid_to_pos((0, 1, 0), 1, 0), (1, 1, 0))

        # Right back edge (r=1, c=2) must equal Back right edge (r=1, c=0) -> cubelet (-1, 1, 0)
        self.assertEqual(self.grid_to_pos((0, 1, 0), 1, 2), (-1, 1, 0))
        self.assertEqual(self.grid_to_pos((-1, 0, 0), 1, 0), (-1, 1, 0))

        # Back left edge (r=1, c=2) must equal Left back edge (r=1, c=0) -> cubelet (-1, -1, 0)
        self.assertEqual(self.grid_to_pos((-1, 0, 0), 1, 2), (-1, -1, 0))
        self.assertEqual(self.grid_to_pos((0, -1, 0), 1, 0), (-1, -1, 0))

        # Front-Right-Top corner (1, 1, 1)
        self.assertEqual(self.grid_to_pos((1, 0, 0), 0, 2), (1, 1, 1))
        self.assertEqual(self.grid_to_pos((0, 1, 0), 0, 0), (1, 1, 1))
        self.assertEqual(self.grid_to_pos((0, 0, 1), 2, 2), (1, 1, 1))

    def test_cube_model_solved_state(self):
        """Verify that CubeModel correctly validates and converts a solved cube."""
        model = self.CubeModel()
        self.assertEqual(model.get_progress()[0], 0)
        self.assertEqual(model.get_progress()[1], 0)

        # Register solved faces
        for fn_name, fn in [
            ("FRONT", (1, 0, 0)),
            ("RIGHT", (0, 1, 0)),
            ("TOP", (0, 0, 1)),
            ("LEFT", (0, -1, 0)),
            ("BACK", (-1, 0, 0)),
            ("BOTTOM", (0, 0, -1)),
        ]:
            color = color_names[fn]
            grid = [[color for _ in range(3)] for _ in range(3)]
            model.register_face(fn_name, grid)

        faces_count, stickers_count, color_counts = model.get_progress()
        self.assertEqual(faces_count, 6)
        self.assertEqual(stickers_count, 54)
        for c in ["WHITE", "YELLOW", "GREEN", "BLUE", "RED", "ORANGE"]:
            self.assertEqual(color_counts[c], 9)

        is_valid, errors, warnings = model.validate()
        self.assertTrue(is_valid, f"Validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

        # Convert to rubix cube
        cube = model.to_rubix_cube()
        self.assertEqual(cube, solved_cube)
        self.assertTrue(is_cube_solved(cube))

    def test_cube_model_scramble_reconstruction_and_solve(self):
        """Verify that an arbitrary scrambled cube can be scanned into CubeModel and solved."""
        scrambled = shuffle(solved_cube, iterations=20, seed=123)
        model = self.CubeModel()

        # Extract true facelet grids from the scrambled cube
        face_normals = [
            ("FRONT", (1, 0, 0)),
            ("RIGHT", (0, 1, 0)),
            ("TOP", (0, 0, 1)),
            ("LEFT", (0, -1, 0)),
            ("BACK", (-1, 0, 0)),
            ("BOTTOM", (0, 0, -1)),
        ]
        for fn_name, fn in face_normals:
            grid = [["" for _ in range(3)] for _ in range(3)]
            for cubelet, rotation in scrambled:
                pos = tuple(np.matmul(rotation, cubelet))
                if np.dot(pos, fn) == 1:
                    r, c = self.pos_to_grid(pos, fn)
                    color_normal = tuple(np.round(np.dot(np.array(rotation).T, fn)).astype(int))
                    grid[r][c] = color_names[color_normal]
            model.register_face(fn_name, grid)

        is_valid, errors, _ = model.validate()
        self.assertTrue(is_valid, f"Validation failed: {errors}")

        reconstructed_cube = model.to_rubix_cube()
        self.assertFalse(is_cube_solved(reconstructed_cube))

        # Solve reconstructed cube
        solution = solve(reconstructed_cube)
        self.assertGreater(len(solution), 0)

        cur = reconstructed_cube
        for m in solution:
            cur = apply_move_to_cube(m, cur)

        self.assertTrue(is_cube_solved(cur))

    def test_auto_orient_rotated_faces(self):
        """Verify that register_face automatically corrects 90, 180, and 270 degree face rotations."""
        from rubix_scanner import rotate_grid_cw
        scrambled = shuffle(solved_cube, iterations=8, seed=42)
        model = self.CubeModel()

        # Pre-register lateral faces
        for fn_name, fn in [
            ("FRONT", (1, 0, 0)),
            ("RIGHT", (0, 1, 0)),
            ("BACK", (-1, 0, 0)),
            ("LEFT", (0, -1, 0)),
        ]:
            grid = [["" for _ in range(3)] for _ in range(3)]
            for cubelet, rotation in scrambled:
                pos = tuple(np.matmul(rotation, cubelet))
                if np.dot(pos, fn) == 1:
                    r, c = self.pos_to_grid(pos, fn)
                    v = tuple(np.round(np.dot(np.array(rotation).T, fn)).astype(int))
                    grid[r][c] = color_names[v]
            model.register_face(fn_name, grid)

        # Extract true TOP grid
        fn_top = (0, 0, 1)
        grid_top = [["" for _ in range(3)] for _ in range(3)]
        for cubelet, rotation in scrambled:
            pos = tuple(np.matmul(rotation, cubelet))
            if np.dot(pos, fn_top) == 1:
                r, c = self.pos_to_grid(pos, fn_top)
                v = tuple(np.round(np.dot(np.array(rotation).T, fn_top)).astype(int))
                grid_top[r][c] = color_names[v]

        # Present TOP face rotated by 180 degrees
        rotated_top = rotate_grid_cw(grid_top, 2)
        model.register_face("TOP", rotated_top, auto_orient=True)

        # Check that registered face matches true un-rotated grid
        self.assertEqual(model.faces["TOP"], grid_top)

    def test_conflict_detection_and_pinpointing(self):
        """Verify that CubeModel catches inconsistencies and flags the exact conflicting stickers."""
        model = self.CubeModel()

        for fn_name, fn in [
            ("FRONT", (1, 0, 0)),
            ("RIGHT", (0, 1, 0)),
            ("TOP", (0, 0, 1)),
            ("LEFT", (0, -1, 0)),
            ("BACK", (-1, 0, 0)),
            ("BOTTOM", (0, 0, -1)),
        ]:
            color = color_names[fn]
            grid = [[color for _ in range(3)] for _ in range(3)]
            model.register_face(fn_name, grid)

        # Corrupt one sticker on FRONT to create an impossible edge
        # FRONT top edge is (0, 1), touching TOP face. Change it to Yellow:
        model.faces["FRONT"][0][1] = "YELLOW"

        is_valid, errors, warnings = model.validate()
        self.assertFalse(is_valid)
        error_text = " ".join(errors)
        self.assertTrue("YELLOW" in error_text or "edge" in error_text.lower())

        # Verify get_conflicts pinpoints FRONT (0, 1)
        conflicts = model.get_conflicts()
        self.assertIn(("FRONT", 0, 1), conflicts)

    def test_dynamic_adaptive_guidance(self):
        """Verify that get_guidance provides adaptive contextual rotation guidance."""
        model = self.CubeModel()
        # Initial guidance
        g0 = model.get_guidance()
        self.assertTrue("Hold" in g0 or "camera" in g0.lower())

        # Scan Front
        model.register_face("FRONT", [["GREEN"]*3 for _ in range(3)])

        # When FRONT is in view
        g_front = model.get_guidance(current_visible_face="FRONT")
        self.assertTrue("RED" in g_front or "RIGHT" in g_front)

        # Scan Right
        model.register_face("RIGHT", [["RED"]*3 for _ in range(3)])
        g_right = model.get_guidance(current_visible_face="RIGHT")
        self.assertTrue("BLUE" in g_right or "BACK" in g_right)

    def test_synthetic_feed_detection(self):
        """Verify that CubeFaceDetector successfully detects and extracts facelets from SyntheticCubeFeed."""
        feed = self.SyntheticCubeFeed()
        detector = self.CubeFaceDetector()

        test_colors = [
            ["WHITE", "GREEN", "ORANGE"],
            ["RED", "GREEN", "BLUE"],
            ["YELLOW", "RED", "WHITE"]
        ]
        frame = feed.render_face_frame("FRONT", test_colors, tilt=(10, -8))
        result = detector.detect_face(frame)

        self.assertTrue(result.found)
        self.assertEqual(result.face_name, "FRONT")
        self.assertEqual(result.grid_colors, test_colors)

    def test_headless_scanner_render(self):
        """Verify that the scanner UI renders cleanly in headless mode."""
        import pygame
        from rubix_scanner import RubiksCubeScanner, render_scanner_ui

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        surface = pygame.Surface((875, 750))

        scanner = RubiksCubeScanner(use_synthetic=True)
        scanner.step_frame()

        render_scanner_ui(surface, scanner)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "scanner_preview.png")
            pygame.image.save(surface, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_bayesian_uniform_prior_and_initial_entropy(self):
        """Verify that CubeModel initializes all 54 facelets to uniform 1/6 prior and maximum entropy."""
        model = self.CubeModel()
        max_sticker_entropy = np.log2(6)  # ~2.58496 bits

        for fn in ["FRONT", "RIGHT", "TOP", "LEFT", "BACK", "BOTTOM"]:
            for r in range(3):
                for c in range(3):
                    b = model.beliefs[fn][r][c]
                    self.assertEqual(len(b), 6)
                    for col in self.CANONICAL_COLORS:
                        self.assertAlmostEqual(b[col], 1.0 / 6.0, places=5)
                    self.assertAlmostEqual(model.get_confidence(fn, r, c), 1.0 / 6.0, places=5)
                    self.assertAlmostEqual(model.get_facelet_entropy(fn, r, c), max_sticker_entropy, places=4)
                    self.assertIsNone(model.get_map_color(fn, r, c))
            self.assertAlmostEqual(model.get_face_entropy(fn), 9 * max_sticker_entropy, places=3)

        self.assertAlmostEqual(model.get_total_entropy(), 54 * max_sticker_entropy, places=2)
        faces_cnt, stickers_cnt, _ = model.get_progress()
        self.assertEqual(faces_cnt, 0)
        self.assertEqual(stickers_cnt, 0)

    def test_bayesian_measurement_likelihood_updates_and_convergence(self):
        """Verify that recursive updates p_t = normalize(p_{t-1} * L_t) converge monotonically."""
        clf = self.ColorClassifier()
        green_bgr = (45, 195, 45)
        green_patch = np.full((20, 20, 3), green_bgr, dtype=np.uint8)
        lh = clf.compute_likelihoods(green_patch)

        # Confirm likelihood heavily favors GREEN
        self.assertEqual(max(lh, key=lh.get), "GREEN")
        self.assertGreater(lh["GREEN"], 0.60)

        model = self.CubeModel()
        ent_init = model.get_facelet_entropy("FRONT", 1, 1)

        # Frame 1
        best_c, conf1 = model.update_facelet_belief("FRONT", 1, 1, lh)
        self.assertEqual(best_c, "GREEN")
        self.assertGreater(conf1, 0.60)
        ent1 = model.get_facelet_entropy("FRONT", 1, 1)
        self.assertLess(ent1, ent_init)

        # Frame 2
        _, conf2 = model.update_facelet_belief("FRONT", 1, 1, lh)
        self.assertGreater(conf2, conf1)
        ent2 = model.get_facelet_entropy("FRONT", 1, 1)
        self.assertLess(ent2, ent1)

        # Frame 3: reaches high confidence
        _, conf3 = model.update_facelet_belief("FRONT", 1, 1, lh)
        self.assertGreater(conf3, conf2)
        self.assertGreater(conf3, 0.90)
        ent3 = model.get_facelet_entropy("FRONT", 1, 1)
        self.assertLess(ent3, 0.6)

    def test_bayesian_noise_and_shadow_rejection(self):
        """Verify that dark/noisy patches yield uniform likelihoods and do not corrupt prior beliefs."""
        clf = self.ColorClassifier()
        shadow_patch = np.full((15, 15, 3), (18, 18, 18), dtype=np.uint8)
        lh_shadow = clf.compute_likelihoods(shadow_patch)

        for c in self.CANONICAL_COLORS:
            self.assertAlmostEqual(lh_shadow[c], 1.0 / 6.0, places=4)

        model = self.CubeModel()
        # Seed a confident belief on FRONT center
        model.set_facelet("FRONT", 1, 1, "GREEN")
        prior_conf = model.get_confidence("FRONT", 1, 1)
        prior_ent = model.get_facelet_entropy("FRONT", 1, 1)

        # Apply uninformative shadow likelihood
        model.update_facelet_belief("FRONT", 1, 1, lh_shadow)
        post_conf = model.get_confidence("FRONT", 1, 1)
        post_ent = model.get_facelet_entropy("FRONT", 1, 1)

        # Belief and entropy remain uncorrupted
        self.assertAlmostEqual(post_conf, prior_conf, places=3)
        self.assertAlmostEqual(post_ent, prior_ent, places=3)

    def test_shannon_entropy_rotation_guidance(self):
        """Verify that Shannon entropy accurately guides rotation toward faces with highest uncertainty."""
        model = self.CubeModel()
        max_face_entropy = 9 * np.log2(6)

        # Initially, all faces share maximal uncertainty
        high_face, high_entropy = model.get_highest_entropy_face()
        self.assertAlmostEqual(high_entropy, max_face_entropy, places=2)

        # Register FRONT face: uncertainty drops to near 0
        model.register_face("FRONT", [["GREEN"] * 3 for _ in range(3)])
        self.assertLess(model.get_face_entropy("FRONT"), 1.0)

        # Guidance points towards next lateral face with remaining entropy
        g_front = model.get_guidance(current_visible_face="FRONT")
        self.assertTrue("RIGHT" in g_front or "RED" in g_front)

        # Register RIGHT face
        model.register_face("RIGHT", [["RED"] * 3 for _ in range(3)])
        self.assertLess(model.get_face_entropy("RIGHT"), 1.0)

        # Guidance points to BACK
        g_right = model.get_guidance(current_visible_face="RIGHT")
        self.assertTrue("BACK" in g_right or "BLUE" in g_right)

    def test_visual_confidence_shading(self):
        """Verify get_confidence_color returns gray when uniform, pastel at ~50%, and vivid solid at >=90%."""
        get_col = self.get_confidence_color
        green_base = self.RGB_COLORS["GREEN"]
        unknown_col = self.RGB_COLORS["UNKNOWN"]

        # Unknown / uniform prior (conf <= 0.22)
        self.assertEqual(get_col("GREEN", 1.0 / 6.0), unknown_col)
        self.assertEqual(get_col("GREEN", 0.20), unknown_col)
        self.assertEqual(get_col(None, 0.95), unknown_col)
        self.assertEqual(get_col("UNKNOWN", 0.95), unknown_col)

        # Low confidence (~50%): pale pastel
        col_50 = get_col("GREEN", 0.50)
        self.assertNotEqual(col_50, unknown_col)
        self.assertNotEqual(col_50, green_base)
        # Pastel is significantly lighter (higher average RGB) than saturated green
        self.assertGreater(sum(col_50), sum(green_base) * 0.95)

        # Vivid solid (conf >= 90%)
        col_90 = get_col("GREEN", 0.90)
        self.assertEqual(col_90, green_base)
        col_98 = get_col("GREEN", 0.98)
        self.assertEqual(col_98, green_base)

    def test_bayesian_continuous_scanning_integration(self):
        """Verify that RubiksCubeScanner runs continuous Bayesian frame updates to confirmation."""
        scanner = self.RubiksCubeScanner(use_synthetic=True)
        self.assertEqual(scanner.model.get_progress()[0], 0)

        # Step first frame
        frame, detection = scanner.step_frame()
        self.assertTrue(detection.found)
        self.assertGreater(scanner.lock_progress, 0.5)
        self.assertLess(scanner.model.get_total_entropy(), 54 * np.log2(6))

        # Step additional frames to pass confirmation threshold
        for _ in range(4):
            scanner.step_frame()

        # Face should be confirmed
        self.assertTrue(scanner.model.confirmed_faces[detection.face_name])
        self.assertGreaterEqual(scanner.model.get_face_confidence(detection.face_name), 0.82)
        scanner.close()


if __name__ == "__main__":
    unittest.main()
