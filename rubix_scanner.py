"""Rubik's Cube Webcam & Real-Time Computer Vision Scanner.

Provides Apple-card-scanner style continuous scanning:
- Users hold and rotate the cube naturally in front of the camera (no rigid static grid).
- Real-time contour & perspective tracking detects the cube face, locks onto stickers,
  and overlays an augmented reality (AR) 3x3 grid.
- A live "Current Guess" model displays the evolving cube hypothesis in an unfolded net.
- Real-time dynamic feedback provides step-by-step guidance on how to rotate the cube
  and highlights/resolves any conflicting stickers or lighting misunderstandings.
- Once verified, the scanned cube seamlessly converts into the canonical rubix.py
  algebraic representation for automated solving.
"""

import cv2
import numpy as np
import pygame
import pygame.freetype
from typing import Dict, List, Tuple, Optional
import os
import sys

from rubix import (
    solved_cube,
    color_names,
    unit_vectors,
    norm1,
    tupled,
)

# Canonical face names, outward unit normals, and their center colors
FACE_NORMALS: Dict[str, Tuple[int, int, int]] = {
    "FRONT":  ( 1,  0,  0),  # GREEN
    "RIGHT":  ( 0,  1,  0),  # RED
    "TOP":    ( 0,  0,  1),  # WHITE
    "LEFT":   ( 0, -1,  0),  # ORANGE
    "BACK":   (-1,  0,  0),  # BLUE
    "BOTTOM": ( 0,  0, -1),  # YELLOW
}

NORMAL_TO_FACE: Dict[Tuple[int, int, int], str] = {
    v: k for k, v in FACE_NORMALS.items()
}

COLOR_TO_NORMAL: Dict[str, Tuple[int, int, int]] = {
    v: k for k, v in color_names.items()
}

COLOR_TO_FACE: Dict[str, str] = {
    color: NORMAL_TO_FACE[vec] for color, vec in COLOR_TO_NORMAL.items()
}

# Canonical 12 valid edge color pairs and 8 corner color triplets on a standard cube
VALID_EDGES = set()
for c, _ in solved_cube:
    if norm1(c) == 2:
        colors = tuple(sorted(color_names[v] for v in unit_vectors if np.dot(c, v) == 1))
        VALID_EDGES.add(colors)

VALID_CORNERS = set()
for c, _ in solved_cube:
    if norm1(c) == 3:
        colors = tuple(sorted(color_names[v] for v in unit_vectors if np.dot(c, v) == 1))
        VALID_CORNERS.add(colors)

# Color palettes (RGB for Pygame, BGR for OpenCV)
RGB_COLORS: Dict[str, Tuple[int, int, int]] = {
    "WHITE":      (236, 240, 241),
    "YELLOW":     (241, 196, 15),
    "GREEN":      (46, 204, 113),
    "BLUE":       (52, 152, 219),
    "RED":        (231, 76, 60),
    "ORANGE":     (230, 126, 34),
    "UNKNOWN":    (55, 60, 65),
    "BACKGROUND": (39, 40, 35),
    "PANEL_BG":   (30, 31, 28),
    "BORDER":     (54, 56, 50),
    "TEXT":       (220, 225, 230),
    "TEXT_MUTED": (131, 148, 150),
    "ACCENT":     (46, 204, 113),
    "WARNING":    (243, 156, 18),
    "CYAN":       (26, 188, 156),
}

BGR_COLORS: Dict[str, Tuple[int, int, int]] = {
    name: (rgb[2], rgb[1], rgb[0]) for name, rgb in RGB_COLORS.items()
}


# ==============================================================================
# Coordinate Space Bijections: Grid (row, col) <-> Euclidean 3D Position
# ==============================================================================

def pos_to_grid(pos: Tuple[int, int, int], fn: Tuple[int, int, int]) -> Tuple[int, int]:
    """Maps a 3D cubelet position to its 2D (row, col) on the given face net."""
    if fn[0] != 0:
        x, y = pos[1] + 1, -pos[2] + 1
    elif fn[1] != 0:
        x, y = -pos[0] + 1, -pos[2] + 1
    else:
        x, y = pos[0] + 1, pos[1] + 1
    return (y, x)


def grid_to_pos(fn: Tuple[int, int, int], r: int, c: int) -> Tuple[int, int, int]:
    """Inverts pos_to_grid: maps face normal and (row, col) back to 3D cubelet position."""
    if fn[0] != 0:
        return (fn[0], c - 1, -(r - 1))
    elif fn[1] != 0:
        return (-(c - 1), fn[1], -(r - 1))
    else:
        return (c - 1, r - 1, fn[2])


# ==============================================================================
# Color Classification (CIELAB + HSV Perceptual Metrics)
# ==============================================================================

class ColorClassifier:
    """Perceptually classifies webcam BGR patches into canonical Rubik's cube colors."""

    def __init__(self):
        # Initial reference centers in CIELAB space (L*, a*, b*)
        self.lab_targets = {
            "WHITE":  np.array([230, 128, 128], dtype=np.float32),
            "YELLOW": np.array([210, 110, 200], dtype=np.float32),
            "GREEN":  np.array([160,  80, 155], dtype=np.float32),
            "BLUE":   np.array([120, 135,  70], dtype=np.float32),
            "RED":    np.array([130, 190, 160], dtype=np.float32),
            "ORANGE": np.array([170, 165, 185], dtype=np.float32),
        }

    def classify_bgr(self, bgr: Tuple[int, int, int]) -> Tuple[str, float]:
        """Classify a single BGR color triple into a color name and confidence."""
        pix = np.uint8([[bgr]])
        hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
        lab = cv2.cvtColor(pix, cv2.COLOR_BGR2LAB)[0, 0]

        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])

        # 1. White detection: low saturation and moderate/high lightness
        if s < 65 and L > 115:
            conf = min(1.0, (140 - s) / 80.0)
            return ("WHITE", max(0.5, conf))

        # 2. Hue-based classification with LAB refinement
        if 38 <= h < 88:
            return ("GREEN", 0.95)
        elif 88 <= h < 140:
            return ("BLUE", 0.95)
        elif 23 <= h < 38:
            return ("YELLOW", 0.95)
        elif 9 <= h < 23:
            # Orange vs Yellow/Red: check LAB b* and a*
            if b > 180 and h >= 22:
                return ("YELLOW", 0.85)
            elif a > 175 and h <= 10:
                return ("RED", 0.85)
            return ("ORANGE", 0.95)
        else:
            # Red wraps around hue 0 and 180
            return ("RED", 0.95)

    def classify_patch(self, patch_bgr: np.ndarray) -> Tuple[str, float]:
        """Samples the trimmed median of an image patch to reject glares and borders."""
        if patch_bgr.size == 0:
            return ("UNKNOWN", 0.0)

        # Flatten pixels and compute trimmed median
        pixels = patch_bgr.reshape(-1, 3).astype(np.float32)
        med_bgr = tuple(int(round(x)) for x in np.median(pixels, axis=0))
        return self.classify_bgr(med_bgr)


# ==============================================================================
# Quadrilateral Geometry & Corner Ordering
# ==============================================================================

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 2D points clockwise starting from top-left: [TL, TR, BR, BL]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    center = pts.mean(axis=0)

    # Sort points clockwise by polar angle relative to center
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # Rotate array so top-left (minimum sum x+y) is index 0
    sums = ordered.sum(axis=1)
    tl_idx = int(np.argmin(sums))
    ordered = np.roll(ordered, -tl_idx, axis=0)

    # Confirm clockwise orientation (cross product)
    v1 = ordered[1] - ordered[0]
    v2 = ordered[3] - ordered[0]
    if v1[0] * v2[1] - v1[1] * v2[0] < 0:
        ordered = ordered[[0, 3, 2, 1]]

    return ordered


# ==============================================================================
# Computer Vision Face Detector
# ==============================================================================

class FaceDetectionResult:
    """Holds the result of a single frame's cube face detection."""

    def __init__(
        self,
        found: bool,
        corners: Optional[np.ndarray] = None,
        rectified: Optional[np.ndarray] = None,
        grid_colors: Optional[List[List[str]]] = None,
        face_name: str = "UNKNOWN",
        confidence: float = 0.0,
    ):
        self.found = found
        self.corners = corners
        self.rectified = rectified
        self.grid_colors = grid_colors or [["UNKNOWN"] * 3 for _ in range(3)]
        self.face_name = face_name
        self.confidence = confidence


class CubeFaceDetector:
    """Detects and tracks a 3x3 Rubik's cube face in continuous webcam video."""

    def __init__(self):
        self.classifier = ColorClassifier()
        self.tracked_corners: Optional[np.ndarray] = None
        self.alpha_smooth = 0.35  # Exponential moving average smoothing for corners

    def detect_face(self, frame: np.ndarray) -> FaceDetectionResult:
        """Finds the most prominent 3x3 cube face quad in the frame."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding highlights black plastic borders and sticker edges
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        candidate_quad = None
        max_area = 0.0

        # Method 1: Detect candidate individual sticker quads and take convex hull
        sticker_quads = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < 40000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    d1 = np.linalg.norm(pts[0] - pts[1])
                    d2 = np.linalg.norm(pts[1] - pts[2])
                    ratio = max(d1, d2) / (min(d1, d2) + 1e-5)
                    if ratio < 1.8:
                        sticker_quads.append(pts)

        if len(sticker_quads) >= 8:
            all_pts = np.vstack(sticker_quads)
            hull = cv2.convexHull(all_pts)
            hull_peri = cv2.arcLength(hull, True)
            hull_approx = cv2.approxPolyDP(hull, 0.035 * hull_peri, True)
            if len(hull_approx) == 4 and cv2.isContourConvex(hull_approx):
                candidate_quad = hull_approx.reshape(4, 2)
                max_area = cv2.contourArea(hull)

        # Method 2: If sticker cluster was incomplete, fall back to direct outer quad
        if candidate_quad is None:
            min_area = (w * h) * 0.03
            max_frame_area = (w * h) * 0.85
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_frame_area and area > max_area:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        candidate_quad = approx.reshape(4, 2)
                        max_area = area

        if candidate_quad is None:
            self.tracked_corners = None
            return FaceDetectionResult(found=False)

        # Order corners consistently: TL, TR, BR, BL
        ordered_corners = order_quad_points(candidate_quad)

        # Smooth corners across frames using EMA
        if self.tracked_corners is not None:
            ordered_corners = (
                self.alpha_smooth * ordered_corners
                + (1.0 - self.alpha_smooth) * self.tracked_corners
            )
        self.tracked_corners = ordered_corners

        # Perspective warp to canonical 300x300 image
        warp_size = 300
        dst_pts = np.float32([
            [0, 0],
            [warp_size, 0],
            [warp_size, warp_size],
            [0, warp_size]
        ])
        M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
        warped = cv2.warpPerspective(frame, M, (warp_size, warp_size))

        # Sample 3x3 sticker grid (inner 45% of each 100x100 cell to avoid borders)
        grid_colors = []
        confidences = []
        cell_size = warp_size // 3
        margin = int(cell_size * 0.28)

        for r in range(3):
            row_colors = []
            for c in range(3):
                y1 = r * cell_size + margin
                y2 = (r + 1) * cell_size - margin
                x1 = c * cell_size + margin
                x2 = (c + 1) * cell_size - margin
                patch = warped[y1:y2, x1:x2]
                color_name, conf = self.classifier.classify_patch(patch)
                row_colors.append(color_name)
                confidences.append(conf)
            grid_colors.append(row_colors)

        # Center sticker defines face identity
        center_color = grid_colors[1][1]
        face_name = COLOR_TO_FACE.get(center_color, "UNKNOWN")
        avg_confidence = float(np.mean(confidences))

        return FaceDetectionResult(
            found=True,
            corners=ordered_corners,
            rectified=warped,
            grid_colors=grid_colors,
            face_name=face_name,
            confidence=avg_confidence,
        )


# ==============================================================================
# Live Cube Model Hypothesis & Invariant Validator
# ==============================================================================

class CubeModel:
    """Maintains the live hypothesis of the scanned Rubik's cube."""

    def __init__(self):
        self.faces: Dict[str, List[List[Optional[str]]]] = {
            name: [[None] * 3 for _ in range(3)] for name in FACE_NORMALS
        }
        self.confirmed_faces: Dict[str, bool] = {name: False for name in FACE_NORMALS}
        self.face_scan_counts: Dict[str, int] = {name: 0 for name in FACE_NORMALS}

    def reset(self):
        """Clears all scanned facelets."""
        for name in FACE_NORMALS:
            self.faces[name] = [[None] * 3 for _ in range(3)]
            self.confirmed_faces[name] = False
            self.face_scan_counts[name] = 0

    def register_face(self, face_name: str, grid_colors: List[List[str]]) -> bool:
        """Registers a 3x3 face scan into the model."""
        if face_name not in self.faces:
            return False

        # Set facelets
        self.faces[face_name] = [row[:] for row in grid_colors]
        self.confirmed_faces[face_name] = True
        self.face_scan_counts[face_name] += 1
        return True

    def get_progress(self) -> Tuple[int, int, Dict[str, int]]:
        """Returns (confirmed_faces_count, confirmed_stickers_count, color_counts)."""
        faces_count = sum(1 for conf in self.confirmed_faces.values() if conf)
        stickers_count = 0
        color_counts = {c: 0 for c in ["WHITE", "YELLOW", "GREEN", "BLUE", "RED", "ORANGE"]}

        for face in self.faces.values():
            for row in face:
                for c in row:
                    if c and c in color_counts:
                        color_counts[c] += 1
                        stickers_count += 1

        return (faces_count, stickers_count, color_counts)

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Validates physical Rubik's cube invariants (color counts, edges, corners)."""
        errors = []
        warnings = []
        faces_count, stickers_count, color_counts = self.get_progress()

        if stickers_count < 54:
            missing = 54 - stickers_count
            warnings.append(f"Incomplete: {missing} stickers remaining across unconfirmed faces.")

        # Check color counts
        for c, count in color_counts.items():
            if count > 9:
                errors.append(f"Too many {c} stickers ({count}/9). Lighting or sticker misread.")
            elif faces_count == 6 and count < 9:
                errors.append(f"Missing {c} stickers ({count}/9).")

        # Map facelet colors by (pos, fn)
        facelet_colors: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], str] = {}
        for face_name, grid in self.faces.items():
            fn = FACE_NORMALS[face_name]
            for r in range(3):
                for c in range(3):
                    color = grid[r][c]
                    if color:
                        pos = grid_to_pos(fn, r, c)
                        facelet_colors[(pos, fn)] = color

        # Check edges
        for cubelet, _ in solved_cube:
            if norm1(cubelet) == 2:
                vis_fns = [fn for fn in unit_vectors if np.dot(cubelet, fn) == 1]
                fn1, fn2 = vis_fns[0], vis_fns[1]
                c1 = facelet_colors.get((cubelet, fn1))
                c2 = facelet_colors.get((cubelet, fn2))
                if c1 and c2:
                    if c1 == c2:
                        errors.append(f"Impossible duplicate edge color {c1}-{c2} at {cubelet}.")
                    pair = tuple(sorted([c1, c2]))
                    if pair not in VALID_EDGES:
                        errors.append(f"Impossible edge piece {pair[0]}-{pair[1]} at {cubelet}.")

        # Check corners
        for cubelet, _ in solved_cube:
            if norm1(cubelet) == 3:
                vis_fns = [fn for fn in unit_vectors if np.dot(cubelet, fn) == 1]
                colors = [facelet_colors.get((cubelet, fn)) for fn in vis_fns]
                if all(colors):
                    triplet = tuple(sorted(colors))
                    if triplet not in VALID_CORNERS:
                        errors.append(f"Impossible corner triplet {triplet} at {cubelet}.")

        is_valid = (len(errors) == 0 and stickers_count == 54)
        return (is_valid, errors, warnings)

    def get_guidance(self) -> str:
        """Returns real-time conversational guidance on which face to rotate to next."""
        faces_count, stickers_count, color_counts = self.get_progress()

        is_valid, errors, _ = self.validate()
        if errors:
            return f"⚠️ {errors[0]} Please re-show that face to correct."

        if is_valid:
            return "🎉 Cube completely recognized and verified! Press SPACE or 'Solve' to begin."

        # Structured rotation flow
        if not self.confirmed_faces["FRONT"]:
            return "Hold the FRONT (Green) face up to the camera with White on Top."
        elif not self.confirmed_faces["RIGHT"]:
            return "👉 Front captured! Rotate cube 90° to the RIGHT to show the RED face."
        elif not self.confirmed_faces["BACK"]:
            return "👉 Right captured! Rotate to the RIGHT again to show the BLUE (Back) face."
        elif not self.confirmed_faces["LEFT"]:
            return "👉 Back captured! Rotate to the RIGHT again to show the ORANGE (Left) face."
        elif not self.confirmed_faces["TOP"]:
            return "👉 Lateral faces complete! Tilt the cube DOWN to show the WHITE (Top) face."
        elif not self.confirmed_faces["BOTTOM"]:
            return "👉 Almost done! Tilt the cube UP to show the YELLOW (Bottom) face."
        else:
            return "Analyzing cube state..."

    def to_rubix_cube(self) -> Tuple[Tuple[Tuple[int, int, int], Tuple[Tuple[int, ...], ...]], ...]:
        """Converts the 54 confirmed facelets into canonical rubix.py algebraic cube representation."""
        is_valid, errors, _ = self.validate()
        if not is_valid:
            raise ValueError(f"Cannot convert invalid or incomplete cube: {errors}")

        facelet_colors = {}
        for face_name, grid in self.faces.items():
            fn = FACE_NORMALS[face_name]
            for r in range(3):
                for c in range(3):
                    pos = grid_to_pos(fn, r, c)
                    facelet_colors[(pos, fn)] = grid[r][c]

        by_pos = {}
        for (pos, fn), color in facelet_colors.items():
            by_pos.setdefault(pos, []).append((fn, color))

        cube_dict = {}

        # Centers: Standard SO(3) identity rotation
        for v in unit_vectors:
            cube_dict[v] = tupled(np.eye(3, dtype=int))

        # Edges
        for pos, stickers in by_pos.items():
            k = norm1(pos)
            if k == 2:
                (n1, c1), (n2, c2) = stickers
                v1, v2 = COLOR_TO_NORMAL[c1], COLOR_TO_NORMAL[c2]
                orig_cubelet = tuple(np.array(v1) + np.array(v2))
                n3 = tuple(np.cross(n1, n2))
                v3 = tuple(np.cross(v1, v2))
                V = np.column_stack([v1, v2, v3])
                N = np.column_stack([n1, n2, n3])
                R = tupled(N @ V.T)
                cube_dict[orig_cubelet] = R
            elif k == 3:
                (n1, c1), (n2, c2), (n3, c3) = stickers
                v1, v2, v3 = COLOR_TO_NORMAL[c1], COLOR_TO_NORMAL[c2], COLOR_TO_NORMAL[c3]
                orig_cubelet = tuple(np.array(v1) + np.array(v2) + np.array(v3))
                V = np.column_stack([v1, v2, v3])
                N = np.column_stack([n1, n2, n3])
                R_mat = N @ np.linalg.inv(V)
                R = tupled(np.round(R_mat).astype(int))
                cube_dict[orig_cubelet] = R

        # Return canonical ordered tuple
        return tuple((c, cube_dict[c]) for c, _ in solved_cube)


# ==============================================================================
# Synthetic Video Generator (Camera-Free Testing & Simulation)
# ==============================================================================

class SyntheticCubeFeed:
    """Generates realistic synthetic webcam frames of a rotating Rubik's cube."""

    def __init__(self, size: Tuple[int, int] = (640, 480)):
        self.size = size
        self.face_sequence = ["FRONT", "RIGHT", "BACK", "LEFT", "TOP", "BOTTOM"]
        self.current_face_idx = 0
        self.frame_counter = 0

    def render_face_frame(
        self,
        face_name: str,
        grid_colors: List[List[str]],
        tilt: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Renders a single frame with a perspective-tilted cube face."""
        w, h = self.size
        frame = np.full((h, w, 3), (35, 30, 28), dtype=np.uint8)

        # 300x300 canonical face on black plastic frame
        face_img = np.full((300, 300, 3), (18, 18, 18), dtype=np.uint8)
        for r in range(3):
            for c in range(3):
                col_name = grid_colors[r][c]
                bgr = BGR_COLORS.get(col_name, (100, 100, 100))
                cv2.rectangle(
                    face_img,
                    (c * 100 + 8, r * 100 + 8),
                    ((c + 1) * 100 - 8, (r + 1) * 100 - 8),
                    bgr,
                    -1,
                )

        # Perspective warp
        src = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
        cx, cy = w // 2, h // 2
        half_w = 135
        tx, ty = tilt
        dst = np.float32([
            [cx - half_w + tx, cy - half_w + ty],
            [cx + half_w + tx + 8, cy - half_w - ty],
            [cx + half_w - tx, cy + half_w - ty - 8],
            [cx - half_w - tx - 8, cy + half_w + ty],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(face_img, M, (w, h))

        mask = (warped > 0).any(axis=2)
        frame[mask] = warped[mask]

        # Add simulated hand/lighting texture
        cv2.putText(
            frame,
            "Synthetic Simulation Feed [Press D to toggle]",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )
        return frame

    def get_frame(self, target_cube: Optional[List[Tuple]] = None) -> np.ndarray:
        """Returns the next video frame, automatically rotating faces periodically."""
        self.frame_counter += 1
        # Switch faces every ~35 frames (simulating human rotation)
        if self.frame_counter % 35 == 0:
            self.current_face_idx = (self.current_face_idx + 1) % len(self.face_sequence)

        face_name = self.face_sequence[self.current_face_idx]
        fn = FACE_NORMALS[face_name]

        # Determine colors for this face
        cube = target_cube or solved_cube
        grid = [["" for _ in range(3)] for _ in range(3)]
        for cubelet, rotation in cube:
            pos = tuple(np.matmul(rotation, cubelet))
            if np.dot(pos, fn) == 1:
                r, c = pos_to_grid(pos, fn)
                color_normal = tuple(np.round(np.dot(np.array(rotation).T, fn)).astype(int))
                grid[r][c] = color_names[color_normal]

        # Slight continuous breathing / tilt motion
        t = self.frame_counter * 0.08
        tilt = (int(np.sin(t) * 12), int(np.cos(t) * 8))
        return self.render_face_frame(face_name, grid, tilt=tilt)


# ==============================================================================
# Scanner Controller & State Machine
# ==============================================================================

class RubiksCubeScanner:
    """Manages the video stream, real-time CV detection, and scanning state machine."""

    def __init__(self, use_synthetic: bool = False):
        self.use_synthetic = use_synthetic
        self.cap = None

        if not use_synthetic:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[Scanner] No physical webcam found at index 0. Falling back to synthetic feed.")
                self.use_synthetic = True
                self.cap = None

        self.synthetic_feed = SyntheticCubeFeed()
        self.detector = CubeFaceDetector()
        self.model = CubeModel()

        # Stability accumulator
        self.stability_threshold = 4  # Frames of stable recognition before locking face
        self.candidate_face_name: Optional[str] = None
        self.candidate_grid: Optional[List[List[str]]] = None
        self.candidate_streak = 0
        self.lock_progress = 0.0

        # Latest frame & detection
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detection: Optional[FaceDetectionResult] = None
        self.flash_timer = 0  # Frame countdown for green confirmation flash

    def toggle_synthetic(self):
        """Toggles between real webcam and synthetic demo feed."""
        if self.use_synthetic:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.use_synthetic = False
                print("[Scanner] Switched to physical webcam.")
            else:
                print("[Scanner] Webcam still unavailable.")
        else:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.use_synthetic = True
            print("[Scanner] Switched to synthetic demo feed.")

    def step_frame(self) -> Tuple[np.ndarray, FaceDetectionResult]:
        """Captures and processes a single video frame."""
        if self.use_synthetic:
            frame = self.synthetic_feed.get_frame()
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                frame = self.synthetic_feed.get_frame()
            else:
                frame = cv2.resize(frame, (640, 480))

        detection = self.detector.detect_face(frame)
        self.latest_frame = frame
        self.latest_detection = detection

        if self.flash_timer > 0:
            self.flash_timer -= 1

        # Stability accumulation
        if detection.found and detection.face_name != "UNKNOWN":
            if detection.face_name == self.candidate_face_name:
                self.candidate_streak += 1
                self.lock_progress = min(1.0, self.candidate_streak / self.stability_threshold)
                if self.candidate_streak >= self.stability_threshold:
                    # Lock face!
                    self.model.register_face(detection.face_name, detection.grid_colors)
                    self.flash_timer = 6
                    self.candidate_streak = 0
                    self.lock_progress = 0.0
            else:
                self.candidate_face_name = detection.face_name
                self.candidate_grid = detection.grid_colors
                self.candidate_streak = 1
                self.lock_progress = 1.0 / self.stability_threshold
        else:
            self.candidate_streak = max(0, self.candidate_streak - 1)
            self.lock_progress = self.candidate_streak / self.stability_threshold

        return (frame, detection)

    def close(self):
        """Releases camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ==============================================================================
# UI & Augmented Reality Rendering
# ==============================================================================

def draw_ar_overlay(frame: np.ndarray, detection: FaceDetectionResult, lock_progress: float, flash: bool):
    """Draws sleek Apple-style augmented reality overlays on the webcam image."""
    h, w = frame.shape[:2]

    if not detection.found:
        # Draw sleek search reticle in center
        cx, cy = w // 2, h // 2
        rw, rh = 160, 160
        color = (120, 120, 120)
        cv2.rectangle(frame, (cx - rw, cy - rh), (cx + rw, cy + rh), color, 1)
        # Corner brackets
        blen = 25
        # Top-left
        cv2.line(frame, (cx - rw, cy - rh), (cx - rw + blen, cy - rh), (0, 220, 220), 3)
        cv2.line(frame, (cx - rw, cy - rh), (cx - rw, cy - rh + blen), (0, 220, 220), 3)
        # Top-right
        cv2.line(frame, (cx + rw, cy - rh), (cx + rw - blen, cy - rh), (0, 220, 220), 3)
        cv2.line(frame, (cx + rw, cy - rh), (cx + rw, cy - rh + blen), (0, 220, 220), 3)
        # Bottom-left
        cv2.line(frame, (cx - rw, cy + rh), (cx - rw + blen, cy + rh), (0, 220, 220), 3)
        cv2.line(frame, (cx - rw, cy + rh), (cx - rw, cy + rh - blen), (0, 220, 220), 3)
        # Bottom-right
        cv2.line(frame, (cx + rw, cy + rh), (cx + rw - blen, cy + rh), (0, 220, 220), 3)
        cv2.line(frame, (cx + rw, cy + rh), (cx + rw, cy + rh - blen), (0, 220, 220), 3)
        return

    # Draw detected quad contour with cyan/green glowing polygon
    pts = detection.corners.astype(np.int32).reshape((-1, 1, 2))
    outline_color = (0, 255, 120) if flash else (220, 200, 0)
    thickness = 4 if flash else 2
    cv2.polylines(frame, [pts], True, outline_color, thickness)

    # Project 3x3 grid lines back onto camera frame
    warp_size = 300
    src_grid = []
    for i in (100, 200):
        src_grid.append([0, i])
        src_grid.append([warp_size, i])
        src_grid.append([i, 0])
        src_grid.append([i, warp_size])
    src_pts = np.float32(src_grid).reshape(-1, 1, 2)
    dst_pts = np.float32([[0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]])
    M_inv = cv2.getPerspectiveTransform(dst_pts, detection.corners.astype(np.float32))
    projected = cv2.perspectiveTransform(src_pts, M_inv).reshape(-1, 2).astype(np.int32)

    # Draw internal grid lines
    line_col = (180, 220, 220)
    cv2.line(frame, tuple(projected[0]), tuple(projected[1]), line_col, 1)
    cv2.line(frame, tuple(projected[2]), tuple(projected[3]), line_col, 1)
    cv2.line(frame, tuple(projected[4]), tuple(projected[5]), line_col, 1)
    cv2.line(frame, tuple(projected[6]), tuple(projected[7]), line_col, 1)

    # Draw glowing dots on each detected sticker
    cell_centers = []
    for r in range(3):
        for c in range(3):
            cell_centers.append([c * 100 + 50, r * 100 + 50])
    centers_pts = np.float32(cell_centers).reshape(-1, 1, 2)
    proj_centers = cv2.perspectiveTransform(centers_pts, M_inv).reshape(-1, 2).astype(np.int32)

    idx = 0
    for r in range(3):
        for c in range(3):
            pt = tuple(proj_centers[idx])
            col_name = detection.grid_colors[r][c]
            bgr = BGR_COLORS.get(col_name, (100, 100, 100))
            cv2.circle(frame, pt, 9, (20, 20, 20), -1)
            cv2.circle(frame, pt, 7, bgr, -1)
            idx += 1

    # Status badge above quad
    top_y = min(detection.corners[:, 1])
    center_x = int(np.mean(detection.corners[:, 0]))
    badge_text = f"{detection.face_name} - {int(lock_progress * 100)}%"
    cv2.putText(
        frame,
        badge_text,
        (max(10, center_x - 70), max(30, int(top_y) - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        outline_color,
        2,
    )


def render_scanner_ui(surface: pygame.Surface, scanner: RubiksCubeScanner, font=None, font_bold=None):
    """Renders the complete scanner view onto the Pygame surface."""
    surface.fill(RGB_COLORS["BACKGROUND"])

    # Fonts
    if font is None or font_bold is None:
        try:
            font = pygame.freetype.Font("fonts/Roboto-Light.ttf", 16)
            font_bold = pygame.freetype.Font("fonts/Roboto-Medium.ttf", 18)
            font_title = pygame.freetype.Font("fonts/Roboto-Medium.ttf", 22)
        except Exception:
            font = pygame.freetype.SysFont("Arial", 16)
            font_bold = pygame.freetype.SysFont("Arial", 18, bold=True)
            font_title = pygame.freetype.SysFont("Arial", 22, bold=True)
    else:
        font_title = font_bold

    # 1. Header Bar
    header_rect = pygame.Rect(0, 0, surface.get_width(), 55)
    pygame.draw.rect(surface, RGB_COLORS["PANEL_BG"], header_rect)
    pygame.draw.line(surface, RGB_COLORS["BORDER"], (0, 55), (surface.get_width(), 55), 1)
    font_title.render_to(surface, (25, 17), "Rubik's Cube Scanner", RGB_COLORS["TEXT"])

    # Progress pill in header
    faces_cnt, stickers_cnt, _ = scanner.model.get_progress()
    progress_text = f"{faces_cnt}/6 Faces ({stickers_cnt}/54 Stickers)"
    font.render_to(surface, (surface.get_width() - 250, 20), progress_text, RGB_COLORS["CYAN"])

    # 2. Left Panel: Live Camera Feed with AR
    cam_x, cam_y = 20, 70
    cam_w, cam_h = 460, 345

    if scanner.latest_frame is not None:
        display_frame = scanner.latest_frame.copy()
        if scanner.latest_detection is not None:
            draw_ar_overlay(
                display_frame,
                scanner.latest_detection,
                scanner.lock_progress,
                flash=(scanner.flash_timer > 0),
            )
        # Convert to RGB for Pygame
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (cam_w, cam_h))
        cam_surf = pygame.image.frombuffer(frame_resized.tobytes(), (cam_w, cam_h), "RGB")
        surface.blit(cam_surf, (cam_x, cam_y))
    else:
        pygame.draw.rect(surface, RGB_COLORS["PANEL_BG"], (cam_x, cam_y, cam_w, cam_h))

    pygame.draw.rect(surface, RGB_COLORS["BORDER"], (cam_x, cam_y, cam_w, cam_h), 2)

    # 3. Right Panel: "Current Guess" Unfolded Net View
    net_panel_x, net_panel_y = 500, 70
    net_panel_w, net_panel_h = 355, 345
    pygame.draw.rect(
        surface, RGB_COLORS["PANEL_BG"], (net_panel_x, net_panel_y, net_panel_w, net_panel_h), border_radius=6
    )
    pygame.draw.rect(
        surface, RGB_COLORS["BORDER"], (net_panel_x, net_panel_y, net_panel_w, net_panel_h), 1, border_radius=6
    )

    font_bold.render_to(surface, (net_panel_x + 15, net_panel_y + 12), "Current Cube Guess", RGB_COLORS["TEXT"])

    # Unfolded Net Layout:
    # Top at (1, 0)
    # Left at (0, 1), Front at (1, 1), Right at (2, 1), Back at (3, 1)
    # Bottom at (1, 2)
    face_offsets = {
        "TOP":    (1, 0),
        "LEFT":   (0, 1),
        "FRONT":  (1, 1),
        "RIGHT":  (2, 1),
        "BACK":   (3, 1),
        "BOTTOM": (1, 2),
    }

    sticker_size = 20
    sticker_gap = 2
    face_size = 3 * sticker_size + 2 * sticker_gap
    face_spacing = 8
    base_x = net_panel_x + 18
    base_y = net_panel_y + 40
    sticker_rects = []

    for face_name, (fx, fy) in face_offsets.items():
        fx_pos = base_x + fx * (face_size + face_spacing)
        fy_pos = base_y + fy * (face_size + face_spacing)

        is_confirmed = scanner.model.confirmed_faces[face_name]
        is_active = (
            scanner.latest_detection
            and scanner.latest_detection.found
            and scanner.latest_detection.face_name == face_name
        )

        grid = scanner.model.faces[face_name]
        for r in range(3):
            for c in range(3):
                sx = fx_pos + c * (sticker_size + sticker_gap)
                sy = fy_pos + r * (sticker_size + sticker_gap)
                col_name = grid[r][c]
                rgb = RGB_COLORS.get(col_name or "UNKNOWN", RGB_COLORS["UNKNOWN"])

                # Draw sticker rect
                s_rect = pygame.Rect(sx, sy, sticker_size, sticker_size)
                sticker_rects.append((s_rect, face_name, r, c))
                pygame.draw.rect(surface, rgb, s_rect, border_radius=3)
                pygame.draw.rect(surface, (20, 20, 20), s_rect, 1, border_radius=3)

        # Highlight active face border
        if is_active:
            pygame.draw.rect(
                surface, RGB_COLORS["CYAN"], (fx_pos - 2, fy_pos - 2, face_size + 4, face_size + 4), 2, border_radius=4
            )
        elif is_confirmed:
            pygame.draw.rect(
                surface, RGB_COLORS["ACCENT"], (fx_pos - 2, fy_pos - 2, face_size + 4, face_size + 4), 1, border_radius=4
            )

    # Color counters below net
    _, _, color_tallies = scanner.model.get_progress()
    tally_y = net_panel_y + 270
    tally_x = net_panel_x + 15
    for i, c_name in enumerate(["WHITE", "YELLOW", "GREEN", "BLUE", "RED", "ORANGE"]):
        col_rgb = RGB_COLORS[c_name]
        cnt = color_tallies[c_name]
        bx = tally_x + (i % 3) * 110
        by = tally_y + (i // 3) * 30
        pygame.draw.circle(surface, col_rgb, (bx + 8, by + 10), 6)
        text_col = RGB_COLORS["TEXT"] if cnt <= 9 else RGB_COLORS["WARNING"]
        font.render_to(surface, (bx + 20, by + 2), f"{c_name[:3]}: {cnt}/9", text_col)

    # 4. Bottom Guidance & Feedback Card
    guidance_y = 430
    guidance_w = surface.get_width() - 40
    guidance_h = 135
    pygame.draw.rect(
        surface, RGB_COLORS["PANEL_BG"], (20, guidance_y, guidance_w, guidance_h), border_radius=8
    )
    pygame.draw.rect(
        surface, RGB_COLORS["BORDER"], (20, guidance_y, guidance_w, guidance_h), 1, border_radius=8
    )

    guidance_text = scanner.model.get_guidance()
    is_valid, errors, _ = scanner.model.validate()

    headline_color = RGB_COLORS["ACCENT"] if is_valid else (RGB_COLORS["WARNING"] if errors else RGB_COLORS["TEXT"])
    font_bold.render_to(surface, (35, guidance_y + 15), "Scanner Guidance", RGB_COLORS["TEXT_MUTED"])
    font_bold.render_to(surface, (35, guidance_y + 40), guidance_text, headline_color)

    # Detailed hints
    hint_1 = "• Rotate cube freely; the app automatically tracks faces and validates consistency."
    hint_2 = "• Click any sticker on the right to manually cycle colors if webcam lighting is harsh."
    font.render_to(surface, (35, guidance_y + 75), hint_1, RGB_COLORS["TEXT_MUTED"])
    font.render_to(surface, (35, guidance_y + 98), hint_2, RGB_COLORS["TEXT_MUTED"])

    # 5. Action Buttons
    btn_y = 585
    btn_h = 42

    buttons = {
        "back": pygame.Rect(20, btn_y, 130, btn_h),
        "reset": pygame.Rect(165, btn_y, 140, btn_h),
        "demo": pygame.Rect(320, btn_y, 150, btn_h),
        "solve": pygame.Rect(surface.get_width() - 220, btn_y, 200, btn_h),
    }

    # Draw Back button
    pygame.draw.rect(surface, (60, 65, 70), buttons["back"], border_radius=6)
    font_bold.render_to(surface, (buttons["back"].x + 24, btn_y + 12), "Back (ESC)", RGB_COLORS["TEXT"])

    # Draw Reset button
    pygame.draw.rect(surface, (60, 65, 70), buttons["reset"], border_radius=6)
    font_bold.render_to(surface, (buttons["reset"].x + 22, btn_y + 12), "Reset Scan (R)", RGB_COLORS["TEXT"])

    # Draw Demo button
    demo_bg = RGB_COLORS["CYAN"] if scanner.use_synthetic else (60, 65, 70)
    pygame.draw.rect(surface, demo_bg, buttons["demo"], border_radius=6)
    font_bold.render_to(surface, (buttons["demo"].x + 20, btn_y + 12), "Demo Feed (D)", RGB_COLORS["TEXT"])

    # Draw Solve button
    solve_bg = RGB_COLORS["ACCENT"] if is_valid else (50, 75, 60)
    solve_fg = (255, 255, 255) if is_valid else (120, 140, 130)
    pygame.draw.rect(surface, solve_bg, buttons["solve"], border_radius=6)
    font_bold.render_to(surface, (buttons["solve"].x + 30, btn_y + 12), "Solve Cube (SPACE)", solve_fg)

    return buttons, sticker_rects


# ==============================================================================
# Interactive Runner Loop
# ==============================================================================

def run_scanner(
    screen: Optional[pygame.Surface] = None,
    headless: bool = False,
    use_synthetic: bool = False,
) -> Optional[Tuple]:
    """Runs the interactive Rubik's cube scanner. Returns canonical cube or None."""
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode((875, 650))
        pygame.display.set_caption("Rubik's Cube Scanner")

    scanner = RubiksCubeScanner(use_synthetic=use_synthetic)
    clock = pygame.time.Clock()
    running = True
    scanned_cube = None

    color_cycle = ["WHITE", "YELLOW", "GREEN", "BLUE", "RED", "ORANGE"]

    while running:
        dt = clock.tick(30)
        scanner.step_frame()

        buttons, sticker_rects = render_scanner_ui(screen, scanner)
        pygame.display.flip()

        if headless:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    scanner.model.reset()
                elif event.key == pygame.K_d:
                    scanner.toggle_synthetic()
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    is_valid, _, _ = scanner.model.validate()
                    if is_valid:
                        scanned_cube = scanner.model.to_rubix_cube()
                        running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if buttons["back"].collidepoint(pos):
                    running = False
                elif buttons["reset"].collidepoint(pos):
                    scanner.model.reset()
                elif buttons["demo"].collidepoint(pos):
                    scanner.toggle_synthetic()
                elif buttons["solve"].collidepoint(pos):
                    is_valid, _, _ = scanner.model.validate()
                    if is_valid:
                        scanned_cube = scanner.model.to_rubix_cube()
                        running = False
                else:
                    # Check if user clicked any sticker to cycle its color
                    for s_rect, face_name, r, c in sticker_rects:
                        if s_rect.collidepoint(pos):
                            cur = scanner.model.faces[face_name][r][c]
                            idx = 0 if cur not in color_cycle else (color_cycle.index(cur) + 1) % len(color_cycle)
                            scanner.model.faces[face_name][r][c] = color_cycle[idx]
                            scanner.model.confirmed_faces[face_name] = True
                            break

    scanner.close()
    return scanned_cube


if __name__ == "__main__":
    print("[Rubix Scanner] Launching interactive scanner...")
    scanned = run_scanner()
    if scanned is not None:
        print("[Rubix Scanner] Successfully scanned cube!")
        from rubix import solve
        sol = solve(scanned)
        print(f"[Rubix Scanner] Computed solution in {len(sol)} moves:")
        for m in sol:
            print(" ", m)
