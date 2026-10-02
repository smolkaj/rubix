#!/usr/bin/env python3
"""Generate the Eigencube logo: a 3x3x3 cube whose fixed centers carry color,
with the eigenvector of the top-face rotation rising from the top center.

Geometry follows eigencube.py (+X front, +Y right, +Z top; cubelets at {-1,0,1}^3),
projected orthographically. Writes img/logo.svg and img/social-preview.png.
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "fonts"
sys.path.insert(0, str(REPO_ROOT))
from eigencube import color_names  # noqa: E402
from eigencube_gui import COLORS  # noqa: E402

BODY = (35, 42, 54)
EDGE = (58, 68, 88)
BLANK = (59, 70, 89)  # stickers that are not fixed centers
ARROW = COLORS[color_names[(0, 0, 1)]]  # the top center's color: the axis is part of it

# Camera: looking at the (+X, +Y, +Z) corner, with screen x horizontal so the Z axis stays upright.
VIEW = np.array([1.0, 0.8, 0.78]) / np.linalg.norm([1.0, 0.8, 0.78])
RIGHT = np.cross([0.0, 0.0, 1.0], VIEW) / np.linalg.norm(np.cross([0.0, 0.0, 1.0], VIEW))
UP = np.cross(VIEW, RIGHT)
LIGHT = np.array([0.35, -0.25, 1.0]) / np.linalg.norm([0.35, -0.25, 1.0])


def project(p):
    return (float(np.dot(p, RIGHT)), float(-np.dot(p, UP)))  # image y points down


def shade(rgb, normal, ambient):
    brightness = ambient + (1 - ambient) * max(0.0, float(np.dot(normal, LIGHT)))
    return tuple(min(255, round(channel * brightness)) for channel in rgb)


def rounded_square(half, radius, steps=5):
    points = []
    for sign_x, sign_y, start in [(1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)]:
        for i in range(steps + 1):
            angle = math.radians(start + 90 * i / steps)
            points.append((sign_x * (half - radius) + radius * math.cos(angle),
                           sign_y * (half - radius) + radius * math.sin(angle)))
    return points


def logo_shapes():
    """Polygons (points, fill, stroke, stroke_width) in model units, back to front."""
    shapes = []
    cubelets = [np.array(index) - 1 for index in np.ndindex(3, 3, 3)]
    for cubelet in sorted(cubelets, key=lambda cubelet: np.dot(cubelet, VIEW)):
        for normal in np.vstack([np.eye(3), -np.eye(3)]):
            if np.dot(normal, VIEW) <= 0:
                continue
            u, v = np.roll(normal, 1), np.roll(normal, 2)  # tangents spanning this face
            face = cubelet + 0.5 * normal
            corners = [face + 0.5 * (a * u + b * v) for a, b in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
            shapes.append(([project(p) for p in corners], shade(BODY, normal, 0.58), EDGE, 0.02))
            if np.dot(cubelet, normal) != 1:  # internal face: no sticker
                continue
            is_center = np.abs(cubelet).sum() == 1  # the fixed points: only they keep their color
            color = COLORS[color_names[tuple(int(x) for x in normal)]] if is_center else BLANK
            sticker = [face + a * u + b * v for a, b in rounded_square(0.40, 0.09)]
            shapes.append(([project(p) for p in sticker], shade(color, normal, 0.66), None, 0))
    # Eigenvector of every top-face rotation: rises from the top center sticker.
    # A dark keyline under the white arrow keeps it legible on light backgrounds.
    for width, color in [(0.183, BODY), (0.1, ARROW)]:
        shapes += arrow_shapes((0, 0, 1.5), (0, 0, 2.9), width, color)
    return shapes


def arrow_shapes(start, end, width, color):
    tail, tip = np.array(project(start)), np.array(project(end))
    along = (tip - tail) / np.linalg.norm(tip - tail)
    across = np.array([-along[1], along[0]])
    head = 2.3 * width  # head scales with width, so the keyline's head frames the arrow's
    neck = tip - along * head
    # Half the width comes from a round-join stroke, which softens the corners.
    half_core = width / 4
    shaft = [tail + across * half_core, neck + across * half_core, neck - across * half_core, tail - across * half_core]
    point = [tip + along * head / 2, neck + across * 0.85 * head, neck - across * 0.85 * head]
    return [([tuple(p) for p in polygon], color, color, width / 2) for polygon in (shaft, point)]


def bounds(shapes, pad):
    xs = [x for pts, *_ in shapes for x, _ in pts]
    ys = [y for pts, *_ in shapes for _, y in pts]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * pad
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return cx - side / 2, cy - side / 2, side


def hexcolor(rgb):
    return "#%02x%02x%02x" % rgb


def write_svg(shapes, path):
    x0, y0, side = bounds(shapes, pad=0.15)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.3f} {y0:.3f} {side:.3f} {side:.3f}">']
    for pts, fill, stroke, stroke_width in shapes:
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
        st = f' stroke="{hexcolor(stroke)}" stroke-width="{stroke_width:g}" stroke-linejoin="round"' if stroke else ""
        out.append(f'<polygon points="{points}" fill="{hexcolor(fill)}"{st}/>')
    out.append("</svg>\n")
    path.write_text("\n".join(out))


def draw_logo(draw, shapes, x, y, size):
    x0, y0, side = bounds(shapes, pad=0.15)
    pixels_per_unit = size / side
    for pts, fill, stroke, stroke_width in shapes:
        xy = [(x + (px - x0) * pixels_per_unit, y + (py - y0) * pixels_per_unit) for px, py in pts]
        draw.polygon(xy, fill=fill)
        if stroke:  # an SVG round-join stroke, drawn exactly (Pillow's own outlines are off-center)
            radius = stroke_width * pixels_per_unit / 2
            for p, q in zip(xy, xy[1:] + xy[:1]):
                p, q = np.array(p), np.array(q)
                offset = np.array([q[1] - p[1], p[0] - q[0]]) / (np.linalg.norm(q - p) or 1) * radius
                draw.polygon([tuple(p + offset), tuple(q + offset), tuple(q - offset), tuple(p - offset)], fill=stroke)
                draw.ellipse([p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius], fill=stroke)


def write_social_preview(shapes, path, scale=3):
    # GitHub's recommended social preview size.
    img = Image.new("RGB", (1280 * scale, 640 * scale), (15, 17, 23))
    draw = ImageDraw.Draw(img)
    draw_logo(draw, shapes, 90 * scale, 110 * scale, 420 * scale)
    title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 112 * scale)
    tagline = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 34 * scale)
    draw.text((560 * scale, 210 * scale), "eigencube", fill=(236, 240, 241), font=title)
    draw.text((566 * scale, 360 * scale), "A Rubik's Cube solver in linear algebra.", fill=(165, 180, 202), font=tagline)
    draw.text((566 * scale, 408 * scale), "Vectors, rotation matrices, and dot products.", fill=(165, 180, 202), font=tagline)
    img.resize((1280, 640), Image.Resampling.LANCZOS).save(path, optimize=True)


if __name__ == "__main__":
    shapes = logo_shapes()
    for name, write in [("logo.svg", write_svg), ("social-preview.png", write_social_preview)]:
        out_path = REPO_ROOT / "img" / name
        write(shapes, out_path)
        print(f"Generated {out_path}")
