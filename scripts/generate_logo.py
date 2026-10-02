#!/usr/bin/env python3
"""Generate the Eigencube logo: a 3x3x3 cube whose fixed centers carry color,
with the eigenvector of the top-face rotation rising from the top center.

Geometry follows eigencube.py (+X front, +Y right, +Z top; cubelets at {-1,0,1}^3),
projected orthographically. Writes the logo (img/logo.svg), the GitHub social preview,
the GUI window icon, and web favicons into img/.
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
OUTLINE = (112, 126, 152)  # silhouette rim: lifts the dark cube off dark backgrounds
ARROW = COLORS[color_names[(0, 0, 1)]]  # the axis is drawn in the top center's own color
BACKGROUND = (15, 17, 23)  # dark ground for the opaque images, as in the README diagrams

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


def logo_shapes(weight=1.0):
    """Polygons (points, fill, stroke, stroke_width) in model units, in drawing order.

    `weight` makes the silhouette rim and the arrow bolder, which would otherwise vanish at icon sizes.
    """
    # The cube's silhouette: all corners but the nearest and farthest from the camera.
    silhouette = sorted((np.array(c, float) for c in np.ndindex(2, 2, 2)), key=lambda c: np.dot(c, VIEW))[1:-1]
    rim = [project(3 * c - 1.5) for c in silhouette]
    center = np.mean(rim, axis=0)
    rim.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    shapes = [(rim, BODY, OUTLINE, 0.09 * weight)]
    cubelets = [np.array(index) - 1 for index in np.ndindex(3, 3, 3)]
    for cubelet in cubelets:
        for normal in np.vstack([np.eye(3), -np.eye(3)]):
            # Only outer faces turned toward the camera show; on a convex cube these never overlap,
            # so no depth sorting is needed.
            if np.dot(normal, VIEW) <= 0 or np.dot(cubelet, normal) != 1:
                continue
            u, v = np.roll(normal, 1), np.roll(normal, 2)  # tangents spanning this face
            face = cubelet + 0.5 * normal
            corners = [face + 0.5 * (a * u + b * v) for a, b in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
            shapes.append(([project(p) for p in corners], shade(BODY, normal, 0.58), EDGE, 0.02))
            is_center = np.abs(cubelet).sum() == 1  # the fixed points: only they keep their color
            color = COLORS[color_names[tuple(int(x) for x in normal)]] if is_center else BLANK
            sticker = [face + a * u + b * v for a, b in rounded_square(0.40, 0.09)]
            shapes.append(([project(p) for p in sticker], shade(color, normal, 0.66), None, 0))
    # Eigenvector of every top-face rotation: rises from the top center sticker.
    # The same arrow drawn first with a wider dark stroke gives it a keyline,
    # which keeps the white arrow legible on light backgrounds.
    arrow = arrow_polygons(project((0, 0, 1.5)), project((0, 0, 3.0)), weight)
    for color, stroke_width in [(BODY, 0.133 * weight), (ARROW, 0.05 * weight)]:
        shapes += [(polygon, color, color, stroke_width) for polygon in arrow]
    return shapes


def arrow_polygons(tail, apex, weight=1.0):
    """Shaft and head outlines; the round-join stroke they are drawn with adds the rest of their width."""
    shaft_width, head_length, head_half_width = 0.05 * weight, 0.33 * weight, 0.18 * weight
    tail, apex = np.array(tail), np.array(apex)
    along = (apex - tail) / np.linalg.norm(apex - tail)
    across = np.array([-along[1], along[0]])
    neck = apex - along * head_length
    half = shaft_width / 2
    shaft = [tail + across * half, neck + across * half, neck - across * half, tail - across * half]
    head = [apex, neck + across * head_half_width, neck - across * head_half_width]
    return [[tuple(p) for p in polygon] for polygon in (shaft, head)]


def bounds(shapes, pad):
    # Strokes reach half their width beyond the polygon outline.
    xs = [x + sign * width / 2 for points, _, _, width in shapes for x, _ in points for sign in (1, -1)]
    ys = [y + sign * width / 2 for points, _, _, width in shapes for _, y in points for sign in (1, -1)]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * pad
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return cx - side / 2, cy - side / 2, side


def hexcolor(rgb):
    return "#%02x%02x%02x" % rgb


def write_svg(path):
    shapes = logo_shapes()
    x0, y0, side = bounds(shapes, pad=0.15)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.3f} {y0:.3f} {side:.3f} {side:.3f}" width="256" height="256">']
    for points, fill, stroke, stroke_width in shapes:
        coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        stroke_attrs = f' stroke="{hexcolor(stroke)}" stroke-width="{stroke_width:g}" stroke-linejoin="round"' if stroke else ""
        out.append(f'<polygon points="{coords}" fill="{hexcolor(fill)}"{stroke_attrs}/>')
    out.append("</svg>\n")
    path.write_text("\n".join(out))


def draw_logo(draw, shapes, x, y, size):
    x0, y0, side = bounds(shapes, pad=0.15)
    pixels_per_unit = size / side
    for points, fill, stroke, stroke_width in shapes:
        xy = [(x + (px - x0) * pixels_per_unit, y + (py - y0) * pixels_per_unit) for px, py in points]
        draw.polygon(xy, fill=fill)
        if stroke:  # an SVG round-join stroke, drawn exactly (Pillow's own outlines are off-center)
            radius = stroke_width * pixels_per_unit / 2
            for p, q in zip(xy, xy[1:] + xy[:1]):
                p, q = np.array(p), np.array(q)
                offset = np.array([q[1] - p[1], p[0] - q[0]]) / (np.linalg.norm(q - p) or 1) * radius
                draw.polygon([tuple(p + offset), tuple(q + offset), tuple(q - offset), tuple(p - offset)], fill=stroke)
                draw.ellipse([p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius], fill=stroke)


def write_social_preview(path, scale=3):
    # GitHub's recommended social preview size.
    img = Image.new("RGB", (1280 * scale, 640 * scale), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw_logo(draw, logo_shapes(), 90 * scale, 110 * scale, 420 * scale)
    title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 112 * scale)
    tagline = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 34 * scale)
    draw.text((560 * scale, 210 * scale), "Eigencube", fill=(236, 240, 241), font=title)
    draw.text((566 * scale, 360 * scale), "A Rubik's Cube solver in linear algebra.", fill=(165, 180, 202), font=tagline)
    draw.text((566 * scale, 408 * scale), "Vectors, rotation matrices, and dot products.", fill=(165, 180, 202), font=tagline)
    img.resize((1280, 640), Image.Resampling.LANCZOS).save(path, optimize=True)


def render_icon(shapes, size, background=None, margin=0.0, scale=4):
    """The logo as a square raster image, supersampled for smooth edges."""
    img = Image.new("RGBA", (size * scale, size * scale), background or (0, 0, 0, 0))
    inset = margin * size * scale
    draw_logo(ImageDraw.Draw(img), shapes, inset, inset, size * scale - 2 * inset)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def write_window_icon(path):
    # Window managers shrink this to ~24 px in title bars, so it is drawn bolder than the logo.
    render_icon(logo_shapes(weight=1.8), 256).save(path, optimize=True)


def write_favicon(path):
    # Rendered per size rather than downscaled from one image, so small sizes stay crisp.
    # Below 48 px the rim and arrow get proportionally heavier, or they would shrink below a pixel.
    sizes = [16, 32, 48]
    icons = [render_icon(logo_shapes(weight=max(1.0, 40 / size)), size) for size in sizes]
    icons[-1].save(path, sizes=[(size, size) for size in sizes], append_images=icons[:-1])


def write_apple_touch_icon(path):
    # iOS fills transparency with black and rounds the corners itself, so supply an opaque, padded square.
    render_icon(logo_shapes(), 180, background=BACKGROUND + (255,), margin=0.12).convert("RGB").save(path, optimize=True)


if __name__ == "__main__":
    for name, write in [("logo.svg", write_svg), ("social-preview.png", write_social_preview),
                        ("icon.png", write_window_icon), ("favicon.ico", write_favicon),
                        ("apple-touch-icon.png", write_apple_touch_icon)]:
        out_path = REPO_ROOT / "img" / name
        write(out_path)
        print(f"Generated {out_path}")
