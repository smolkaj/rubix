#!/usr/bin/env python3
"""Generate a clean 4-panel diagram illustrating the four Cubelet types:
Corner (3 facelets), Edge (2 facelets), Center (1 facelet), and Core (0 facelets).
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "fonts"

# Canvas size matching cube-anatomy.png
WIDTH = 1200
HEIGHT = 380
SCALE = 2

SW = WIDTH * SCALE
SH = HEIGHT * SCALE

img = Image.new("RGBA", (SW, SH), (18, 19, 22, 255))
draw = ImageDraw.Draw(img)

# Fonts
font_panel_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 21 * SCALE)
font_panel_sub = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 14 * SCALE)
font_panel_stat = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 14 * SCALE)
font_panel_desc = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 14 * SCALE)

# High-contrast color palette
C_BG = (15, 17, 23)
C_CARD_BG = (24, 28, 38)
C_CARD_BORDER = (65, 75, 96)
C_TEXT_TITLE = (255, 255, 255)
C_TEXT_SUB = (205, 215, 230)
C_TEXT_MUTED = (165, 180, 202)

# Cubelet base colors (dark charcoal for unhighlighted faces)
C_DARK_TOP = (42, 48, 60)
C_DARK_FRONT = (32, 37, 48)
C_DARK_RIGHT = (25, 29, 39)
C_DARK_EDGE = (62, 72, 92)

# Toned-down face wash colors (subtle background presence; hero cubelet clearly dominates)
# Top face wash: soft slate-blue wash for the White face (+Z)
C_WASH_TOP = (88, 100, 122)
C_WASH_TOP_EDGE = (125, 142, 168)

# Front face wash: deep muted emerald wash for the Green face (+X)
C_WASH_FRONT = (25, 64, 48)
C_WASH_FRONT_EDGE = (48, 112, 84)

# Right face wash: deep muted ruby wash for the Red face (+Y)
C_WASH_RIGHT = (68, 30, 38)
C_WASH_RIGHT_EDGE = (120, 55, 66)

# Colors for facelets
C_WHITE = (255, 255, 255)
C_GREEN = (52, 211, 153)
C_RED = (248, 113, 113)
C_ACCENT = (65, 205, 255)  # Electric Cyan
C_PURPLE = (192, 132, 252)
C_GOLD = (250, 204, 21)

# Isometric projection setup
ang30 = math.radians(30)
cos30 = math.cos(ang30)
sin30 = math.sin(ang30)

def project_pt(x, y, z, cx, cy, sz):
    sx = cx + (y - x) * cos30 * sz
    sy = cy + (x + y) * sin30 * sz - z * sz
    return (sx, sy)

def draw_cubelet_face(draw, i, j, k, face_type, cx, cy, sz, fill, outline, width=1):
    gap = 0.06
    r = 0.5 - gap
    if face_type == "top":
        p1 = project_pt(i - r, j - r, k + r, cx, cy, sz)
        p2 = project_pt(i + r, j - r, k + r, cx, cy, sz)
        p3 = project_pt(i + r, j + r, k + r, cx, cy, sz)
        p4 = project_pt(i - r, j + r, k + r, cx, cy, sz)
    elif face_type == "front":
        p1 = project_pt(i + r, j - r, k - r, cx, cy, sz)
        p2 = project_pt(i + r, j + r, k - r, cx, cy, sz)
        p3 = project_pt(i + r, j + r, k + r, cx, cy, sz)
        p4 = project_pt(i + r, j - r, k + r, cx, cy, sz)
    elif face_type == "right":
        p1 = project_pt(i - r, j + r, k - r, cx, cy, sz)
        p2 = project_pt(i + r, j + r, k - r, cx, cy, sz)
        p3 = project_pt(i + r, j + r, k + r, cx, cy, sz)
        p4 = project_pt(i - r, j + r, k + r, cx, cy, sz)
    else:
        return
    draw.polygon([p1, p2, p3, p4], fill=fill, outline=outline, width=width)

def render_cubelet_type(draw, cx, cy, sz, mode):
    cubelets = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                cubelets.append((i, j, k))
    cubelets.sort(key=lambda c: (c[0] + c[1] + c[2], c[2], c[0], c[1]))

    if mode == "core":
        # Render the internal core at (0, 0, 0) inside a wireframe cube cage
        # 1. Solid glowing core at origin
        c_core_top = (168, 85, 247)
        c_core_front = (147, 51, 234)
        c_core_right = (126, 34, 206)
        c_core_edge = (216, 180, 254)

        draw_cubelet_face(draw, 0, 0, 0, "top", cx, cy, sz, c_core_top, c_core_edge, int(2 * SCALE))
        draw_cubelet_face(draw, 0, 0, 0, "front", cx, cy, sz, c_core_front, c_core_edge, int(2 * SCALE))
        draw_cubelet_face(draw, 0, 0, 0, "right", cx, cy, sz, c_core_right, c_core_edge, int(2 * SCALE))

        # 2. 3D Cross arm stubs extending from core along +X, +Y, +Z
        r_core = 0.5 - 0.06
        p_c_top = project_pt(0, 0, r_core, cx, cy, sz)
        p_c_front = project_pt(r_core, 0, 0, cx, cy, sz)
        p_c_right = project_pt(0, r_core, 0, cx, cy, sz)

        p_end_z = project_pt(0, 0, 1.5, cx, cy, sz)
        p_end_x = project_pt(1.5, 0, 0, cx, cy, sz)
        p_end_y = project_pt(0, 1.5, 0, cx, cy, sz)

        draw.line([p_c_top, p_end_z], fill=(220, 225, 235), width=int(1.5 * SCALE))
        draw.line([p_c_front, p_end_x], fill=C_GREEN, width=int(1.5 * SCALE))
        draw.line([p_c_right, p_end_y], fill=C_RED, width=int(1.5 * SCALE))

        # 3. Outer wireframe box
        col_box = (100, 112, 135)
        w_box = int(1.5 * SCALE)

        for i in [-1.5, 1.5]:
            for j in [-1.5, 1.5]:
                draw.line([project_pt(i, j, -1.5, cx, cy, sz), project_pt(i, j, 1.5, cx, cy, sz)], fill=col_box, width=w_box)

        for poly in [
            [project_pt(-1.5, -1.5, 1.5, cx, cy, sz), project_pt(1.5, -1.5, 1.5, cx, cy, sz), project_pt(1.5, 1.5, 1.5, cx, cy, sz), project_pt(-1.5, 1.5, 1.5, cx, cy, sz)],
            [project_pt(1.5, -1.5, -1.5, cx, cy, sz), project_pt(1.5, 1.5, -1.5, cx, cy, sz), project_pt(1.5, 1.5, 1.5, cx, cy, sz), project_pt(1.5, -1.5, 1.5, cx, cy, sz)],
            [project_pt(-1.5, 1.5, -1.5, cx, cy, sz), project_pt(1.5, 1.5, -1.5, cx, cy, sz), project_pt(1.5, 1.5, 1.5, cx, cy, sz), project_pt(-1.5, 1.5, 1.5, cx, cy, sz)],
        ]:
            draw.polygon(poly, outline=col_box, width=w_box)
        return

    # Configuration per cubelet type
    active_faces = {
        "corner": {"top", "front", "right"},
        "edge": {"top", "front"},
        "center": {"top"},
    }.get(mode, set())

    target = {
        "corner": (1, 1, 1),
        "edge": (1, 0, 1),
        "center": (0, 0, 1),
    }.get(mode, None)

    # Render cubelets
    for i, j, k in cubelets:
        is_target = (target and i == target[0] and j == target[1] and k == target[2])

        # Top face
        if k == 1:
            if is_target:
                fill_t = C_WHITE
                edge_t = (255, 255, 255)
                w_t = int(2.5 * SCALE)
            elif "top" in active_faces:
                fill_t = C_WASH_TOP
                edge_t = C_WASH_TOP_EDGE
                w_t = 1 * SCALE
            else:
                fill_t = C_DARK_TOP
                edge_t = C_DARK_EDGE
                w_t = 1 * SCALE
            draw_cubelet_face(draw, i, j, k, "top", cx, cy, sz, fill_t, edge_t, w_t)

        # Front face
        if i == 1:
            if is_target:
                fill_f = C_GREEN
                edge_f = (140, 255, 195)
                w_f = int(2.5 * SCALE)
            elif "front" in active_faces:
                fill_f = C_WASH_FRONT
                edge_f = C_WASH_FRONT_EDGE
                w_f = 1 * SCALE
            else:
                fill_f = C_DARK_FRONT
                edge_f = C_DARK_EDGE
                w_f = 1 * SCALE
            draw_cubelet_face(draw, i, j, k, "front", cx, cy, sz, fill_f, edge_f, w_f)

        # Right face
        if j == 1:
            if is_target:
                fill_r = C_RED
                edge_r = (255, 160, 160)
                w_r = int(2.5 * SCALE)
            elif "right" in active_faces:
                fill_r = C_WASH_RIGHT
                edge_r = C_WASH_RIGHT_EDGE
                w_r = 1 * SCALE
            else:
                fill_r = C_DARK_RIGHT
                edge_r = C_DARK_EDGE
                w_r = 1 * SCALE
            draw_cubelet_face(draw, i, j, k, "right", cx, cy, sz, fill_r, edge_r, w_r)

    # Re-draw the target hero cubelet on top to ensure crisp borders
    if target:
        ti, tj, tk = target
        if tk == 1:
            draw_cubelet_face(draw, ti, tj, tk, "top", cx, cy, sz, C_WHITE, (255, 255, 255), int(2.5 * SCALE))
        if ti == 1 and mode in ("corner", "edge"):
            draw_cubelet_face(draw, ti, tj, tk, "front", cx, cy, sz, C_GREEN, (180, 255, 220), int(2.5 * SCALE))
        if tj == 1 and mode == "corner":
            draw_cubelet_face(draw, ti, tj, tk, "right", cx, cy, sz, C_RED, (255, 180, 180), int(2.5 * SCALE))


# Define 4 panels in descending order: 3 -> 2 -> 1 -> 0 faces
panels = [
    {
        "title": "Corner",
        "sub": "Vertex cubelet",
        "stat": "1 of 8 corners",
        "stat_col": C_GOLD,
        "mode": "corner",
        "desc": "Part of 3 faces",
    },
    {
        "title": "Edge",
        "sub": "Border cubelet",
        "stat": "1 of 12 edges",
        "stat_col": C_ACCENT,
        "mode": "edge",
        "desc": "Part of 2 faces",
    },
    {
        "title": "Center",
        "sub": "Face-center cubelet",
        "stat": "1 of 6 centers",
        "stat_col": C_WHITE,
        "mode": "center",
        "desc": "Part of 1 face",
    },
    {
        "title": "Core",
        "sub": "Interior cubelet",
        "stat": "1 of 1 core",
        "stat_col": C_PURPLE,
        "mode": "core",
        "desc": "Part of 0 faces",
    },
]

# Layout dimensions
pad_x = 24 * SCALE
pad_y = 20 * SCALE
card_w = (SW - pad_x * 2 - (len(panels) - 1) * 16 * SCALE) // len(panels)
card_h = SH - pad_y * 2

for idx, p in enumerate(panels):
    x0 = pad_x + idx * (card_w + 16 * SCALE)
    y0 = pad_y
    x1 = x0 + card_w
    y1 = y0 + card_h

    # Card background
    draw.rounded_rectangle([x0, y0, x1, y1], radius=12 * SCALE, fill=C_CARD_BG, outline=C_CARD_BORDER, width=1 * SCALE)

    # Header text
    draw.text((x0 + 18 * SCALE, y0 + 16 * SCALE), p["title"], fill=C_TEXT_TITLE, font=font_panel_title)
    draw.text((x0 + 18 * SCALE, y0 + 44 * SCALE), p["sub"], fill=C_TEXT_SUB, font=font_panel_sub)

    # Stat badge in top right of card
    badge_w = font_panel_stat.getbbox(p["stat"])[2]
    draw.text((x1 - badge_w - 18 * SCALE, y0 + 18 * SCALE), p["stat"], fill=p["stat_col"], font=font_panel_stat)

    # Divider line
    draw.line([(x0 + 16 * SCALE, y0 + 70 * SCALE), (x1 - 16 * SCALE, y0 + 70 * SCALE)], fill=C_CARD_BORDER, width=1 * SCALE)

    # Render cube centered in lower portion of card
    cube_cx = x0 + card_w // 2
    cube_cy = y0 + 195 * SCALE
    cube_sz = 26 * SCALE

    render_cubelet_type(draw, cube_cx, cube_cy, cube_sz, p["mode"])

    # Bottom caption
    cap_w = font_panel_desc.getbbox(p["desc"])[2]
    draw.text((x0 + (card_w - cap_w) // 2, y1 - 28 * SCALE), p["desc"], fill=C_TEXT_MUTED, font=font_panel_desc)

# Save output
out_path = REPO_ROOT / "img" / "cubelet-types.png"
final_img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
final_img.save(out_path, optimize=True)
print(f"Generated {out_path}")
