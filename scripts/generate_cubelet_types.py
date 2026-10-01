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

# Cubelet base colors (clearly visible 3D geometry for unhighlighted)
C_DARK_TOP = (58, 66, 82)
C_DARK_FRONT = (46, 53, 67)
C_DARK_RIGHT = (36, 42, 54)
C_DARK_EDGE = (90, 102, 126)

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
        c_core_top = (139, 92, 246)    # Vibrant Purple
        c_core_front = (109, 40, 217)
        c_core_right = (91, 33, 182)
        c_core_edge = (196, 181, 253)

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
        col_grid = (75, 84, 104)
        w_box = int(1.5 * SCALE)
        w_grid = 1 * SCALE

        # Outer vertical edges
        for i in [-1.5, 1.5]:
            for j in [-1.5, 1.5]:
                draw.line([project_pt(i, j, -1.5, cx, cy, sz), project_pt(i, j, 1.5, cx, cy, sz)], fill=col_box, width=w_box)

        # Top boundary polygon
        p_t1 = project_pt(-1.5, -1.5, 1.5, cx, cy, sz)
        p_t2 = project_pt(1.5, -1.5, 1.5, cx, cy, sz)
        p_t3 = project_pt(1.5, 1.5, 1.5, cx, cy, sz)
        p_t4 = project_pt(-1.5, 1.5, 1.5, cx, cy, sz)
        draw.polygon([p_t1, p_t2, p_t3, p_t4], outline=col_box, width=w_box)

        # Front boundary polygon
        p_f1 = project_pt(1.5, -1.5, -1.5, cx, cy, sz)
        p_f2 = project_pt(1.5, 1.5, -1.5, cx, cy, sz)
        p_f3 = project_pt(1.5, 1.5, 1.5, cx, cy, sz)
        p_f4 = project_pt(1.5, -1.5, 1.5, cx, cy, sz)
        draw.polygon([p_f1, p_f2, p_f3, p_f4], outline=col_box, width=w_box)

        # Right boundary polygon
        p_r1 = project_pt(-1.5, 1.5, -1.5, cx, cy, sz)
        p_r2 = project_pt(1.5, 1.5, -1.5, cx, cy, sz)
        p_r3 = project_pt(1.5, 1.5, 1.5, cx, cy, sz)
        p_r4 = project_pt(-1.5, 1.5, 1.5, cx, cy, sz)
        draw.polygon([p_r1, p_r2, p_r3, p_r4], outline=col_box, width=w_box)

        # Grid lines on top, front, right faces
        for u in [-0.5, 0.5]:
            draw.line([project_pt(-1.5, u, 1.5, cx, cy, sz), project_pt(1.5, u, 1.5, cx, cy, sz)], fill=col_grid, width=w_grid)
            draw.line([project_pt(u, -1.5, 1.5, cx, cy, sz), project_pt(u, 1.5, 1.5, cx, cy, sz)], fill=col_grid, width=w_grid)
            draw.line([project_pt(1.5, -1.5, u, cx, cy, sz), project_pt(1.5, 1.5, u, cx, cy, sz)], fill=col_grid, width=w_grid)
            draw.line([project_pt(1.5, u, -1.5, cx, cy, sz), project_pt(1.5, u, 1.5, cx, cy, sz)], fill=col_grid, width=w_grid)
            draw.line([project_pt(-1.5, 1.5, u, cx, cy, sz), project_pt(1.5, 1.5, u, cx, cy, sz)], fill=col_grid, width=w_grid)
            draw.line([project_pt(u, 1.5, -1.5, cx, cy, sz), project_pt(u, 1.5, 1.5, cx, cy, sz)], fill=col_grid, width=w_grid)

    elif mode == "center":
        # Highlight top-center (0, 0, 1): exposes exactly 1 facelet (+Z)
        for i, j, k in cubelets:
            is_center = (i == 0 and j == 0 and k == 1)
            fill_top = C_WHITE if is_center else C_DARK_TOP
            fill_front = C_DARK_FRONT
            fill_right = C_DARK_RIGHT
            edge_top = (255, 255, 255) if is_center else C_DARK_EDGE
            edge_front = C_DARK_EDGE
            edge_right = C_DARK_EDGE
            w_top = 2 * SCALE if is_center else 1 * SCALE
            w_front = 1 * SCALE
            w_right = 1 * SCALE

            if k == 1:
                draw_cubelet_face(draw, i, j, k, "top", cx, cy, sz, fill_top, edge_top, w_top)
            if i == 1:
                draw_cubelet_face(draw, i, j, k, "front", cx, cy, sz, fill_front, edge_front, w_front)
            if j == 1:
                draw_cubelet_face(draw, i, j, k, "right", cx, cy, sz, fill_right, edge_right, w_right)

    elif mode == "edge":
        # Highlight front-top edge (1, 0, 1): exposes exactly 2 facelets (+Z, +X)
        for i, j, k in cubelets:
            is_edge = (i == 1 and j == 0 and k == 1)
            fill_top = C_WHITE if is_edge else C_DARK_TOP
            fill_front = C_GREEN if is_edge else C_DARK_FRONT
            fill_right = C_DARK_RIGHT
            edge_top = (255, 255, 255) if is_edge else C_DARK_EDGE
            edge_front = (120, 255, 180) if is_edge else C_DARK_EDGE
            edge_right = C_DARK_EDGE
            w_top = 2 * SCALE if is_edge else 1 * SCALE
            w_front = 2 * SCALE if is_edge else 1 * SCALE
            w_right = 1 * SCALE

            if k == 1:
                draw_cubelet_face(draw, i, j, k, "top", cx, cy, sz, fill_top, edge_top, w_top)
            if i == 1:
                draw_cubelet_face(draw, i, j, k, "front", cx, cy, sz, fill_front, edge_front, w_front)
            if j == 1:
                draw_cubelet_face(draw, i, j, k, "right", cx, cy, sz, fill_right, edge_right, w_right)

    elif mode == "corner":
        # Highlight front-top-right corner (1, 1, 1): exposes exactly 3 facelets (+Z, +X, +Y)
        for i, j, k in cubelets:
            is_corner = (i == 1 and j == 1 and k == 1)
            fill_top = C_WHITE if is_corner else C_DARK_TOP
            fill_front = C_GREEN if is_corner else C_DARK_FRONT
            fill_right = C_RED if is_corner else C_DARK_RIGHT
            edge_top = (255, 255, 255) if is_corner else C_DARK_EDGE
            edge_front = (120, 255, 180) if is_corner else C_DARK_EDGE
            edge_right = (255, 140, 130) if is_corner else C_DARK_EDGE
            w_top = 2 * SCALE if is_corner else 1 * SCALE
            w_front = 2 * SCALE if is_corner else 1 * SCALE
            w_right = 2 * SCALE if is_corner else 1 * SCALE

            if k == 1:
                draw_cubelet_face(draw, i, j, k, "top", cx, cy, sz, fill_top, edge_top, w_top)
            if i == 1:
                draw_cubelet_face(draw, i, j, k, "front", cx, cy, sz, fill_front, edge_front, w_front)
            if j == 1:
                draw_cubelet_face(draw, i, j, k, "right", cx, cy, sz, fill_right, edge_right, w_right)


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
