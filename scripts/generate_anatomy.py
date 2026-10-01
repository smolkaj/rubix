#!/usr/bin/env python3
"""Generate a clean 4-panel diagram illustrating Rubix terminology:
Face, Slice, Cubelet, and Facelet.
"""

from math import cos, sin, radians
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canvas size
WIDTH = 1200
HEIGHT = 380
SCALE = 2  # 2x supersampling for crisp edges

SW = WIDTH * SCALE
SH = HEIGHT * SCALE

img = Image.new("RGBA", (SW, SH), (18, 19, 22, 255))
draw = ImageDraw.Draw(img)

# Fonts
font_dir = Path("/usr/share/fonts")
bold_font_path = None
regular_font_path = None

for path in [
    font_dir / "truetype" / "roboto" / "unhinted" / "Roboto-Bold.ttf",
    font_dir / "truetype" / "dejavu" / "DejaVuSans-Bold.ttf",
]:
    if path.exists():
        bold_font_path = str(path)
        break

for path in [
    font_dir / "truetype" / "roboto" / "unhinted" / "Roboto-Regular.ttf",
    font_dir / "truetype" / "dejavu" / "DejaVuSans.ttf",
]:
    if path.exists():
        regular_font_path = str(path)
        break

font_panel_title = ImageFont.truetype(bold_font_path or "sans-serif", 20 * SCALE)
font_panel_sub = ImageFont.truetype(regular_font_path or "sans-serif", 13 * SCALE)
font_panel_stat = ImageFont.truetype(bold_font_path or "sans-serif", 13 * SCALE)

# Color palette
C_BG = (18, 19, 22)
C_CARD_BG = (24, 26, 30)
C_CARD_BORDER = (42, 45, 52)
C_TEXT_TITLE = (240, 242, 246)
C_TEXT_SUB = (150, 155, 166)
C_TEXT_MUTED = (110, 115, 125)

# Cubelet base colors (muted dark for unhighlighted)
C_DARK_TOP = (40, 43, 50)
C_DARK_FRONT = (32, 34, 40)
C_DARK_RIGHT = (26, 28, 33)
C_DARK_EDGE = (55, 60, 70)

# Highlight colors
C_WHITE = (245, 247, 250)
C_WHITE_FRONT = (200, 205, 212)
C_WHITE_RIGHT = (165, 170, 178)

C_ACCENT = (56, 189, 248)  # Cyan
C_ACCENT_FRONT = (34, 150, 205)
C_ACCENT_RIGHT = (22, 115, 165)

C_GREEN = (46, 204, 113)
C_RED = (231, 76, 60)
C_GOLD = (241, 196, 15)

# Isometric projection setup
# Angles for X, Y, Z
ang30 = radians(30)
cos30 = cos(ang30)
sin30 = sin(ang30)

def project_pt(x, y, z, cx, cy, sz):
    """Project 3D point (x, y, z) to screen coordinates (sx, sy)."""
    # X axis: (-cos30, sin30)
    # Y axis: (cos30, sin30)
    # Z axis: (0, -1)
    sx = cx + (y - x) * cos30 * sz
    sy = cy + (x + y) * sin30 * sz - z * sz
    return (sx, sy)

def draw_cubelet_face(draw, i, j, k, face_type, cx, cy, sz, fill, outline, width=1):
    """Draw a single face (+Z, +X, or +Y) of cubelet (i, j, k).
    Coordinates i, j, k in {-1, 0, 1}.
    """
    gap = 0.06  # Small aesthetic gap between cubelets
    r = 0.5 - gap

    if face_type == "top":  # +Z face
        p1 = project_pt(i - r, j - r, k + r, cx, cy, sz)
        p2 = project_pt(i + r, j - r, k + r, cx, cy, sz)
        p3 = project_pt(i + r, j + r, k + r, cx, cy, sz)
        p4 = project_pt(i - r, j + r, k + r, cx, cy, sz)
    elif face_type == "front":  # +X face
        p1 = project_pt(i + r, j - r, k - r, cx, cy, sz)
        p2 = project_pt(i + r, j + r, k - r, cx, cy, sz)
        p3 = project_pt(i + r, j + r, k + r, cx, cy, sz)
        p4 = project_pt(i + r, j - r, k + r, cx, cy, sz)
    elif face_type == "right":  # +Y face
        p1 = project_pt(i - r, j + r, k - r, cx, cy, sz)
        p2 = project_pt(i + r, j + r, k - r, cx, cy, sz)
        p3 = project_pt(i + r, j + r, k + r, cx, cy, sz)
        p4 = project_pt(i - r, j + r, k + r, cx, cy, sz)
    else:
        return

    draw.polygon([p1, p2, p3, p4], fill=fill, outline=outline, width=width)

def render_cube(draw, cx, cy, sz, mode):
    """Render full 3x3x3 cube in one of 4 modes:
    - 'face': highlight the top face (+Z, all 9 cubelets)
    - 'slice': highlight the top slice (k = 1, all 9 cubelets with depth)
    - 'cubelet': highlight a single corner cubelet (1, 1, 1)
    - 'facelet': highlight a single 1x1 sticker (top sticker of corner)
    """
    # Draw order: back to front so nearer cubelets overlap farther ones.
    # Sorted by sum of coordinates: i + j + k ascending.
    cubelets = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                cubelets.append((i, j, k))
    cubelets.sort(key=lambda c: (c[0] + c[1] + c[2], c[2], c[0], c[1]))

    for i, j, k in cubelets:
        # Determine coloring based on mode
        is_highlight_cubelet = False
        is_highlight_slice = (k == 1)
        is_corner = (i == 1 and j == 1 and k == 1)

        # Base muted fills
        fill_top = C_DARK_TOP
        fill_front = C_DARK_FRONT
        fill_right = C_DARK_RIGHT
        edge_top = C_DARK_EDGE
        edge_front = C_DARK_EDGE
        edge_right = C_DARK_EDGE
        w_top = 1 * SCALE
        w_front = 1 * SCALE
        w_right = 1 * SCALE

        if mode == "face":
            # Highlight only the top 2D surface of k=1
            if k == 1:
                fill_top = C_WHITE
                edge_top = (180, 185, 195)
                w_top = int(1.5 * SCALE)

        elif mode == "slice":
            # Highlight top slice (all faces of k=1)
            if k == 1:
                fill_top = C_ACCENT
                fill_front = C_ACCENT_FRONT
                fill_right = C_ACCENT_RIGHT
                edge_top = (150, 225, 255)
                edge_front = (100, 190, 235)
                edge_right = (80, 170, 215)
                w_top = int(1.5 * SCALE)
                w_front = int(1.5 * SCALE)
                w_right = int(1.5 * SCALE)

        elif mode == "cubelet":
            # Highlight only corner (1, 1, 1) and explode it slightly outward
            if is_corner:
                fill_top = C_WHITE
                fill_front = C_GREEN
                fill_right = C_RED
                edge_top = (255, 255, 255)
                edge_front = (120, 255, 180)
                edge_right = (255, 140, 130)
                w_top = 2 * SCALE
                w_front = 2 * SCALE
                w_right = 2 * SCALE

        elif mode == "facelet":
            # Highlight only a single 1x1 sticker (top of corner (1, 1, 1))
            if is_corner:
                fill_top = C_GOLD
                edge_top = (255, 240, 150)
                w_top = int(2.5 * SCALE)

        # Only draw exterior visible faces
        if k == 1:
            draw_cubelet_face(draw, i, j, k, "top", cx, cy, sz, fill_top, edge_top, w_top)
        if i == 1:
            draw_cubelet_face(draw, i, j, k, "front", cx, cy, sz, fill_front, edge_front, w_front)
        if j == 1:
            draw_cubelet_face(draw, i, j, k, "right", cx, cy, sz, fill_right, edge_right, w_right)


# Define 4 panels
panels = [
    {
        "title": "Face",
        "sub": "2D exterior surface",
        "stat": "1 of 6 sides",
        "stat_col": C_WHITE,
        "mode": "face",
        "desc": "9 coplanar facelets",
    },
    {
        "title": "Slice",
        "sub": "3D mechanical layer",
        "stat": "1 of 9 slices",
        "stat_col": C_ACCENT,
        "mode": "slice",
        "desc": "9 rotating cubelets",
    },
    {
        "title": "Cubelet",
        "sub": "Constituent 3D block",
        "stat": "1 of 27 blocks",
        "stat_col": C_GREEN,
        "mode": "cubelet",
        "desc": "Rigid 1×1×1 body",
    },
    {
        "title": "Facelet",
        "sub": "Surface sticker patch",
        "stat": "1 of 54 stickers",
        "stat_col": C_GOLD,
        "mode": "facelet",
        "desc": "1×1 colored face",
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
    t_w = font_panel_title.getbbox(p["title"])[2]
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

    render_cube(draw, cube_cx, cube_cy, cube_sz, p["mode"])

    # Bottom caption
    cap_w = font_panel_sub.getbbox(p["desc"])[2]
    draw.text((x0 + (card_w - cap_w) // 2, y1 - 28 * SCALE), p["desc"], fill=C_TEXT_MUTED, font=font_panel_sub)

# Save output
out_path = REPO_ROOT / "img" / "cube-anatomy.png"
final_img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
final_img.save(out_path, optimize=True)
print(f"Generated {out_path}")
