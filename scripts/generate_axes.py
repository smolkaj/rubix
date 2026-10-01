#!/usr/bin/env python3
"""Generate a clean diagram illustrating the 3D coordinate system and
how the 3 axes physically pierce and anchor the cubelets at the core and centers.
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "fonts"

# Canvas setup
WIDTH, HEIGHT = 840, 520
SCALE = 2
W, H = WIDTH * SCALE, HEIGHT * SCALE

img = Image.new("RGBA", (W, H), (18, 19, 22, 255))
draw = ImageDraw.Draw(img)

# Fonts
font_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 22 * SCALE)
font_subtitle = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 14 * SCALE)
font_axis = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 16 * SCALE)
font_label = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 14 * SCALE)
font_small = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 13 * SCALE)

# Palette
C_BG = (18, 19, 22)
C_TEXT_TITLE = (240, 242, 246)
C_TEXT_SUB = (150, 155, 166)
C_MUTED = (110, 115, 125)

# Axis colors
C_X = (46, 204, 113)   # Front / +X (Green)
C_Y = (231, 76, 60)    # Right / +Y (Red)
C_Z = (245, 247, 250)  # Top / +Z (White)

# Wireframe / cubelet colors
C_DARK_TOP = (32, 35, 42)
C_DARK_FRONT = (26, 28, 34)
C_DARK_RIGHT = (22, 24, 29)
C_DARK_EDGE = (48, 52, 62)

# Centers highlight
C_CENTER_FRONT = (46, 204, 113)  # Green
C_CENTER_RIGHT = (231, 76, 60)   # Red
C_CENTER_TOP = (245, 247, 250)   # White

# Isometric projection setup
rad30 = math.radians(30)
cos30 = math.cos(rad30)
sin30 = math.sin(rad30)

CX = W // 2 - 10 * SCALE
CY = H // 2 + 15 * SCALE
SZ = 44 * SCALE

def project_pt(x, y, z):
    """Project 3D point (x, y, z) to screen coordinates (sx, sy)."""
    sx = CX + (y - x) * cos30 * SZ
    sy = CY + (x + y) * sin30 * SZ - z * SZ
    return (sx, sy)

def draw_arrow(start, end, color, width=3*SCALE, arrow_len=16*SCALE):
    """Draw a line with a sharp arrowhead."""
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return
    angle = math.atan2(dy, dx)
    arrow_angle = math.radians(24)
    p1 = (
        end[0] - arrow_len * math.cos(angle - arrow_angle),
        end[1] - arrow_len * math.sin(angle - arrow_angle),
    )
    p2 = (
        end[0] - arrow_len * math.cos(angle + arrow_angle),
        end[1] - arrow_len * math.sin(angle + arrow_angle),
    )
    draw.polygon([end, p1, p2], fill=color)

def draw_cubelet_face(i, j, k, face_type, fill, outline, width=1):
    gap = 0.05
    r = 0.5 - gap
    if face_type == "top":  # +Z
        p1 = project_pt(i - r, j - r, k + r)
        p2 = project_pt(i + r, j - r, k + r)
        p3 = project_pt(i + r, j + r, k + r)
        p4 = project_pt(i - r, j + r, k + r)
    elif face_type == "front":  # +X
        p1 = project_pt(i + r, j - r, k - r)
        p2 = project_pt(i + r, j + r, k - r)
        p3 = project_pt(i + r, j + r, k + r)
        p4 = project_pt(i + r, j - r, k + r)
    elif face_type == "right":  # +Y
        p1 = project_pt(i - r, j + r, k - r)
        p2 = project_pt(i + r, j + r, k - r)
        p3 = project_pt(i + r, j + r, k + r)
        p4 = project_pt(i - r, j + r, k + r)
    else:
        return
    draw.polygon([p1, p2, p3, p4], fill=fill, outline=outline, width=width)

# Draw title and subtitle
draw.text((36 * SCALE, 22 * SCALE), "Discrete 3D Coordinate Space: Spindle Axes & Fixed Points", fill=C_TEXT_TITLE, font=font_title)
draw.text((36 * SCALE, 52 * SCALE), "The 3 coordinate axes pierce the core at (0,0,0) and the 6 centers, creating invariant Euclidean fixed points.", fill=C_TEXT_SUB, font=font_subtitle)

# Negative axis arrows (drawn behind the cube)
# -Z axis (Bottom)
draw_arrow(project_pt(0, 0, 0), project_pt(0, 0, -2.5), (140, 145, 155), width=2*SCALE, arrow_len=14*SCALE)
# -X axis (Back)
draw_arrow(project_pt(0, 0, 0), project_pt(-2.5, 0, 0), (140, 145, 155), width=2*SCALE, arrow_len=14*SCALE)
# -Y axis (Left)
draw_arrow(project_pt(0, 0, 0), project_pt(0, -2.5, 0), (140, 145, 155), width=2*SCALE, arrow_len=14*SCALE)

# Draw all 27 cubelets from back to front
cubelets = []
for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            cubelets.append((i, j, k))
cubelets.sort(key=lambda c: (c[0] + c[1] + c[2], c[2], c[0], c[1]))

for i, j, k in cubelets:
    # Determine fills
    fill_top = C_DARK_TOP
    fill_front = C_DARK_FRONT
    fill_right = C_DARK_RIGHT
    edge_top = C_DARK_EDGE
    edge_front = C_DARK_EDGE
    edge_right = C_DARK_EDGE
    w_top = 1 * SCALE
    w_front = 1 * SCALE
    w_right = 1 * SCALE

    # Highlight centers
    if i == 0 and j == 0 and k == 1:  # Top Center
        fill_top = C_CENTER_TOP
        edge_top = (255, 255, 255)
        w_top = int(1.5 * SCALE)
    if i == 1 and j == 0 and k == 0:  # Front Center
        fill_front = C_CENTER_FRONT
        edge_front = (120, 255, 180)
        w_front = int(1.5 * SCALE)
    if i == 0 and j == 1 and k == 0:  # Right Center
        fill_right = C_CENTER_RIGHT
        edge_right = (255, 140, 130)
        w_right = int(1.5 * SCALE)

    # Draw exterior visible faces
    if k == 1:
        draw_cubelet_face(i, j, k, "top", fill_top, edge_top, w_top)
    if i == 1:
        draw_cubelet_face(i, j, k, "front", fill_front, edge_front, w_front)
    if j == 1:
        draw_cubelet_face(i, j, k, "right", fill_right, edge_right, w_right)

# Positive axis arrows (drawn emerging out of the centers)
# +Z axis (Top)
draw_arrow(project_pt(0, 0, 0.9), project_pt(0, 0, 2.7), C_Z, width=4*SCALE, arrow_len=18*SCALE)
# +X axis (Front)
draw_arrow(project_pt(0.9, 0, 0), project_pt(2.7, 0, 0), C_X, width=4*SCALE, arrow_len=18*SCALE)
# +Y axis (Right)
draw_arrow(project_pt(0, 0.9, 0), project_pt(0, 2.7, 0), C_Y, width=4*SCALE, arrow_len=18*SCALE)

# Axis labels
# +Z (Top)
pt_pz = project_pt(0, 0, 2.75)
draw.text((pt_pz[0] - 35*SCALE, pt_pz[1] - 32*SCALE), "+Z: Top (0, 0, 1)", fill=C_Z, font=font_axis)

# +X (Front)
pt_px = project_pt(2.75, 0, 0)
draw.text((pt_px[0] - 120*SCALE, pt_px[1] + 12*SCALE), "+X: Front (1, 0, 0)", fill=C_X, font=font_axis)

# +Y (Right)
pt_py = project_pt(0, 2.75, 0)
draw.text((pt_py[0] + 16*SCALE, pt_py[1] + 12*SCALE), "+Y: Right (0, 1, 0)", fill=C_Y, font=font_axis)

# -Z (Bottom)
pt_nz = project_pt(0, 0, -2.55)
draw.text((pt_nz[0] - 40*SCALE, pt_nz[1] + 10*SCALE), "−Z: Bottom (0, 0, -1)", fill=C_MUTED, font=font_small)

# -X (Back)
pt_nx = project_pt(-2.55, 0, 0)
draw.text((pt_nx[0] + 14*SCALE, pt_nx[1] - 22*SCALE), "−X: Back (-1, 0, 0)", fill=C_MUTED, font=font_small)

# -Y (Left)
pt_ny = project_pt(0, -2.55, 0)
draw.text((pt_ny[0] - 130*SCALE, pt_ny[1] - 22*SCALE), "−Y: Left (0, -1, 0)", fill=C_MUTED, font=font_small)

# Bottom callout card
box_w = 480 * SCALE
box_h = 48 * SCALE
bx0 = (W - box_w) // 2
by0 = H - box_h - 18 * SCALE
draw.rounded_rectangle([bx0, by0, bx0 + box_w, by0 + box_h], radius=8*SCALE, fill=(24, 26, 31), outline=(45, 48, 58), width=1*SCALE)
draw.text((bx0 + 20*SCALE, by0 + 8*SCALE), "Invariant: All 7 blocks on the axes (Core + 6 Centers) are fixed points.", fill=C_TEXT_TITLE, font=font_label)
draw.text((bx0 + 20*SCALE, by0 + 26*SCALE), "Rotating an outer slice leaves its axle and center permanently locked in space.", fill=C_TEXT_SUB, font=font_small)

# Save
out_path = REPO_ROOT / "img" / "coordinate-frame.png"
final_img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
final_img.save(out_path, optimize=True)
print(f"Generated {out_path}")
