#!/usr/bin/env python3
"""Generate a clean diagram illustrating a Move:
A 90° outer-slice rotation around a coordinate axis, highlighting the fixed center invariant.
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "fonts"

WIDTH = 1200
HEIGHT = 440
SCALE = 2

SW = WIDTH * SCALE
SH = HEIGHT * SCALE

img = Image.new("RGBA", (SW, SH), (18, 19, 22, 255))
draw = ImageDraw.Draw(img)

# Fonts
font_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 22 * SCALE)
font_sub = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 14 * SCALE)
font_stat = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 14 * SCALE)
font_callout_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 15 * SCALE)
font_callout_body = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 13 * SCALE)

# Color palette
C_BG = (18, 19, 22)
C_CARD_BG = (24, 26, 30)
C_CARD_BORDER = (42, 45, 52)
C_TEXT_TITLE = (240, 242, 246)
C_TEXT_SUB = (150, 155, 166)
C_TEXT_MUTED = (110, 115, 125)

C_DARK_TOP = (40, 43, 50)
C_DARK_FRONT = (32, 34, 40)
C_DARK_RIGHT = (26, 28, 33)
C_DARK_EDGE = (55, 60, 70)

C_ACCENT = (56, 189, 248)       # Cyan
C_ACCENT_FRONT = (34, 150, 205)
C_ACCENT_RIGHT = (22, 115, 165)
C_WHITE = (245, 247, 250)
C_GREEN = (46, 204, 113)
C_GOLD = (241, 196, 15)

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

# Outer card container
pad = 20 * SCALE
draw.rounded_rectangle([pad, pad, SW - pad, SH - pad], radius=16 * SCALE, fill=C_CARD_BG, outline=C_CARD_BORDER, width=1 * SCALE)

# Header
draw.text((pad + 24 * SCALE, pad + 20 * SCALE), "Outer-Slice Move", fill=C_TEXT_TITLE, font=font_title)
draw.text((pad + 24 * SCALE, pad + 54 * SCALE), "A 90° rotation of an outer layer around its coordinate axis", fill=C_TEXT_SUB, font=font_sub)

# Badge in top-right
badge_text = "θ = 90° (π/2)"
badge_w = font_stat.getbbox(badge_text)[2]
draw.text((SW - pad - badge_w - 24 * SCALE, pad + 24 * SCALE), badge_text, fill=C_GOLD, font=font_stat)

# Divider line
draw.line([(pad + 20 * SCALE, pad + 86 * SCALE), (SW - pad - 20 * SCALE, pad + 86 * SCALE)], fill=C_CARD_BORDER, width=1 * SCALE)

# 3D Render on left side
cube_cx = pad + 250 * SCALE
cube_cy = pad + 285 * SCALE
cube_sz = 36 * SCALE

cubelets = []
for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            cubelets.append((i, j, k))
cubelets.sort(key=lambda c: (c[0] + c[1] + c[2], c[2], c[0], c[1]))

for i, j, k in cubelets:
    is_top_slice = (k == 1)

    if is_top_slice:
        fill_top = C_ACCENT
        fill_front = C_ACCENT_FRONT
        fill_right = C_ACCENT_RIGHT
        edge_top = (150, 225, 255)
        edge_front = (100, 190, 235)
        edge_right = (80, 170, 215)
        w_top = int(1.5 * SCALE)
        w_front = int(1.5 * SCALE)
        w_right = int(1.5 * SCALE)
    else:
        fill_top = C_DARK_TOP
        fill_front = C_DARK_FRONT
        fill_right = C_DARK_RIGHT
        edge_top = C_DARK_EDGE
        edge_front = C_DARK_EDGE
        edge_right = C_DARK_EDGE
        w_top = 1 * SCALE
        w_front = 1 * SCALE
        w_right = 1 * SCALE

    if k == 1:
        draw_cubelet_face(draw, i, j, k, "top", cube_cx, cube_cy, cube_sz, fill_top, edge_top, w_top)
    if i == 1:
        draw_cubelet_face(draw, i, j, k, "front", cube_cx, cube_cy, cube_sz, fill_front, edge_front, w_front)
    if j == 1:
        draw_cubelet_face(draw, i, j, k, "right", cube_cx, cube_cy, cube_sz, fill_right, edge_right, w_right)

# Axis line (+Z) piercing through center cubelet (0, 0, 1)
p_axis_bot = project_pt(0, 0, 1.48, cube_cx, cube_cy, cube_sz)
p_axis_top = project_pt(0, 0, 2.8, cube_cx, cube_cy, cube_sz)
draw.line([p_axis_bot, p_axis_top], fill=C_GOLD, width=int(2.5 * SCALE))

# Axis arrowhead (+Z)
ax_x, ax_y = p_axis_top
draw.polygon([
    (ax_x, ax_y - 12 * SCALE),
    (ax_x - 7 * SCALE, ax_y + 4 * SCALE),
    (ax_x + 7 * SCALE, ax_y + 4 * SCALE)
], fill=C_GOLD)

draw.text((ax_x + 14 * SCALE, ax_y - 10 * SCALE), "+Z axis (ez)", fill=C_GOLD, font=font_stat)

# Curved rotation arrow orbiting the top face around the Z-axis
arc_pts = []
for a_deg in range(25, 120, 2):
    rad = math.radians(a_deg)
    x = 1.65 * math.cos(rad)
    y = 1.65 * math.sin(rad)
    z = 1.65
    arc_pts.append(project_pt(x, y, z, cube_cx, cube_cy, cube_sz))

for idx in range(len(arc_pts) - 1):
    draw.line([arc_pts[idx], arc_pts[idx+1]], fill=C_GOLD, width=int(2.5 * SCALE))

# Arrowhead at end of rotation arc
tip = arc_pts[-1]
prev = arc_pts[-4]
dx = tip[0] - prev[0]
dy = tip[1] - prev[1]
L = math.hypot(dx, dy)
if L > 0:
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    ah1 = (tip[0] - 14 * SCALE * ux + 7 * SCALE * nx, tip[1] - 14 * SCALE * uy + 7 * SCALE * ny)
    ah2 = (tip[0] - 14 * SCALE * ux - 7 * SCALE * nx, tip[1] - 14 * SCALE * uy - 7 * SCALE * ny)
    draw.polygon([tip, ah1, ah2], fill=C_GOLD)

# Rotation angle label next to arc
lbl_pt = arc_pts[len(arc_pts) // 2]
draw.text((lbl_pt[0] - 55 * SCALE, lbl_pt[1] - 22 * SCALE), "90° Turn", fill=C_GOLD, font=font_stat)

# Right side: 3 structured concept callouts
cx_right = pad + 520 * SCALE
callouts = [
    {
        "num": "1",
        "title": "Layer Rotation",
        "body": "The 9 cubelets in the outer slice rotate together as a single 3D unit.",
        "color": C_ACCENT,
    },
    {
        "num": "2",
        "title": "Axis-Aligned Pivot",
        "body": "The slice turns 90° (π/2) around the coordinate axis normal to the face.",
        "color": C_GOLD,
    },
    {
        "num": "3",
        "title": "Fixed Center Invariant",
        "body": "The center cubelet sits on the rotation axis—it spins in place, but its (x, y, z) position never changes.",
        "color": C_GREEN,
    },
]

card_y = pad + 110 * SCALE
for c in callouts:
    c_h = 82 * SCALE
    c_w = SW - pad - cx_right - 24 * SCALE

    # Card background
    draw.rounded_rectangle([cx_right, card_y, cx_right + c_w, card_y + c_h], radius=10 * SCALE, fill=(30, 33, 40), outline=(48, 52, 64), width=1 * SCALE)

    # Number pill
    pill_r = 13 * SCALE
    pill_cx = cx_right + 26 * SCALE
    pill_cy = card_y + 28 * SCALE
    draw.ellipse([pill_cx - pill_r, pill_cy - pill_r, pill_cx + pill_r, pill_cy + pill_r], fill=c["color"])
    draw.text((pill_cx - 5 * SCALE, pill_cy - 8 * SCALE), c["num"], fill=(18, 19, 22), font=font_stat)

    # Text
    draw.text((cx_right + 54 * SCALE, card_y + 16 * SCALE), c["title"], fill=C_TEXT_TITLE, font=font_callout_title)
    draw.text((cx_right + 54 * SCALE, card_y + 44 * SCALE), c["body"], fill=C_TEXT_SUB, font=font_callout_body)

    card_y += c_h + 14 * SCALE

# Save output
out_path = REPO_ROOT / "img" / "outer-slice-move.png"
final_img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
final_img.save(out_path, optimize=True)
print(f"Generated {out_path}")
