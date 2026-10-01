#!/usr/bin/env python3
"""Generate a clean diagram illustrating a Move:
A 90° outer-slice rotation shown mid-move around its center cubelet.
No premature Euclidean space or radian concepts; canonical terminology only.
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

img = Image.new("RGBA", (SW, SH), (15, 17, 23, 255))
draw = ImageDraw.Draw(img)

# Fonts
font_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 22 * SCALE)
font_sub = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 14 * SCALE)
font_stat = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 14 * SCALE)
font_callout_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 16 * SCALE)
font_callout_body = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 14 * SCALE)

# High-contrast color palette
C_BG = (15, 17, 23)
C_CARD_BG = (24, 28, 38)
C_CARD_BORDER = (65, 75, 96)
C_TEXT_TITLE = (255, 255, 255)
C_TEXT_SUB = (205, 215, 230)
C_TEXT_MUTED = (165, 180, 202)

C_DARK_TOP = (58, 66, 82)
C_DARK_FRONT = (46, 53, 67)
C_DARK_RIGHT = (36, 42, 54)
C_DARK_EDGE = (90, 102, 126)

C_ACCENT = (65, 205, 255)       # Electric Cyan
C_ACCENT_FRONT = (38, 165, 225)
C_ACCENT_RIGHT = (25, 130, 185)
C_WHITE = (255, 255, 255)
C_GREEN = (52, 211, 153)
C_GOLD = (250, 204, 21)

# Isometric projection setup
ang30 = math.radians(30)
cos30 = math.cos(ang30)
sin30 = math.sin(ang30)

def project_pt(x, y, z, cx, cy, sz):
    sx = cx + (y - x) * cos30 * sz
    sy = cy + (x + y) * sin30 * sz - z * sz
    return (sx, sy)

# Outer card container
pad = 20 * SCALE
draw.rounded_rectangle([pad, pad, SW - pad, SH - pad], radius=16 * SCALE, fill=C_CARD_BG, outline=C_CARD_BORDER, width=1 * SCALE)

# Header
draw.text((pad + 24 * SCALE, pad + 20 * SCALE), "Outer-Slice Move", fill=C_TEXT_TITLE, font=font_title)
draw.text((pad + 24 * SCALE, pad + 54 * SCALE), "A 90° rotation of an outer slice around its center", fill=C_TEXT_SUB, font=font_sub)

# Badge in top-right: purely "90° Turn" (no radians, no pi/2)
badge_text = "90° Turn"
badge_w = font_stat.getbbox(badge_text)[2]
draw.text((SW - pad - badge_w - 24 * SCALE, pad + 24 * SCALE), badge_text, fill=C_GOLD, font=font_stat)

# Divider line
draw.line([(pad + 20 * SCALE, pad + 86 * SCALE), (SW - pad - 20 * SCALE, pad + 86 * SCALE)], fill=C_CARD_BORDER, width=1 * SCALE)

# 3D Render on left side: Top slice shown mid-move (turned 25 degrees)
cube_cx = pad + 250 * SCALE
cube_cy = pad + 285 * SCALE
cube_sz = 35 * SCALE

gap = 0.06
r = 0.5 - gap
phi = math.radians(25)  # Mid-move rotation angle

quads = []

# 1. Stationary bottom slices (k = -1 and k = 0)
for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        for k in [-1, 0]:
            if k == 0:
                pts = [
                    (i - r, j - r, k + r),
                    (i + r, j - r, k + r),
                    (i + r, j + r, k + r),
                    (i - r, j + r, k + r),
                ]
                depth = sum(x + y + z for x, y, z in pts) / 4.0
                quads.append((depth, pts, (38, 44, 56), (65, 75, 95), 1 * SCALE))

            if i == 1:
                pts = [
                    (i + r, j - r, k - r),
                    (i + r, j + r, k - r),
                    (i + r, j + r, k + r),
                    (i + r, j - r, k + r),
                ]
                depth = sum(x + y + z for x, y, z in pts) / 4.0
                quads.append((depth, pts, C_DARK_FRONT, C_DARK_EDGE, 1 * SCALE))

            if j == 1:
                pts = [
                    (i - r, j + r, k - r),
                    (i + r, j + r, k - r),
                    (i + r, j + r, k + r),
                    (i - r, j + r, k + r),
                ]
                depth = sum(x + y + z for x, y, z in pts) / 4.0
                quads.append((depth, pts, C_DARK_RIGHT, C_DARK_EDGE, 1 * SCALE))

# 2. Rotated top slice (k = 1)
c_p = math.cos(phi)
s_p = math.sin(phi)

for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        k = 1
        # Top face (+Z)
        pts_local = [
            (i - r, j - r, k + r),
            (i + r, j - r, k + r),
            (i + r, j + r, k + r),
            (i - r, j + r, k + r),
        ]
        pts_rot = [(x * c_p - y * s_p, x * s_p + y * c_p, z) for x, y, z in pts_local]
        depth = sum(x + y + z for x, y, z in pts_rot) / 4.0
        quads.append((depth, pts_rot, C_ACCENT, (150, 225, 255), int(1.5 * SCALE)))

        # Front face
        pts_local_f = [
            (i + r, j - r, k - r),
            (i + r, j + r, k - r),
            (i + r, j + r, k + r),
            (i + r, j - r, k + r),
        ]
        pts_rot_f = [(x * c_p - y * s_p, x * s_p + y * c_p, z) for x, y, z in pts_local_f]
        if c_p + s_p > 0:
            depth = sum(x + y + z for x, y, z in pts_rot_f) / 4.0
            quads.append((depth, pts_rot_f, C_ACCENT_FRONT, (100, 190, 235), int(1.5 * SCALE)))

        # Right face
        pts_local_r = [
            (i - r, j + r, k - r),
            (i + r, j + r, k - r),
            (i + r, j + r, k + r),
            (i - r, j + r, k + r),
        ]
        pts_rot_r = [(x * c_p - y * s_p, x * s_p + y * c_p, z) for x, y, z in pts_local_r]
        if c_p - s_p > 0:
            depth = sum(x + y + z for x, y, z in pts_rot_r) / 4.0
            quads.append((depth, pts_rot_r, C_ACCENT_RIGHT, (80, 170, 215), int(1.5 * SCALE)))

        # Back face if turned enough to expose
        if s_p - c_p > 0:
            pts_local_b = [
                (i - r, j - r, k - r),
                (i + r, j - r, k - r),
                (i + r, j - r, k + r),
                (i - r, j - r, k + r),
            ]
            pts_rot_b = [(x * c_p - y * s_p, x * s_p + y * c_p, z) for x, y, z in pts_local_b]
            depth = sum(x + y + z for x, y, z in pts_rot_b) / 4.0
            quads.append((depth, pts_rot_b, (20, 110, 160), (60, 150, 195), int(1.5 * SCALE)))

# Sort quads back to front by depth
quads.sort(key=lambda q: q[0])

for depth, pts_rot, fill, edge, w in quads:
    poly = [project_pt(x, y, z, cube_cx, cube_cy, cube_sz) for x, y, z in pts_rot]
    draw.polygon(poly, fill=fill, outline=edge, width=w)

# Rotation axis line piercing through center cubelet (clean axis, no coordinate labels)
p_axis_bot = project_pt(0, 0, 1.48, cube_cx, cube_cy, cube_sz)
p_axis_top = project_pt(0, 0, 2.75, cube_cx, cube_cy, cube_sz)
draw.line([p_axis_bot, p_axis_top], fill=C_GOLD, width=int(2.5 * SCALE))

# Axis arrowhead
ax_x, ax_y = p_axis_top
draw.polygon([
    (ax_x, ax_y - 12 * SCALE),
    (ax_x - 7 * SCALE, ax_y + 4 * SCALE),
    (ax_x + 7 * SCALE, ax_y + 4 * SCALE)
], fill=C_GOLD)

# Label: purely "Rotation axis"
draw.text((ax_x + 14 * SCALE, ax_y - 10 * SCALE), "Rotation axis", fill=C_GOLD, font=font_stat)

# Curved rotation arrow orbiting the top face around the axis
arc_pts = []
for a_deg in range(20, 115, 2):
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

# Label on arc: purely "90°"
lbl_pt = arc_pts[len(arc_pts) // 2]
draw.text((lbl_pt[0] - 42 * SCALE, lbl_pt[1] - 22 * SCALE), "90°", fill=C_GOLD, font=font_stat)

# Right side: 3 structured concept callouts (canonical terminology only, zero premature coordinate space)
cx_right = pad + 520 * SCALE
callouts = [
    {
        "num": "1",
        "title": "Outer-Slice Rotation",
        "body": "The 9 cubelets in the outer slice rotate together as a single 3D unit.",
        "color": C_ACCENT,
    },
    {
        "num": "2",
        "title": "Face-Center Pivot",
        "body": "The slice rotates 90° around its center cubelet.",
        "color": C_GOLD,
    },
    {
        "num": "3",
        "title": "Fixed Center",
        "body": "The center cubelet spins in place, but never changes its position.",
        "color": C_GREEN,
    },
]

card_y = pad + 110 * SCALE
for c in callouts:
    c_h = 82 * SCALE
    c_w = SW - pad - cx_right - 24 * SCALE

    # Card background
    draw.rounded_rectangle([cx_right, card_y, cx_right + c_w, card_y + c_h], radius=10 * SCALE, fill=(30, 35, 48), outline=(65, 75, 96), width=1 * SCALE)

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
