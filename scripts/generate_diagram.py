"""Generate img/basis-colors-diag.png illustrating the geometric intuition behind basis vectors, colors, and diag(c)."""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1800, 800
SCALE = 2
W, H = WIDTH * SCALE, HEIGHT * SCALE

img = Image.new("RGB", (W, H), (21, 23, 26))
draw = ImageDraw.Draw(img)

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "fonts"

# Fonts
font_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 34 * SCALE)
font_subtitle = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 20 * SCALE)
font_section = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 23 * SCALE)
font_label_bold = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 19 * SCALE)
font_label = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 18 * SCALE)
font_body = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 17 * SCALE)
font_code = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 20 * SCALE)
font_math = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 22 * SCALE)
font_small = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 15 * SCALE)
font_small_bold = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 15 * SCALE)

# Colors
C_GREEN = (46, 204, 113)
C_RED = (231, 76, 60)
C_WHITE = (245, 247, 250)
C_BLUE = (52, 152, 219)
C_ORANGE = (230, 126, 34)
C_YELLOW = (241, 196, 15)
C_TEXT = (220, 226, 235)
C_MUTED = (140, 150, 165)
C_ACCENT = (100, 210, 255)
C_BOX_BG = (30, 33, 38)
C_BOX_BORDER = (50, 55, 65)
C_DARK_FACE = (45, 50, 60)

# Header
draw.text((60 * SCALE, 30 * SCALE), "Geometric Intuition: Basis Vectors, Colors, and diag(c)", fill=(255, 255, 255), font=font_title)
draw.text((60 * SCALE, 74 * SCALE), "Why Rubix encodes puzzle state as discrete 3D Euclidean vectors and diagonal matrix decompositions", fill=C_MUTED, font=font_subtitle)

card_y = 120 * SCALE
card_h = 645 * SCALE
card_w = 540 * SCALE
spacing = 30 * SCALE

rad30 = math.radians(30)
cos30 = math.cos(rad30)
sin30 = math.sin(rad30)

def iso(x, y, z, cx, cy, scale):
    u = cx + (y * cos30 - x * cos30) * scale
    v = cy + (x * sin30 + y * sin30 - z) * scale
    return (u, v)

def draw_arrow(start, end, color, width=3*SCALE, arrow_len=14*SCALE):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return
    angle = math.atan2(dy, dx)
    arrow_angle = math.radians(24)
    p1 = (end[0] - arrow_len * math.cos(angle - arrow_angle), end[1] - arrow_len * math.sin(angle - arrow_angle))
    p2 = (end[0] - arrow_len * math.cos(angle + arrow_angle), end[1] - arrow_len * math.sin(angle + arrow_angle))
    draw.polygon([end, p1, p2], fill=color)

def draw_cubelet(cx, cy, s, front_color=C_GREEN, right_color=C_RED, top_color=C_WHITE, outline_color=(20, 22, 25)):
    t_f_l = iso(0.5, -0.5, 0.5, cx, cy, s)
    t_f_r = iso(0.5, 0.5, 0.5, cx, cy, s)
    t_b_r = iso(-0.5, 0.5, 0.5, cx, cy, s)
    t_b_l = iso(-0.5, -0.5, 0.5, cx, cy, s)
    b_f_l = iso(0.5, -0.5, -0.5, cx, cy, s)
    b_f_r = iso(0.5, 0.5, -0.5, cx, cy, s)
    b_b_r = iso(-0.5, 0.5, -0.5, cx, cy, s)
    b_b_l = iso(-0.5, -0.5, -0.5, cx, cy, s)
    
    draw.polygon([t_f_l, t_f_r, t_b_r, t_b_l], fill=top_color, outline=outline_color)
    draw.polygon([t_f_l, t_f_r, b_f_r, b_f_l], fill=front_color, outline=outline_color)
    draw.polygon([t_f_r, t_b_r, b_b_r, b_f_r], fill=right_color, outline=outline_color)
    for p in [(t_f_l, t_f_r), (t_f_r, t_b_r), (t_b_r, t_b_l), (t_b_l, t_f_l),
              (t_f_l, b_f_l), (t_f_r, b_f_r), (t_b_r, b_b_r),
              (b_f_l, b_f_r), (b_f_r, b_b_r)]:
        draw.line(p, fill=outline_color, width=2*SCALE)

def draw_b(x, y, h, w=7*SCALE, color=C_MUTED):
    draw.line([(x + w, y), (x, y), (x, y + h), (x + w, y + h)], fill=color, width=2*SCALE)

def draw_rb(x, y, h, w=7*SCALE, color=C_MUTED):
    draw.line([(x - w, y), (x, y), (x, y + h), (x - w, y + h)], fill=color, width=2*SCALE)

# ================= PANEL 1 =================
p1_x = 60 * SCALE
draw.rounded_rectangle([p1_x, card_y, p1_x + card_w, card_y + card_h], radius=16*SCALE, fill=C_BOX_BG, outline=C_BOX_BORDER, width=2*SCALE)
draw.text((p1_x + 24*SCALE, card_y + 24*SCALE), "1. Colors ARE Center Cubelets", fill=C_WHITE, font=font_section)
draw.text((p1_x + 24*SCALE, card_y + 58*SCALE), "Fixed centers physically define the 3D coordinate axes:", fill=C_MUTED, font=font_body)

cx1 = p1_x + card_w // 2
cy1 = card_y + 260 * SCALE
ax_len = 135 * SCALE

vx = (-cos30 * ax_len, sin30 * ax_len)
vy = (cos30 * ax_len, sin30 * ax_len)
vz = (0, -ax_len)

# Negative axes
draw.line([(cx1, cy1), (cx1 - vx[0]*0.75, cy1 - vx[1]*0.75)], fill=(65, 70, 80), width=2*SCALE)
draw.line([(cx1, cy1), (cx1 - vy[0]*0.75, cy1 - vy[1]*0.75)], fill=(65, 70, 80), width=2*SCALE)
draw.line([(cx1, cy1), (cx1 - vz[0]*0.75, cy1 - vz[1]*0.75)], fill=(65, 70, 80), width=2*SCALE)

# Positive axes with arrows
draw_arrow((cx1, cy1), (cx1 + vx[0], cy1 + vx[1]), C_GREEN, width=4*SCALE)
draw_arrow((cx1, cy1), (cx1 + vy[0], cy1 + vy[1]), C_RED, width=4*SCALE)
draw_arrow((cx1, cy1), (cx1 + vz[0], cy1 + vz[1]), C_WHITE, width=4*SCALE)

# Mini center cubelets at axis tips
cube_s = 22 * SCALE
draw_cubelet(cx1 + vz[0], cy1 + vz[1], cube_s, front_color=C_DARK_FACE, right_color=C_DARK_FACE, top_color=C_WHITE)
draw_cubelet(cx1 + vx[0], cy1 + vx[1], cube_s, front_color=C_GREEN, right_color=C_DARK_FACE, top_color=C_DARK_FACE)
draw_cubelet(cx1 + vy[0], cy1 + vy[1], cube_s, front_color=C_DARK_FACE, right_color=C_RED, top_color=C_DARK_FACE)

# Labels
draw.text((cx1 + vz[0] - 70*SCALE, cy1 + vz[1] - 42*SCALE), "+Z (0,0,1): WHITE", fill=C_WHITE, font=font_label_bold)
draw.text((cx1 + vx[0] - 145*SCALE, cy1 + vx[1] + 16*SCALE), "+X (1,0,0): GREEN", fill=C_GREEN, font=font_label_bold)
draw.text((cx1 + vy[0] + 16*SCALE, cy1 + vy[1] + 16*SCALE), "+Y (0,1,0): RED", fill=C_RED, font=font_label_bold)

draw.text((cx1 - vz[0] + 15*SCALE, cy1 - vz[1]*0.75 - 10*SCALE), "-Z: Yellow", fill=C_YELLOW, font=font_small)
draw.text((cx1 - vx[0]*0.75 + 15*SCALE, cy1 - vx[1]*0.75 - 20*SCALE), "-X: Blue", fill=C_BLUE, font=font_small)
draw.text((cx1 - vy[0]*0.75 - 85*SCALE, cy1 - vy[1]*0.75 - 20*SCALE), "-Y: Orange", fill=C_ORANGE, font=font_small)

# Core mechanism dot
draw.ellipse([cx1 - 7*SCALE, cy1 - 7*SCALE, cx1 + 7*SCALE, cy1 + 7*SCALE], fill=(160, 170, 185))
draw.text((cx1 + 12*SCALE, cy1 - 10*SCALE), "Core (0,0,0)", fill=C_MUTED, font=font_small)

box1_y = card_y + 420 * SCALE
draw.rounded_rectangle([p1_x + 18*SCALE, box1_y, p1_x + card_w - 18*SCALE, card_y + card_h - 18*SCALE], radius=10*SCALE, fill=(24, 26, 30))
notes1 = [
    ("The 6 center cubelets never move", " relative to the core."),
    ("Slice moves rotate perimeter cubelets around them.", ""),
    ("Therefore, ", "colors ARE the constant center vectors:"),
    ("  e_x = Green Center,  e_y = Red Center,  e_z = White Center", ""),
    ("Sticker colors are not arbitrary labels—they are normal", ""),
    ("vectors pointing toward the corresponding center cubelet.", "")
]
for i, (p_bold, p_reg) in enumerate(notes1):
    ty = box1_y + (14 + i*29)*SCALE
    draw.text((p1_x + 28*SCALE, ty), p_bold, fill=C_ACCENT if i == 3 else C_WHITE if i == 2 else C_TEXT, font=font_small_bold if i in (0, 2, 3) else font_small)
    if p_reg:
        w_bold = font_small_bold.getbbox(p_bold)[2]
        draw.text((p1_x + 28*SCALE + w_bold, ty), p_reg, fill=C_TEXT, font=font_small)


# ================= PANEL 2 =================
p2_x = p1_x + card_w + spacing
draw.rounded_rectangle([p2_x, card_y, p2_x + card_w, card_y + card_h], radius=16*SCALE, fill=C_BOX_BG, outline=C_BOX_BORDER, width=2*SCALE)
draw.text((p2_x + 24*SCALE, card_y + 24*SCALE), "2. Vector Decomposition: diag(c)", fill=C_WHITE, font=font_section)
draw.text((p2_x + 24*SCALE, card_y + 58*SCALE), "Decomposing position c isolates each face normal:", fill=C_MUTED, font=font_body)

cube2_cx = p2_x + 95 * SCALE
cube2_cy = card_y + 175 * SCALE
cube2_s = 48 * SCALE
draw_cubelet(cube2_cx, cube2_cy, cube2_s, front_color=C_GREEN, right_color=C_RED, top_color=C_WHITE)

t_center = iso(0, 0, 0.5, cube2_cx, cube2_cy, cube2_s)
draw_arrow(t_center, (t_center[0], t_center[1] - 38*SCALE), C_WHITE, width=3*SCALE, arrow_len=10*SCALE)

f_center = iso(0.5, 0, 0, cube2_cx, cube2_cy, cube2_s)
draw_arrow(f_center, (f_center[0] - cos30*38*SCALE, f_center[1] + sin30*38*SCALE), C_GREEN, width=3*SCALE, arrow_len=10*SCALE)

r_center = iso(0, 0.5, 0, cube2_cx, cube2_cy, cube2_s)
draw_arrow(r_center, (r_center[0] + cos30*38*SCALE, r_center[1] + sin30*38*SCALE), C_RED, width=3*SCALE, arrow_len=10*SCALE)

draw.text((cube2_cx - 45*SCALE, cube2_cy + 52*SCALE), "Corner c = (1, 1, 1)^T", fill=C_ACCENT, font=font_label_bold)

vx_start = p2_x + 195 * SCALE
vy_base = card_y + 120 * SCALE

draw.text((vx_start, vy_base + 22*SCALE), "c =", fill=C_WHITE, font=font_code)

x1 = vx_start + 40*SCALE
draw_b(x1, vy_base, 70*SCALE)
draw.text((x1 + 10*SCALE, vy_base + 3*SCALE), "1", fill=C_GREEN, font=font_code)
draw.text((x1 + 10*SCALE, vy_base + 24*SCALE), "1", fill=C_RED, font=font_code)
draw.text((x1 + 10*SCALE, vy_base + 45*SCALE), "1", fill=C_WHITE, font=font_code)
draw_rb(x1 + 28*SCALE, vy_base, 70*SCALE)

draw.text((x1 + 38*SCALE, vy_base + 22*SCALE), "=", fill=C_WHITE, font=font_code)

x2 = x1 + 60*SCALE
draw_b(x2, vy_base, 70*SCALE)
draw.text((x2 + 10*SCALE, vy_base + 3*SCALE), "1", fill=C_GREEN, font=font_code)
draw.text((x2 + 10*SCALE, vy_base + 24*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((x2 + 10*SCALE, vy_base + 45*SCALE), "0", fill=C_MUTED, font=font_code)
draw_rb(x2 + 28*SCALE, vy_base, 70*SCALE)

draw.text((x2 + 36*SCALE, vy_base + 22*SCALE), "+", fill=C_WHITE, font=font_code)

x3 = x2 + 56*SCALE
draw_b(x3, vy_base, 70*SCALE)
draw.text((x3 + 10*SCALE, vy_base + 3*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((x3 + 10*SCALE, vy_base + 24*SCALE), "1", fill=C_RED, font=font_code)
draw.text((x3 + 10*SCALE, vy_base + 45*SCALE), "0", fill=C_MUTED, font=font_code)
draw_rb(x3 + 28*SCALE, vy_base, 70*SCALE)

draw.text((x3 + 36*SCALE, vy_base + 22*SCALE), "+", fill=C_WHITE, font=font_code)

x4 = x3 + 56*SCALE
draw_b(x4, vy_base, 70*SCALE)
draw.text((x4 + 10*SCALE, vy_base + 3*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((x4 + 10*SCALE, vy_base + 24*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((x4 + 10*SCALE, vy_base + 45*SCALE), "1", fill=C_WHITE, font=font_code)
draw_rb(x4 + 28*SCALE, vy_base, 70*SCALE)

diag_y = card_y + 250 * SCALE
draw.text((p2_x + 30*SCALE, diag_y + 24*SCALE), "diag(c) =", fill=C_WHITE, font=font_math)

dm_x = p2_x + 155 * SCALE
draw_b(dm_x, diag_y, 75*SCALE)

draw.text((dm_x + 18*SCALE, diag_y + 4*SCALE), "1", fill=C_GREEN, font=font_code)
draw.text((dm_x + 60*SCALE, diag_y + 4*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm_x + 102*SCALE, diag_y + 4*SCALE), "0", fill=C_MUTED, font=font_code)

draw.text((dm_x + 18*SCALE, diag_y + 26*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm_x + 60*SCALE, diag_y + 26*SCALE), "1", fill=C_RED, font=font_code)
draw.text((dm_x + 102*SCALE, diag_y + 26*SCALE), "0", fill=C_MUTED, font=font_code)

draw.text((dm_x + 18*SCALE, diag_y + 48*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm_x + 60*SCALE, diag_y + 48*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm_x + 102*SCALE, diag_y + 48*SCALE), "1", fill=C_WHITE, font=font_code)

draw_rb(dm_x + 130*SCALE, diag_y, 75*SCALE)

draw.text((dm_x + 8*SCALE, diag_y + 82*SCALE), "Green", fill=C_GREEN, font=font_small_bold)
draw.text((dm_x + 55*SCALE, diag_y + 82*SCALE), "Red", fill=C_RED, font=font_small_bold)
draw.text((dm_x + 95*SCALE, diag_y + 82*SCALE), "White", fill=C_WHITE, font=font_small_bold)

draw.text((dm_x + 160*SCALE, diag_y + 24*SCALE), "<- 3 columns = 3 face normals", fill=C_MUTED, font=font_small)

edge_y = diag_y + 115 * SCALE
draw.text((p2_x + 30*SCALE, edge_y), "Edge c = (1, 0, 1)^T :", fill=C_ACCENT, font=font_label_bold)
draw.text((p2_x + 220*SCALE, edge_y), "Column 2 is (0,0,0)^T  (uncolored inner side)", fill=C_MUTED, font=font_small)

box2_y = card_y + 420 * SCALE
draw.rounded_rectangle([p2_x + 18*SCALE, box2_y, p2_x + card_w - 18*SCALE, card_y + card_h - 18*SCALE], radius=10*SCALE, fill=(24, 26, 30))
notes2 = [
    ("Coordinates are in {-1, 0, 1}:", " |x| is a binary indicator."),
    ("  |x| = 1 if on an outer face;  0 if interior along that axis.", ""),
    ("The L1 norm counts visible colored faces directly:", ""),
    ("  ||c||_1 = |x| + |y| + |z| = # of visible faces", ""),
    ("Fundamental Invariant:", ""),
    ("  rank(diag(c)) = ||c||_1 = {0: core, 1: center, 2: edge, 3: corner}", "")
]
for i, (p_bold, p_reg) in enumerate(notes2):
    ty = box2_y + (14 + i*29)*SCALE
    col = C_ACCENT if i in (3, 5) else C_WHITE if i in (0, 2, 4) else C_TEXT
    draw.text((p2_x + 28*SCALE, ty), p_bold, fill=col, font=font_small_bold if i in (0, 2, 3, 4, 5) else font_small)
    if p_reg:
        w_bold = font_small_bold.getbbox(p_bold)[2]
        draw.text((p2_x + 28*SCALE + w_bold, ty), p_reg, fill=C_TEXT, font=font_small)


# ================= PANEL 3 =================
p3_x = p2_x + card_w + spacing
draw.rounded_rectangle([p3_x, card_y, p3_x + card_w, card_y + card_h], radius=16*SCALE, fill=C_BOX_BG, outline=C_BOX_BORDER, width=2*SCALE)
draw.text((p3_x + 24*SCALE, card_y + 24*SCALE), "3. One-Shot Rotation: R · diag(c)", fill=C_WHITE, font=font_section)
draw.text((p3_x + 24*SCALE, card_y + 58*SCALE), "Matrix multiplication transforms all face normals at once:", fill=C_MUTED, font=font_body)

r_y = card_y + 115 * SCALE
draw.text((p3_x + 30*SCALE, r_y + 16*SCALE), "R · diag(c) =", fill=C_WHITE, font=font_math)

rx_col = p3_x + 190 * SCALE
draw_b(rx_col, r_y, 65*SCALE)
draw.text((rx_col + 15*SCALE, r_y + 18*SCALE), "R · n_x", fill=C_GREEN, font=font_code)
draw.text((rx_col + 105*SCALE, r_y + 18*SCALE), "R · n_y", fill=C_RED, font=font_code)
draw.text((rx_col + 195*SCALE, r_y + 18*SCALE), "R · n_z", fill=C_WHITE, font=font_code)
draw_rb(rx_col + 285*SCALE, r_y, 65*SCALE)

draw.text((p3_x + 30*SCALE, r_y + 80*SCALE), "Column i is the current 3D normal vector of the", fill=C_TEXT, font=font_small)
draw.text((p3_x + 30*SCALE, r_y + 104*SCALE), "sticker that originally faced axis i in the solved cube.", fill=C_TEXT, font=font_small)

ex_y = r_y + 138 * SCALE
draw.text((p3_x + 30*SCALE, ex_y), "Example: 90° turn around Z-axis:", fill=C_ACCENT, font=font_small_bold)
draw.text((p3_x + 30*SCALE, ex_y + 24*SCALE), "  • Green (1,0,0)  ->  (0, -1, 0)   [now facing Left / Orange]", fill=C_MUTED, font=font_small)
draw.text((p3_x + 30*SCALE, ex_y + 46*SCALE), "  • Red   (0,1,0)  ->  (1,  0, 0)   [now facing Front / Green]", fill=C_MUTED, font=font_small)
draw.text((p3_x + 30*SCALE, ex_y + 68*SCALE), "  • White (0,0,1)  ->  (0,  0, 1)   [still facing Top / White]", fill=C_MUTED, font=font_small)

box_inv_y = ex_y + 102 * SCALE
draw.rounded_rectangle([p3_x + 25*SCALE, box_inv_y, p3_x + card_w - 25*SCALE, box_inv_y + 70*SCALE], radius=10*SCALE, fill=(20, 48, 38), outline=C_GREEN, width=2*SCALE)
draw.text((p3_x + 40*SCALE, box_inv_y + 10*SCALE), "The Solved Invariant:", fill=C_GREEN, font=font_section)
draw.text((p3_x + 40*SCALE, box_inv_y + 38*SCALE), "R · diag(c) = diag(c)", fill=C_WHITE, font=font_code)
draw.text((p3_x + 265*SCALE, box_inv_y + 42*SCALE), "(all faces point home)", fill=C_MUTED, font=font_small)

box3_y = card_y + 420 * SCALE
draw.rounded_rectangle([p3_x + 18*SCALE, box3_y, p3_x + card_w - 18*SCALE, card_y + card_h - 18*SCALE], radius=10*SCALE, fill=(24, 26, 30))
notes3 = [
    ("No sticker permutation tables to maintain.", ""),
    ("No discrete state-machine transitions.", ""),
    ("Dot product with axes immediately determines", " which"),
    ("  face any sticker currently points to.", ""),
    ("90-degree slice moves preserve integer coordinates,", ""),
    ("  keeping all entries in R in {-1, 0, 1}.", "")
]
for i, (p_bold, p_reg) in enumerate(notes3):
    ty = box3_y + (14 + i*29)*SCALE
    draw.text((p3_x + 28*SCALE, ty), p_bold, fill=C_TEXT, font=font_small_bold if i in (0, 1, 4) else font_small)
    if p_reg:
        w_bold = (font_small_bold if i in (0, 1, 4) else font_small).getbbox(p_bold)[2]
        draw.text((p3_x + 28*SCALE + w_bold, ty), p_reg, fill=C_TEXT, font=font_small)

# Output image path
out_path = REPO_ROOT / "img" / "basis-colors-diag.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
final_img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
final_img.save(out_path, optimize=True)
print(f"Successfully generated {out_path}")
