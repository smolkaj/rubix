import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "fonts"

# Canvas setup - clean, compact, focused
WIDTH, HEIGHT = 1100, 480
SCALE = 2
W, H = WIDTH * SCALE, HEIGHT * SCALE

img = Image.new("RGB", (W, H), (23, 25, 29))
draw = ImageDraw.Draw(img)

# Fonts
font_title = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 26 * SCALE)
font_subtitle = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 17 * SCALE)
font_label = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 18 * SCALE)
font_code = ImageFont.truetype(str(FONTS_DIR / "Roboto-Medium.ttf"), 20 * SCALE)
font_math = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 22 * SCALE)
font_small = ImageFont.truetype(str(FONTS_DIR / "Roboto-Regular.ttf"), 15 * SCALE)
font_small_bold = ImageFont.truetype(str(FONTS_DIR / "Roboto-Bold.ttf"), 15 * SCALE)

# Palette
C_GREEN = (46, 204, 113)
C_RED = (231, 76, 60)
C_WHITE = (245, 247, 250)
C_TEXT = (220, 226, 235)
C_MUTED = (140, 150, 165)
C_ACCENT = (100, 210, 255)
C_BORDER = (45, 50, 60)

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

# Minimal Title
draw.text((60 * SCALE, 30 * SCALE), "The Geometric Trick: Facelet Normals as Matrix Columns", fill=(255, 255, 255), font=font_title)
draw.text((60 * SCALE, 68 * SCALE), "Decomposing cubelet position c isolates each outward facelet normal directly into diag(c)", fill=C_MUTED, font=font_subtitle)

# ============ LEFT: 3D Corner Cubelet with Facelet Normals ============
cube_cx = 240 * SCALE
cube_cy = 240 * SCALE
cube_s = 85 * SCALE

draw_cubelet(cube_cx, cube_cy, cube_s, front_color=C_GREEN, right_color=C_RED, top_color=C_WHITE)

# Top facelet normal
t_center = iso(0, 0, 0.5, cube_cx, cube_cy, cube_s)
draw_arrow(t_center, (t_center[0], t_center[1] - 65*SCALE), C_WHITE, width=4*SCALE, arrow_len=14*SCALE)
draw.text((t_center[0] - 65*SCALE, t_center[1] - 92*SCALE), "n_z = (0, 0, 1)  [Top / White]", fill=C_WHITE, font=font_small_bold)

# Front facelet normal
f_center = iso(0.5, 0, 0, cube_cx, cube_cy, cube_s)
f_end = (f_center[0] - cos30*65*SCALE, f_center[1] + sin30*65*SCALE)
draw_arrow(f_center, f_end, C_GREEN, width=4*SCALE, arrow_len=14*SCALE)
draw.text((f_end[0] - 130*SCALE, f_end[1] + 10*SCALE), "n_x = (1, 0, 0)  [Front / Green]", fill=C_GREEN, font=font_small_bold)

# Right facelet normal
r_center = iso(0, 0.5, 0, cube_cx, cube_cy, cube_s)
r_end = (r_center[0] + cos30*65*SCALE, r_center[1] + sin30*65*SCALE)
draw_arrow(r_center, r_end, C_RED, width=4*SCALE, arrow_len=14*SCALE)
draw.text((r_end[0] + 12*SCALE, r_end[1] + 10*SCALE), "n_y = (0, 1, 0)  [Right / Red]", fill=C_RED, font=font_small_bold)

draw.text((cube_cx - 85*SCALE, cube_cy + 130*SCALE), "Corner Cubelet:  c = (1, 1, 1)", fill=C_ACCENT, font=font_label)

# Divider line
draw.line([(510*SCALE, 115*SCALE), (510*SCALE, 425*SCALE)], fill=C_BORDER, width=1*SCALE)

# ============ RIGHT: The Matrix Equations ============
rx_start = 550 * SCALE

# Equation 1: diag(c) columns
eq1_y = 145 * SCALE
draw.text((rx_start, eq1_y + 30*SCALE), "diag(c) =", fill=C_WHITE, font=font_math)

dm1_x = rx_start + 115 * SCALE
draw_b(dm1_x, eq1_y, 85*SCALE)
draw.text((dm1_x + 18*SCALE, eq1_y + 8*SCALE), "1", fill=C_GREEN, font=font_code)
draw.text((dm1_x + 60*SCALE, eq1_y + 8*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm1_x + 102*SCALE, eq1_y + 8*SCALE), "0", fill=C_MUTED, font=font_code)

draw.text((dm1_x + 18*SCALE, eq1_y + 32*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm1_x + 60*SCALE, eq1_y + 32*SCALE), "1", fill=C_RED, font=font_code)
draw.text((dm1_x + 102*SCALE, eq1_y + 32*SCALE), "0", fill=C_MUTED, font=font_code)

draw.text((dm1_x + 18*SCALE, eq1_y + 56*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm1_x + 60*SCALE, eq1_y + 56*SCALE), "0", fill=C_MUTED, font=font_code)
draw.text((dm1_x + 102*SCALE, eq1_y + 56*SCALE), "1", fill=C_WHITE, font=font_code)
draw_rb(dm1_x + 130*SCALE, eq1_y, 85*SCALE)

# Column labels
draw.text((dm1_x + 12*SCALE, eq1_y + 92*SCALE), "n_x", fill=C_GREEN, font=font_small_bold)
draw.text((dm1_x + 55*SCALE, eq1_y + 92*SCALE), "n_y", fill=C_RED, font=font_small_bold)
draw.text((dm1_x + 95*SCALE, eq1_y + 92*SCALE), "n_z", fill=C_WHITE, font=font_small_bold)

draw.text((dm1_x + 160*SCALE, eq1_y + 30*SCALE), "=   [  n_x    n_y    n_z  ]", fill=C_MUTED, font=font_code)
draw.text((dm1_x + 160*SCALE, eq1_y + 92*SCALE), "3 columns = 3 outward facelet normals", fill=C_ACCENT, font=font_small)

# Equation 2: Rotation action
eq2_y = 290 * SCALE
draw.text((rx_start, eq2_y + 24*SCALE), "R · diag(c) =", fill=C_WHITE, font=font_math)

dm2_x = rx_start + 160 * SCALE
draw_b(dm2_x, eq2_y, 75*SCALE)
draw.text((dm2_x + 18*SCALE, eq2_y + 22*SCALE), "R · n_x", fill=C_GREEN, font=font_code)
draw.text((dm2_x + 115*SCALE, eq2_y + 22*SCALE), "R · n_y", fill=C_RED, font=font_code)
draw.text((dm2_x + 215*SCALE, eq2_y + 22*SCALE), "R · n_z", fill=C_WHITE, font=font_code)
draw_rb(dm2_x + 310*SCALE, eq2_y, 75*SCALE)

draw.text((rx_start, eq2_y + 90*SCALE), "A single matrix multiplication transforms all facelet directions simultaneously.", fill=C_MUTED, font=font_small)

# Save
out_path = REPO_ROOT / "img" / "basis-colors-diag.png"
final_img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
final_img.save(out_path, optimize=True)
print(f"Generated clean minimal {out_path}")
