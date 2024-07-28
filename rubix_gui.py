import pygame
import pygame.freetype
import numpy as np
from rubix import solved_cube, apply_move_to_cube, shuffle, solve, moves, color_names, describe_move, is_cubelet_solved, NUM_CUBELETS


pygame.init()
pygame.display.set_caption("Rubik's Cube Solver")
pygame.key.set_repeat(300, 50)  # delay, interval

WIDTH, HEIGHT, TEXT_SIZE = 875, 750, 19
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Load fonts
try:
    font_regular = pygame.freetype.Font("fonts/Roboto-Light.ttf", size=TEXT_SIZE)
    font_bold = pygame.freetype.Font("fonts/Roboto-Medium.ttf", size=TEXT_SIZE)
except:
    print("Could not load custom font. Falling back to default font.")
    font_regular = pygame.freetype.SysFont("Arial", size=TEXT_SIZE)
    font_bold = font_regular

COLORS = {
    "GREEN": (46, 204, 113),
    "PROGRESS_BAR": (11, 77, 70),
    "RED": (231, 76, 60),
    "WHITE": (236, 240, 241),
    "BLUE": (52, 152, 219),
    "ORANGE": (230, 126, 34),
    "YELLOW": (241, 196, 15),
    "BLACK": (7, 54, 66),  # Solarized base02
    "BACKGROUND": (39, 40, 35),  # Solarized base03
    "TEXT": (131, 148, 150),  # Solarized base0
    "BUBBLE_BG": (30, 31, 28),  # Solarized base02
    "BUBBLE_BORDER": (34, 35, 31),  # Solarized base01
}

BLACK = COLORS["BLACK"]
WHITE = COLORS["WHITE"]
BACKGROUND = COLORS["BACKGROUND"]
TEXT_COLOR = COLORS["TEXT"]
BUBBLE_BG = COLORS["BUBBLE_BG"]
BUBBLE_BORDER = COLORS["BUBBLE_BORDER"]

CUBE_SIZE = 55
CUBE_GAP = 3
FACE_GAP = 20
CUBE_OFFSET_X = 70
CUBE_OFFSET_Y = 100

def get_face_offset(face_index):
    offsets = [
        (1, 1),  # Front
        (2, 1),  # Right
        (1, 0),  # Top
        (0, 1),  # Left
        (3, 1),  # Back
        (1, 2),  # Bottom
    ]
    x, y = offsets[face_index]
    return (x * (3 * CUBE_SIZE + FACE_GAP) + CUBE_OFFSET_X, 
            y * (3 * CUBE_SIZE + FACE_GAP) + CUBE_OFFSET_Y)

def draw_rounded_rect(surface, color, rect, radius=10):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def lerp(a, b, t):
    return a + (b - a) * t

def rotate_cubelet(cubelet, rotation, target_rotation, progress):
    current = np.array(rotation)
    target = np.array(target_rotation)
    interpolated = lerp(current, target, progress)
    return tuple(map(tuple, interpolated))

def draw_cube_animated(cube, next_cube, progress):
    face_normals = [
        ( 1,  0,  0),  # Front
        ( 0,  1,  0),  # Right
        ( 0,  0,  1),  # Top
        ( 0, -1,  0),  # Left
        (-1,  0,  0),  # Back
        ( 0,  0, -1),  # Bottom
    ]
    
    for face_index, face_normal in enumerate(face_normals):
        face_offset_x, face_offset_y = get_face_offset(face_index)
        for (cubelet, rotation), (_, target_rotation) in zip(cube, next_cube):
            interpolated_rotation = rotate_cubelet(cubelet, rotation, target_rotation, progress)
            pos = tuple(np.matmul(interpolated_rotation, cubelet))
            if np.dot(pos, face_normal) == 1:
                if face_normal[0] != 0:  # Front or Back face
                    x, y = pos[1] + 1, -pos[2] + 1
                elif face_normal[1] != 0:  # Left or Right face
                    x, y = -pos[0] + 1, -pos[2] + 1
                else:  # Top or Bottom face
                    x, y = pos[0] + 1, pos[1] + 1

                draw_x = face_offset_x + x * (CUBE_SIZE + CUBE_GAP)
                draw_y = face_offset_y + y * (CUBE_SIZE + CUBE_GAP)
                
                color_normal = tuple(np.round(np.dot(np.array(interpolated_rotation).T, face_normal)).astype(int))
                
                # Handle the edge case where color_normal is (0, 0, 0)
                if all(v == 0 for v in color_normal):
                    # Use the original color (non-interpolated)
                    original_color_normal = tuple(np.round(np.dot(np.array(rotation).T, face_normal)).astype(int))
                    color = COLORS[color_names[original_color_normal]]
                else:
                    color = COLORS[color_names[color_normal]]

                draw_rounded_rect(screen, color, (draw_x, draw_y, CUBE_SIZE, CUBE_SIZE), 5)
                pygame.draw.rect(screen, BUBBLE_BORDER, (draw_x, draw_y, CUBE_SIZE, CUBE_SIZE), 1, border_radius=5)

def draw_cube_static(cube): return draw_cube_animated(cube, cube, 1)

MAX_TEXT_HEIGHT = font_bold.get_sized_height(TEXT_SIZE)

def draw_text_bubble(text, x, y, width, progress=None, bold_part=None):
    padding_x = 10
    rect_height = MAX_TEXT_HEIGHT + 20  # Fixed height based on max possible text height plus some padding

    # Pre-render both bold and regular parts to ensure consistent spacing
    bold_surface, _ = font_bold.render(bold_part or "", TEXT_COLOR)
    regular_surface, _ = font_regular.render(text[len(bold_part or ""):], TEXT_COLOR)
    
    text_width = bold_surface.get_width() + regular_surface.get_width()
    text_surface = pygame.Surface((text_width, MAX_TEXT_HEIGHT), pygame.SRCALPHA)
    text_surface.blit(bold_surface, (0, 0))
    text_surface.blit(regular_surface, (bold_surface.get_width(), 0))

    # Optional: Add a subtle glow effect
    glow_surface = pygame.Surface((width + 4, rect_height + 4), pygame.SRCALPHA)
    pygame.draw.rect(glow_surface, (*BUBBLE_BORDER, 100), (0, 0, width + 4, rect_height + 4))
    screen.blit(glow_surface, (x - 2, y - 2))

    # Draw background
    pygame.draw.rect(screen, BUBBLE_BG, (x, y, width, rect_height))
    
    # Draw progress bar if provided
    if progress is not None:
        progress_width = int(width * progress)
        progress_color = COLORS["PROGRESS_BAR"]
        pygame.draw.rect(screen, progress_color, (x, y, progress_width, rect_height))
    
    # Draw border
    pygame.draw.rect(screen, BUBBLE_BORDER, (x, y, width, rect_height), 1)
    
    # Draw text at a fixed position, centered vertically
    text_y = y + (rect_height - MAX_TEXT_HEIGHT) // 2 + 2
    screen.blit(text_surface, (x + padding_x, text_y))

    return rect_height

def create_button(text, x, y, width, height, color, text_color):
    button_surface = pygame.Surface((width, height))
    button_surface.fill(color)
    text_surface, _ = font_bold.render(text, text_color)
    text_rect = text_surface.get_rect(center=(width/2, height/2))
    button_surface.blit(text_surface, text_rect)
    button_rect = pygame.Rect(x, y, width, height)
    return (button_surface, button_rect)

def draw_move_info(move_index, solution, current_move):
    x, y, width = 10, HEIGHT - 50, WIDTH - 20

    if not solution:
        return draw_text_bubble("Press \"Solve\" to compute solution.", x=x, y=y, width=width, progress=0)
    
    num_moves = len(solution)
    move_text = f"Move: {move_index}/{num_moves}"
    if current_move:
        move_text += f" - {describe_move(current_move)}"
    else:
        move_text += " - Initial state"
    
    progress = move_index / num_moves if move_index > 0 else 0
    return draw_text_bubble(move_text, x=x, y=y, width=width, progress=progress, bold_part="Move:")

def draw_instructions(y):
    instructions = [
        "Right Arrow: step forward",
        "Left Arrow: step backward"
    ]
    for i, instruction in enumerate(instructions):
        draw_text_bubble(instruction, x = WIDTH - 275, y = y - (len(instructions) - i) * 45, width=265, bold_part=instruction.split(':')[0] + ':')

def report_solve_progress(cube):
    x, y, width = 10, HEIGHT - 50, WIDTH - 20
    num_cubelets_solved = sum(is_cubelet_solved(c, r) for c,r in cube)
    text = f"Solving: cubelet {num_cubelets_solved + 1} of {NUM_CUBELETS}"
    progress = num_cubelets_solved / NUM_CUBELETS
    draw_text_bubble(text, x=x, y=y, width=width, progress=progress, bold_part="Solving:")
    draw_cube_static(cube)
    pygame.display.flip()

def main():
    cube = shuffle(solved_cube, iterations=20)
    original_cube = cube
    solution = None
    
    clock = pygame.time.Clock()
    move_index = 0
    running = True
    current_move = None
    animation_progress = 0
    base_animation_speed = 0.06
    animation_speed = base_animation_speed
    next_cube = None
    speed_up_factor = 1
    key_hold_time = 0

    # Create buttons
    button_width, button_height = 150, 40
    button_y = 10
    button_spacing = (WIDTH - 3 * button_width) / 4
    scan_button = create_button("Scan my cube", button_spacing, button_y, button_width, button_height, COLORS["BLUE"], WHITE)
    shuffle_button = create_button("Shuffle", 2 * button_spacing + button_width, button_y, button_width, button_height, COLORS["ORANGE"], WHITE)
    solve_button = create_button("Solve", 3 * button_spacing + 2 * button_width, button_y, button_width, button_height, COLORS["GREEN"], WHITE)

    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    if scan_button[1].collidepoint(event.pos):
                        print("Scan my cube button clicked (no-op for now)")
                    elif shuffle_button[1].collidepoint(event.pos):
                        cube = shuffle(solved_cube, iterations=999, seed=None)
                        original_cube = cube
                        solution = None
                        move_index = 0
                        current_move = None
                        next_cube = None
                        animation_progress = 0
                    elif solve_button[1].collidepoint(event.pos):
                        solution = solve(cube, report_solve_progress)
                        move_index = 0
                        current_move = None
                        next_cube = None
                        animation_progress = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT] or keys[pygame.K_LEFT]:
            key_hold_time += dt
            speed_up_factor = 15 if key_hold_time > .6 else 1
        else:
            key_hold_time = 0
            speed_up_factor = 1

        if solution and next_cube is None:
            if keys[pygame.K_RIGHT] and move_index < len(solution):
                current_move = solution[move_index]
                next_cube = apply_move_to_cube(current_move, cube)
                animation_progress = 0
            elif keys[pygame.K_LEFT] and move_index > 0:
                move_index -= 1
                current_move = solution[move_index]
                inverse_move = (current_move[0], -current_move[1])
                next_cube = apply_move_to_cube(inverse_move, cube)
                animation_progress = 0
                current_move = inverse_move

        screen.fill(BACKGROUND)
        
        # Draw cube
        if next_cube:
            draw_cube_animated(cube, next_cube, animation_progress)
            animation_speed = base_animation_speed * speed_up_factor
            animation_progress += animation_speed
            if animation_progress >= 1:
                cube = next_cube
                next_cube = None
                if current_move == solution[move_index]:
                    move_index += 1
                animation_progress = 0
        else:
            draw_cube_static(cube)

        # Draw buttons
        screen.blit(scan_button[0], scan_button[1])
        screen.blit(shuffle_button[0], shuffle_button[1])
        screen.blit(solve_button[0], solve_button[1])

        # Draw instructions and move info
        if solution: draw_instructions(HEIGHT - 60)
        draw_move_info(move_index, solution, current_move)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
