import pygame
import pygame.freetype
import numpy as np
from rubix import solved_cube, apply_move_to_cube, shuffle, solve, moves, color_names, describe_move

pygame.init()
pygame.key.set_repeat(300, 100)

WIDTH, HEIGHT = 850, 850
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rubik's Cube Solver")

# Load fonts
try:
    font_regular = pygame.freetype.Font("fonts/Roboto-Regular.ttf", 24)
    font_bold = pygame.freetype.Font("fonts/Roboto-Bold.ttf", 24)
except:
    print("Could not load custom font. Falling back to default font.")
    font_regular = pygame.freetype.SysFont("Arial", 24)
    font_bold = font_regular

COLORS = {
    "GREEN": (46, 204, 113),
    "RED": (231, 76, 60),
    "WHITE": (236, 240, 241),
    "BLUE": (52, 152, 219),
    "ORANGE": (230, 126, 34),
    "YELLOW": (241, 196, 15),
    "BLACK": (7, 54, 66),  # Solarized base02
    "BACKGROUND": (0, 43, 54),  # Solarized base03
    "TEXT": (131, 148, 150),  # Solarized base0
    "BUBBLE_BG": (7, 54, 66),  # Solarized base02
    "BUBBLE_BORDER": (88, 110, 117),  # Solarized base01
}

BLACK = COLORS["BLACK"]
WHITE = COLORS["WHITE"]
BACKGROUND = COLORS["BACKGROUND"]
TEXT_COLOR = COLORS["TEXT"]
BUBBLE_BG = COLORS["BUBBLE_BG"]
BUBBLE_BORDER = COLORS["BUBBLE_BORDER"]

CUBE_SIZE = 55
CUBE_GAP = 3
FACE_GAP = 25
CUBE_OFFSET_X = 70  # Increased offset for more space on the right
CUBE_OFFSET_Y = 70  # Increased offset for more space on the top

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

def draw_rounded_rect(surface, color, rect, radius=10, border_color=None):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color:
        pygame.draw.rect(surface, border_color, rect, width=1, border_radius=radius)

def draw_text(text, font, x, y, bold=False):
    font_to_use = font_bold if bold else font_regular
    text_surface, _ = font_to_use.render(text, TEXT_COLOR, size=20)
    screen.blit(text_surface, (x, y))

def draw_cube(cube, current_move=None):
    face_normals = [
        ( 1,  0,  0),  # Front
        ( 0,  1,  0),  # Right
        ( 0,  0,  1),  # Top
        ( 0, -1,  0),  # Left
        (-1,  0,  0),  # Back
        ( 0,  0, -1),  # Bottom
    ]

    if current_move:
        draw_text(f"Move: {describe_move(current_move)}", font_regular, BLACK, 20, HEIGHT - 180, bold=True)

    
    for face_index, face_normal in enumerate(face_normals):
        face_offset_x, face_offset_y = get_face_offset(face_index)
        for cubelet, rotation in cube:
            pos = tuple(np.matmul(rotation, cubelet))
            if np.dot(pos, face_normal) == 1:
                if face_normal[0] != 0:  # Front or Back face
                    x, y = pos[1] + 1, -pos[2] + 1
                elif face_normal[1] != 0:  # Left or Right face
                    x, y = -pos[0] + 1, -pos[2] + 1
                else:  # Top or Bottom face
                    x, y = pos[0] + 1, pos[1] + 1

                draw_x = face_offset_x + x * (CUBE_SIZE + CUBE_GAP)
                draw_y = face_offset_y + y * (CUBE_SIZE + CUBE_GAP)
                
                color_normal = tuple(np.dot(np.array(rotation).T, face_normal))
                color = COLORS[color_names[color_normal]]

                draw_rounded_rect(screen, color, (draw_x, draw_y, CUBE_SIZE, CUBE_SIZE), 5)
                pygame.draw.rect(screen, BUBBLE_BORDER, (draw_x, draw_y, CUBE_SIZE, CUBE_SIZE), 1, border_radius=5)

def draw_text_bubble(text, x, y, width=None, bold_part=None):
    if bold_part:
        bold_surface, _ = font_bold.render(bold_part, TEXT_COLOR, size=20)
        regular_surface, _ = font_regular.render(text[len(bold_part):], TEXT_COLOR, size=20)
        text_surface = pygame.Surface((bold_surface.get_width() + regular_surface.get_width(), max(bold_surface.get_height(), regular_surface.get_height())), pygame.SRCALPHA)
        text_surface.blit(bold_surface, (0, 0))
        text_surface.blit(regular_surface, (bold_surface.get_width(), 0))
    else:
        text_surface, _ = font_regular.render(text, TEXT_COLOR, size=20)
    
    text_width, text_height = text_surface.get_size()
    padding = 25
    rect_width = width if width else text_width + padding * 2
    rect_height = text_height + padding
    draw_rounded_rect(screen, BUBBLE_BG, (x, y, rect_width, rect_height), 20, BUBBLE_BORDER)
    screen.blit(text_surface, (x + padding, y + padding // 2))

def draw_instructions():
    instructions = [
        "Right Arrow: step forward",
        "Left Arrow: step backward",
        "Space: toggle auto-solve",
        "R: reset"
    ]
    for i, instruction in enumerate(instructions):
        draw_text_bubble(instruction, 10, HEIGHT - 180 + i * 45, width=270, bold_part=instruction.split(':')[0] + ':')

def draw_move_info(move_index, total_moves, current_move):
    move_text = f"Move: {move_index}/{total_moves}"
    if current_move:
        move_text += f" - {describe_move(current_move)}"
    
    draw_text_bubble(move_text, 10, 10, bold_part="Move:")

def main():
    cube = shuffle(solved_cube, iterations=20)
    original_cube = cube
    solution = solve(cube)
    
    clock = pygame.time.Clock()
    move_index = 0
    running = True
    auto_solve = False
    current_move = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    if move_index < len(solution):
                        current_move = solution[move_index]
                        cube = apply_move_to_cube(current_move, cube)
                        move_index += 1
                elif event.key == pygame.K_LEFT:
                    if move_index > 0:
                        move_index -= 1
                        current_move = solution[move_index]
                        inverse_move = (current_move[0], -current_move[1])
                        cube = apply_move_to_cube(inverse_move, cube)
                        current_move = inverse_move
                elif event.key == pygame.K_SPACE:
                    auto_solve = not auto_solve
                elif event.key == pygame.K_r:
                    cube = original_cube
                    move_index = 0
                    current_move = None

        screen.fill(BACKGROUND)
        draw_cube(cube)
        draw_move_info(move_index, len(solution), current_move)
        draw_instructions()
        pygame.display.flip()
        

        if auto_solve and move_index < len(solution):
            current_move = solution[move_index]
            cube = apply_move_to_cube(current_move, cube)
            move_index += 1
            pygame.time.wait(300)  # Slightly faster animation

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
