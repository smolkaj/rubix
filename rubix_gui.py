import pygame
import pygame.freetype
import numpy as np
from rubix import solved_cube, apply_move_to_cube, shuffle, solve, moves, color_names, describe_move

pygame.init()
pygame.key.set_repeat(300, 100)

WIDTH, HEIGHT = 800, 800  # Slightly larger window for more whitespace
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

# Modern color palette
COLORS = {
    "GREEN": (76, 175, 80),
    "RED": (244, 67, 54),
    "WHITE": (255, 255, 255),
    "BLUE": (33, 150, 243),
    "ORANGE": (255, 152, 0),
    "YELLOW": (255, 235, 59),
    "BLACK": (33, 33, 33),
    "BACKGROUND": (245, 245, 245),
}

BLACK = COLORS["BLACK"]
WHITE = COLORS["WHITE"]
BACKGROUND = COLORS["BACKGROUND"]

CUBE_SIZE = 55
CUBE_GAP = 3
FACE_GAP = 25

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
    return (x * (3 * CUBE_SIZE + FACE_GAP) + 50, y * (3 * CUBE_SIZE + FACE_GAP) + 50)

def draw_rounded_rect(surface, color, rect, radius=10):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_text(text, font, color, x, y, bold=False):
    font_to_use = font_bold if bold else font_regular
    text_surface, _ = font_to_use.render(text, color, size=20)
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
        draw_text(f"Move: {describe_move(current_move)}", font_regular, BLACK, 20, HEIGHT - 150, bold=True)
    
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
                pygame.draw.rect(screen, BLACK, (draw_x, draw_y, CUBE_SIZE, CUBE_SIZE), 1, border_radius=5)

def draw_instructions():
    instructions = [
        "→: Step forward",
        "←: Step backward",
        "Space: Toggle auto-solve",
        "R: Reset"
    ]
    for i, instruction in enumerate(instructions):
        draw_rounded_rect(screen, WHITE, (10, HEIGHT - 140 + i * 35, 200, 30), 15)
        draw_text(instruction, font_regular, BLACK, 20, HEIGHT - 135 + i * 35)

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
        draw_cube(cube, current_move)

        draw_rounded_rect(screen, WHITE, (10, 10, 200, 40), 20)
        draw_text(f"Move: {move_index}/{len(solution)}", font_regular, BLACK, 20, 20, bold=True)

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
