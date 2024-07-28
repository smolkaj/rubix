import pygame
import pygame.font
import numpy as np
from rubix import solved_cube, apply_move_to_cube, shuffle, solve, moves, color_names, describe_move

pygame.init()
pygame.key.set_repeat(300, 100)  # Set key repeat: 300ms delay, 100ms interval

WIDTH, HEIGHT = 675, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rubik's Cube Solver")

COLORS = {
    "GREEN": (0, 255, 0),
    "RED": (255, 0, 0),
    "WHITE": (255, 255, 255),
    "BLUE": (0, 0, 255),
    "ORANGE": (255, 165, 0),
    "YELLOW": (255, 255, 0),
    "BLACK": (0, 0, 0),
}
BLACK = COLORS["BLACK"]
WHITE = COLORS["WHITE"]

CUBE_SIZE = 50
CUBE_GAP = 5
FACE_GAP = 20

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
    return (x * (3 * CUBE_SIZE + FACE_GAP), y * (3 * CUBE_SIZE + FACE_GAP))

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
        font = pygame.font.Font(None, 36)
        move_text = font.render(f"Move: {describe_move(current_move)}", True, BLACK)
        screen.blit(move_text, (10, HEIGHT - 150))
    
    for face_index, face_normal in enumerate(face_normals):
        face_offset_x, face_offset_y = get_face_offset(face_index)
        for cubelet, rotation in cube:
            pos = tuple(np.matmul(rotation, cubelet))
            if np.dot(pos, face_normal) == 1:
                # Calculate 2D coordinates based on the face
                if face_normal[0] != 0:  # Front or Back face
                    x, y = pos[1] + 1, -pos[2] + 1
                elif face_normal[1] != 0:  # Left or Right face
                    x, y = -pos[0] + 1, -pos[2] + 1
                else:  # Top or Bottom face
                    x, y = pos[0] + 1, pos[1] + 1

                draw_x = face_offset_x + x * (CUBE_SIZE + CUBE_GAP)
                draw_y = face_offset_y + y * (CUBE_SIZE + CUBE_GAP)
                
                # Calculate the color of this face
                color_normal = tuple(np.dot(np.array(rotation).T, face_normal))
                color = COLORS[color_names[color_normal]]
                
                pygame.draw.rect(screen, color, 
                    (draw_x, draw_y, CUBE_SIZE, CUBE_SIZE))
                pygame.draw.rect(screen, BLACK, 
                    (draw_x, draw_y, CUBE_SIZE, CUBE_SIZE), 2)

def draw_instructions():
    font = pygame.font.Font(None, 24)
    instructions = [
        "Right Arrow: Step forward",
        "Left Arrow: Step backward",
        "Space: Toggle auto-solve",
        "R: Reset to original state"
    ]
    for i, instruction in enumerate(instructions):
        text = font.render(instruction, True, BLACK)
        screen.blit(text, (10, HEIGHT - 100 + i * 25))

def main():
    cube = shuffle(solved_cube, iterations=20)
    original_cube = cube  # Store the original shuffled state
    solution = solve(cube)
    print("Solution:", solution)
    
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
                if event.key == pygame.K_RIGHT:  # Step forward
                    if move_index < len(solution):
                        current_move = solution[move_index]
                        cube = apply_move_to_cube(current_move, cube)
                        move_index += 1
                elif event.key == pygame.K_LEFT:  # Step backward
                    if move_index > 0:
                        move_index -= 1
                        current_move = solution[move_index]
                        inverse_move = (current_move[0], -current_move[1])
                        cube = apply_move_to_cube(inverse_move, cube)
                        current_move = inverse_move
                elif event.key == pygame.K_SPACE:  # Toggle auto-solve
                    auto_solve = not auto_solve
                elif event.key == pygame.K_r:  # Reset to original state
                    cube = original_cube
                    move_index = 0
                    current_move = None

        screen.fill(WHITE)
        draw_cube(cube, current_move)

        # Display current move number
        font = pygame.font.Font(None, 36)
        text = font.render(f"Move: {move_index}/{len(solution)}", True, BLACK)
        screen.blit(text, (10, 10))

        draw_instructions()
        pygame.display.flip()

        if auto_solve and move_index < len(solution):
            current_move = solution[move_index]
            cube = apply_move_to_cube(current_move, cube)
            move_index += 1
            pygame.time.wait(500)  # Wait 500ms between moves

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
