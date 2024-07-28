import pygame
import numpy as np
from rubix import solved_cube, apply_move_to_cube, shuffle, solve, moves, color_names

pygame.init()

WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rubik's Cube Solver")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = {
    "GREEN": (0, 255, 0),
    "RED": (255, 0, 0),
    "WHITE": (255, 255, 255),
    "BLUE": (0, 0, 255),
    "ORANGE": (255, 165, 0),
    "YELLOW": (255, 255, 0),
}

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

def draw_cube(cube):
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

def main():
    cube = shuffle(solved_cube, iterations=20)
    solution = solve(cube)
    print("Solution:", solution)
    
    clock = pygame.time.Clock()
    move_index = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        draw_cube(cube)
        pygame.display.flip()

        if move_index < len(solution):
            move = solution[move_index]
            cube = apply_move_to_cube(move, cube)
            move_index += 1
            pygame.time.wait(500)  # Wait 500ms between moves

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
