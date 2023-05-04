# A rubix cube is a discrete object in 3-dimensional space.
# It sits in [-1, 1]^3, centered at the origin (0, 0, 0).
# Each *cubelet* has coordinates (x, y, z) in {-1, 0, 1}^3.
# Each *move* is a 90 degree hyperplane rotation, with the hyperplane given
# by a standard unit vector or its oposite.

import numpy as np
from collections import defaultdict


def norm1(vec): return sum(abs(x) for x in vec)
def projx(vec): return (vec[0], 0, 0)
def projy(vec): return (0, vec[1], 0)
def projz(vec): return (0, 0, vec[2])
def faces(cubelet):
    return [p(cubelet) for p in (projx, projy, projz) if p(cubelet) != (0, 0, 0)]

# Each coordinate x, y, z ranges from -1 to 1.
crange = [-1, 0, 1]

# Each cubelet is uniquely idenified by a vector.
# Visible cubelets are those with nonzero 1-norm.
# More generally, a cubelet with 1-norm n has n colored squares.
vectors = [(x, y, z) for x in crange for y in crange for z in crange]

# Unit vectors correspond to center cubelets.
# Since we restrict ourselves to moves that leave all center cubelets in place,
# the center cubelets are fixed points and correspond 1-to-1 to the colors.
unit_vectors = [v for v in vectors if norm1(v) == 1]

# A move is a clockwise or counterclockwise 90 degree rotation of the
# slice pointed at by a unit vector.
moves = [(v, direction) for v in unit_vectors for direction in [-1, 1]]

solved_cube = {cubelet: {f: f for f in faces(cubelet)} for cubelet in vectors}

color_names = {
    (+1, 0, 0): "GREEN",
    (0, +1, 0): "RED",
    (0, 0, +1): "WHITE",
    (-1, 0, 0): "BLUE",
    (0, -1, 0): "ORANGE",
    (0, 0, -1): "YELLOW"
}

def cubelet_type(cubelet):
    if norm1(cubelet) == 0:
        return "hidden"
    if norm1(cubelet) == 1:
        return "center"
    if norm1(cubelet) == 2:
        return "edge"
    if norm1(cubelet) == 3:
        return "corner"
    assert False

def position(cubelet):
    [x, y, z] = cubelet
    descriptions = []
    descriptions += ["top"] if z == 1 else ["bottom"] if z == -1 else []
    descriptions += ["right"] if y == 1 else ["left"] if y == -1 else []
    descriptions += ["front"] if x == 1 else ["back"] if x == -1 else []
    return "-".join(descriptions)

def cubelet_description(cubelet):
    return position(cubelet) + " " + cubelet_type(cubelet)

def cube_description(cube):
    lines = []
    for cubelet, faces in cube.items():
        if cubelet == (0, 0, 0): continue
        line = cubelet_description(cubelet) + ": "
        line += ", ".join(position(face) + ": " + color_names[color]
                          for face, color in faces.items())
        lines.append(line)
    return "\n".join(sorted(lines))

print(cube_description(solved_cube))

def move_description(move):
    [v, direction] = move
    return "90 degree %s rotation of %s slice" % (
        "clockwise" if direction == 1 else "counterclockwise",
        position(v),
    )

def rotation_matrix(move):
    [v, direction] = move
    fixed_dim = 0 if v[0] else 1 if v[1] else 2
    assert v[fixed_dim] != 0
    M = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            if i == fixed_dim or j == fixed_dim:
                M[i][j] = 1 if i == j else 0
            elif i == j:
                M[i][j] = 0
            else:
                M[i][j] = (1 if i > j else -1) * direction * v[fixed_dim]
    return M

def apply_move_to_vector(move, vector):
    return tuple(rotation_matrix(move) @ vector)

def apply_move_to_cubelet(move, cubelet):
    [v, direction] = move
    if np.dot(cubelet, v) > 0: return apply_move_to_vector(move, cubelet)
    return cubelet

def apply_move_to_cube(move, cube):
    print("\n\n== " + move_description(move) + " " + 20 * "=")
    new_cube = dict()
    for cubelet, faces in cube.items():
        new_cubelet = apply_move_to_cubelet(move, cubelet)
        print("before: %s -> after: %s" %
              (cubelet_description(cubelet), cubelet_description(new_cubelet)))
        if new_cubelet == cubelet:
            new_cube[new_cubelet] = faces
            continue
        
        new_faces = dict()
        for face, color in faces.items():
            new_face = apply_move_to_vector(move, face)
            print(" - before: %s -> after: %s" % (position(face), position(new_face)))
            assert new_face not in new_faces
            new_faces[new_face] = color
        assert new_cubelet not in new_cube
        new_cube[new_cubelet] = new_faces
        # new_cube[new_cubelet] = {apply_move_to_vector(move, face): color
        #                          for face, color in faces.items()}
    return new_cube


# Some simple unit tests.
for move in moves:
    cubes = [solved_cube]
    for _ in range(3):
        cubes.append(apply_move_to_cube(move, cubes[-1]))
    assert all(cubes.count(cube) == 1 for cube in cubes)
    assert apply_move_to_cube(move, cubes[-1]) == cubes[0]

# for move in moves:
#     print("\n----------------------------------\n")
#     cube = apply_move_to_cube(move, solved_cube)
#     print(cube_description(cube))
#     break
