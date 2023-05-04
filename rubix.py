# A rubix cube is a discrete object in 3-dimensional space.
# It sits in [-1, 1]^3, centered at the origin (0, 0, 0).
# Each *cubelet* has coordinates (x, y, z) in {-1, 0, 1}^3.
# Each *move* is a 90 degree hyperplane rotation, with the hyperplane given
# by a standard unit vector or its oposite.

import numpy as np
import random
import functools
from collections import defaultdict
from truth.truth import AssertThat


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
assert all(v in color_names for v in unit_vectors)

def cubelet_type(cubelet):
    if norm1(cubelet) == 0: return "hidden"
    if norm1(cubelet) == 1: return "center"
    if norm1(cubelet) == 2: return "edge"
    if norm1(cubelet) == 3: return "corner"
    assert False

def position(vector):
    [x, y, z] = vector
    descriptions = []
    descriptions += ["top"] if z == 1 else ["bottom"] if z == -1 else []
    descriptions += ["right"] if y == 1 else ["left"] if y == -1 else []
    descriptions += ["front"] if x == 1 else ["back"] if x == -1 else []
    return "-".join(descriptions)

def describe_cubelet(cubelet):
    return position(cubelet) + " " + cubelet_type(cubelet)

def describe_cube(cube):
    lines = []
    for cubelet, faces in cube.items():
        if cubelet == (0, 0, 0): continue
        line = describe_cubelet(cubelet) + ": "
        line += ", ".join(position(face) + ": " + color_names[color]
                          for face, color in faces.items())
        lines.append(line)
    return "\n".join(sorted(lines))

# print(describe_cube(solved_cube))

def describe_move(move):
    [v, direction] = move
    return "90 degree %s rotation of %s slice" % (
        "clockwise" if direction == 1 else "counterclockwise",
        position(v),
    )

@functools.cache
def rotation_matrix(move):
    [v, direction] = move
    # The rotational axis is the dimension into which `v` is pointing.
    fixed_dim = [i for i in range(3) if v[i]][0]
    # The rotation takes place in the other two dimensions.
    [r1, r2] = [i for i in range(3) if i != fixed_dim]

    M = np.zeros([3, 3])
    M[fixed_dim, fixed_dim] = 1
    # The 90 degree rotation matrix [[0, 1], [-1,0]], adjusted by direction.
    # https://en.wikipedia.org/wiki/Rotation_matrix#Common_rotations
    M[r1, r2], M[r2, r1] = direction, -direction
    return M

@functools.cache
def apply_move_to_vector(move, vector):
    return tuple(rotation_matrix(move) @ vector)

@functools.cache
def apply_move_to_cubelet(move, cubelet):
    # print("applying '%s' to %s" % (describe_move(move()move), describe_cubelet(cubelet)))
    [v, direction] = move
    if np.dot(cubelet, v) > 0: return apply_move_to_vector(move, cubelet)
    return cubelet

def apply_move_to_cube(move, cube):
  return {
    (new_cubelet := apply_move_to_cubelet(move, cubelet)) :
    faces if new_cubelet == cubelet else {
      apply_move_to_vector(move, face): color for face, color in faces.items()
    }
    for cubelet, faces in cube.items()
  }

def run_tests():
  # Check that every move is a permutation with cyle length 4.
  for move in moves:
      cubes = [solved_cube]
      for _ in range(3):
          cubes.append(apply_move_to_cube(move, cubes[-1]))
      assert all(cubes.count(cube) == 1 for cube in cubes)
      assert apply_move_to_cube(move, cubes[-1]) == cubes[0]

  # Check that faces get rotated correctly around the x (front/back) dimension.
  move_by_name = { describe_move(move) : move for move in moves }
  face_by_position = { position(f) : f for f in unit_vectors }
  front_clockwise = move_by_name["90 degree clockwise rotation of front slice"]
  back_clockwise = move_by_name["90 degree clockwise rotation of back slice"]
  front_counterclockwise = move_by_name["90 degree counterclockwise rotation of front slice"]
  back_counterclockwise = move_by_name["90 degree counterclockwise rotation of back slice"]
  top, bottom = face_by_position["top"], face_by_position["bottom"]
  left, right = face_by_position["left"], face_by_position["right"]
  front, back = face_by_position["front"], face_by_position["back"]
  for clockwise in [front_clockwise, back_clockwise]:
    AssertThat(apply_move_to_vector(clockwise, top)).IsEqualTo(right)
    AssertThat(apply_move_to_vector(clockwise, right)).IsEqualTo(bottom)
    AssertThat(apply_move_to_vector(clockwise, bottom)).IsEqualTo(left)
    AssertThat(apply_move_to_vector(clockwise, left)).IsEqualTo(top)
    AssertThat(apply_move_to_vector(clockwise, front)).IsEqualTo(front)
    AssertThat(apply_move_to_vector(clockwise, back)).IsEqualTo(back)
  for counterclockwise in [front_counterclockwise, back_counterclockwise]:
    AssertThat(apply_move_to_vector(counterclockwise, top)).IsEqualTo(left)
    AssertThat(apply_move_to_vector(counterclockwise, right)).IsEqualTo(top)
    AssertThat(apply_move_to_vector(counterclockwise, bottom)).IsEqualTo(right)
    AssertThat(apply_move_to_vector(counterclockwise, left)).IsEqualTo(bottom)
    AssertThat(apply_move_to_vector(counterclockwise, front)).IsEqualTo(front)
    AssertThat(apply_move_to_vector(counterclockwise, back)).IsEqualTo(back)

def shuffle(cube, iterations=10_000, seed=42):
    random.seed(seed)
    new_cube = cube
    for _ in range(iterations):
        move = moves[random.randrange(len(moves))]
        new_cube = apply_move_to_cube(move, new_cube)
    return new_cube

run_tests()
print(describe_cube(shuffle(solved_cube)))
