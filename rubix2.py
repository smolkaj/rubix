# We model a rubix cube as a discrete object in 3-dimensional Euclidean space,
# centered at the origin (0, 0, 0).
# Each *cubelet* has coordinates (x, y, z) in {-1, 0, 1}^3.
# Each *move* is a 90 degree hyperplane rotation, with the hyperplane given
# by a standard unit vector or its oposite.

import numpy as np
import random
import functools
from collections import defaultdict, deque
from truth.truth import AssertThat

def norm1(v): return sum(abs(x) for x in v)

def tupled(np_mat):
   return tuple(tuple(int(x) for x in row) for row in np_mat)

crange = [-1, 0, 1]
vectors = [(x,y,z) for x in crange for y in crange for z in crange]
# Cube = Cubelet -> Rotation map. Encoded as a tuple so it can be hashed.
solved_cube = tuple((v, tupled(np.identity(3))) for v in vectors if any(v))
unit_vectors = [v for v in vectors if norm1(v) == 1]

# A move is a clockwise or counterclockwise 90 degree rotation of the
# slice pointed at by a unit vector.
moves = [(v, direction) for v in unit_vectors for direction in [-1, 1]]
color_names = {
  (+1, 0, 0): "GREEN",
  (0, +1, 0): "RED",
  (0, 0, +1): "WHITE",
  (-1, 0, 0): "BLUE",
  (0, -1, 0): "ORANGE",
  (0, 0, -1): "YELLOW"
}
assert all(v in color_names for v in unit_vectors)

def describe_cubelet_type(cubelet):
  if norm1(cubelet) == 0: return "hidden"
  if norm1(cubelet) == 1: return "center"
  if norm1(cubelet) == 2: return "edge"
  if norm1(cubelet) == 3: return "corner"
  assert False

def describe_position(vector):
  [x, y, z] = vector
  descriptions = []
  descriptions += ["top"] if z == 1 else ["bottom"] if z == -1 else []
  descriptions += ["right"] if y == 1 else ["left"] if y == -1 else []
  descriptions += ["front"] if x == 1 else ["back"] if x == -1 else []
  assert descriptions
  return "-".join(descriptions)

def describe_config(cubelet, rotation):
  colors = np.diag(cubelet)
  color_positions = rotation @ colors
  descriptions = (
     describe_position(pos) + ": " + color_names[tuple(color)]
     for color, pos in zip(colors.T, color_positions.T)
     if any(color)
  )
  return ", ".join(sorted(descriptions))

def describe_cube(cube):
  return "\n".join(sorted("%s %s: %s" % (
      describe_position(np.matmul(rotation, cubelet)),
      describe_cubelet_type(cubelet),
      describe_config(cubelet, rotation)
    )
    for cubelet, rotation in cube
  ))

def describe_move(move):
  [v, direction] = move
  return "90 degree %s rotation of %s slice" % (
      "clockwise" if direction == 1 else "counterclockwise",
      describe_position(v),
  )

@functools.cache
def rotation_matrix(move):
  [v, direction] = move
  # The rotational axis is the dimension into which `v` is pointing.
  fixed_dim = next(i for i in range(3) if v[i])
  # The rotation takes place in the other two dimensions.
  [r1, r2] = [i for i in range(3) if i != fixed_dim]

  M = np.zeros([3, 3])
  M[fixed_dim, fixed_dim] = 1
  # The 90 degree rotation matrix [[0, 1], [-1,0]], adjusted by direction.
  # https://en.wikipedia.org/wiki/Rotation_matrix#Common_rotations
  M[r1, r2], M[r2, r1] = direction, -direction
  return M

@functools.cache
def apply_move_to_cubelet_rotation(move, cubelet, rotation):
  [v, direction] = move
  move_applies = np.dot(v, np.matmul(rotation, cubelet)) > 0
  return tupled(rotation_matrix(move) @ rotation) if move_applies else rotation

def apply_move_to_cube(move, cube):
  return tuple(
    (cubelet, apply_move_to_cubelet_rotation(move, cubelet, rotation))
    for cubelet, rotation in cube
  )

# for move in moves[::-1]:
#   print("\n\n== " + describe_move(move) + "==============")
#   print(describe_cube(apply_move_to_cube(move, solved_cube)))

def shuffle(cube, iterations=100_000, seed=42):
  if seed: random.seed(seed)
  for _ in range(iterations):
    move = moves[random.randrange(len(moves))]
    cube = apply_move_to_cube(move, cube)
  return cube

def run_tests():
  # Check that every move is a permutation with cyle length 4.
  for move in moves:
    cubes = [solved_cube]
    for _ in range(3): cubes.append(apply_move_to_cube(move, cubes[-1]))
    assert len(set(cubes)) == 4
    assert apply_move_to_cube(move, cubes[-1]) == cubes[0]

run_tests()
# print(describe_cube(shuffle(solved_cube)))
# print("rotation_matrix: ", rotation_matrix.cache_info())
# print("apply_move_to_cubelet_rotation: ", apply_move_to_cubelet_rotation.cache_info())


def bfs(source, is_dst, get_moves, apply_move):
  state = (source, ())
  if is_dst(source): return state
  seen = set([source])
  frontier = deque([state])
  while frontier:
    [src, path] = frontier.popleft()
    # print("expanding path of length %d" % len(path))
    for i, move in enumerate(get_moves(src)):
      dst = apply_move(move, src)
      if dst in seen:
        # print(" -> extension %d already seen" % i)
        continue
      seen.add(dst)
      state = (dst, path + (move,))
      if is_dst(dst):
        # print(" -> extension %d reached the destination!" % i)
        return state
      # print(" -> extension %d added to frontier" % i)
      frontier.append(state)
  assert False

def solve_top_layer_cross(cube):
  solution = None
  def num_solved_cubelets(cube):
    return sum(1 for cubelet, rotation in cube
                if cubelet[2] == 1 and norm1(cubelet) == 2 
                and rotation == tupled(np.eye(3)))
  for solved_cubelets in range(4):
    print("solving top edge #%d" % (solved_cubelets + 1))
    def is_dst(cube): return num_solved_cubelets(cube) > solved_cubelets
    def get_moves(_): return range(len(moves))
    def apply_move(m, c): return apply_move_to_cube(moves[m], c)
    solution = [cube, path] = bfs(cube, is_dst, get_moves, apply_move)
    print("found solution with %d moves" % len(path))
  return solution

def solve_top_layer_complete(cube):
  solution = None
  def num_solved_cubelets(cube):
    return sum(1 for c, r in cube if c[2] == 1 and r == tupled(np.eye(3)))
  for solved_cubelets in range(9):
    print("solving top cubelet #%d" % (solved_cubelets + 1))
    def is_dst(cube): return num_solved_cubelets(cube) > solved_cubelets
    def get_moves(_): return range(len(moves))
    def apply_move(m, c): return apply_move_to_cube(moves[m], c)
    solution = [cube, path] = bfs(cube, is_dst, get_moves, apply_move)
    print("found solution with %d moves" % len(path))
  return solution

def solve(cube):
  [cube, path1] = solve_top_layer_cross(cube)
  [cube, path2] = solve_top_layer_complete(cube)
  path = path1 + path2
  print("Solved cube in %d moves. Final cube:" % len(path))
  print(describe_cube(cube))
  return path

random_cube = shuffle(solved_cube, seed=1)
solve(random_cube)
