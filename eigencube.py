""" A minimalstic Rubik's cube solver.

We model a Rubik's cube as a discrete object in 3-dimensional Euclidean space,
centered at the origin (0, 0, 0).
Each *cubelet* has coordinates (x, y, z) in {-1, 0, 1}^3.
Each *move* is a 90 degree hyperplane rotation, with the hyperplane given
by a standard unit vector or its opposite.
"""

import numpy as np
import heapq
import random
import functools
import itertools
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

# Whether or not to use randomization during search.
RANDOMIZE_SEARCH = True

# Remember start up time for stats.
STARTUP_TIME = datetime.now()

# Redefine print to include timestamps.
_print = print
def print(*args, **kw):
  _print("[%s]" % (datetime.now().strftime('%H:%M:%S')), *args, **kw)

# Returns the 1-norm of a vector.
def norm1(v): return abs(v[0]) + abs(v[1]) + abs(v[2])

# Returns a 2-dimensional matrix as a tuple.
def tupled(np_mat): return tuple(tuple(int(x) for x in row) for row in np_mat)

# We are interested in x, y, z coordinates ranging over {-1, 0, 1}.
crange = [-1, 0, 1]
vectors = tuple((x,y,z) for x in crange for y in crange for z in crange)
unit_vectors = [v for v in vectors if norm1(v) == 1]

# A cube is encoded as a finite map from cubelets to 3-by-3 rotation matrices
# `r`, where each cubelet is encoded as the 3D vector `c` that indicates the
# position of the cubelet in the solved cube.
# The cubelets current position is given by the matrix-vector product `c*r`.
solved_cube = tuple((c, tupled(np.identity(3))) for c in vectors if any(c))
NUM_CUBELETS = len(solved_cube)

# A move is a clockwise or counterclockwise 90 degree rotation of the
# slice pointed at by a unit vector.
moves = [(v, direction) for v in unit_vectors for direction in [-1, 1]]

# Each color is encoded by the unit vector corresponding to the direction that
# faces of that color point to in a solved cube.
color_names = {
  (+1, 0, 0): "GREEN",   # front
  (0, +1, 0): "RED",     # right
  (0, 0, +1): "WHITE",   # top
  (-1, 0, 0): "BLUE",    # back
  (0, -1, 0): "ORANGE",  # left
  (0, 0, -1): "YELLOW",  # bottom
}
assert all(v in color_names for v in unit_vectors)

cubelet_types = ("hidden", "center", "edge", "corner")
def describe_cubelet_type(cubelet): return cubelet_types[norm1(cubelet)]

def describe_position(vector):
  x, y, z = vector
  descriptions = []
  descriptions += ["top"] if z == 1 else ["bottom"] if z == -1 else []
  descriptions += ["right"] if y == 1 else ["left"] if y == -1 else []
  descriptions += ["front"] if x == 1 else ["back"] if x == -1 else []
  assert descriptions
  return "-".join(descriptions)

def describe_config(cubelet, rotation):
  colors = np.diag(cubelet)
  color_positions = rotation @ colors
  return ", ".join(sorted(
     describe_position(pos) + ": " + color_names[tuple(color)]
     for color, pos in zip(colors.T, color_positions.T)
     if any(color)
  ))

@functools.cache
def position(cubelet, rotation): return tuple(np.matmul(rotation, cubelet))

def describe_cubelet(cubelet, rotation):
  return "%s %s: %s" % (
    describe_cubelet_type(cubelet),
    describe_position(position(cubelet, rotation)),
    describe_config(cubelet, rotation),
  )

def describe_cube(cube):
  return "\n".join(sorted(describe_cubelet(*c) for c in cube))

def describe_move(move):
  v, direction = move
  return "%s rotation of %s slice" % (
      "clockwise" if direction == 1 else "counterclockwise",
      describe_position(v),
  )

def inverse_move(move):
  v, direction = move
  return (v, -direction)

@functools.cache
def rotation_matrix(move):
  v, direction = move
  # The rotational axis is the dimension into which `v` is pointing.
  fixed_dim = next(i for i in range(3) if v[i])
  # The rotation takes place in the other two dimensions.
  r1, r2 = (i for i in range(3) if i != fixed_dim)

  M = np.zeros([3, 3])
  M[fixed_dim, fixed_dim] = 1
  # The 90 degree rotation matrix [[0, 1], [-1,0]], adjusted by direction.
  # https://en.wikipedia.org/wiki/Rotation_matrix#Common_rotations
  M[r1, r2], M[r2, r1] = direction, -direction
  return M

@functools.cache
def apply_move_to_cubelet_rotation(move, cubelet, rotation):
  v, direction = move
  move_applies = np.dot(v, position(cubelet, rotation)) > 0
  return tupled(rotation_matrix(move) @ rotation) if move_applies else rotation

def apply_move_to_cube(move, cube):
  return tuple(
    (cubelet, apply_move_to_cubelet_rotation(move, cubelet, rotation))
    for cubelet, rotation in cube
  )

def shuffle(cube, iterations=100_000, seed=42):
  if seed: random.seed(seed)
  for _ in range(iterations):
    move = moves[random.randrange(len(moves))]
    cube = apply_move_to_cube(move, cube)
  return cube

def run_tests():
  # Check that every move is a permutation with cycle length 4.
  for move in moves:
    cubes = [solved_cube]
    for _ in range(3): cubes.append(apply_move_to_cube(move, cubes[-1]))
    assert len(set(cubes)) == 4
    assert apply_move_to_cube(move, cubes[-1]) == cubes[0]


@dataclass(order=True, frozen=True)
class PrioritizedItem:
  item: Any = field(compare=False)
  priority: int

def astar(start, is_goal, apply_move, heuristic = lambda _: 0,
          get_moves = lambda _: moves, random_weight=0, max_moves=100_000,
          move_cost = lambda _: 1):
  if is_goal(start): return (start, ())
  budget = max_moves

  def reconstruct_solution(dst):
    path, current = [], dst
    while current in came_from:
      src, move = came_from[current]
      path.append(move)
      current = src
    return (dst, tuple(reversed(path)))

  while True:
    frontier = [PrioritizedItem(start, 0)]
    came_from, cost_so_far = {}, { start : 0 }
    moves_simulated = 0
    budget_exceeded = False

    while frontier:
      src = heapq.heappop(frontier).item
      last_move = came_from[src][1] if src in came_from else None
      for move in get_moves(src):
        # Never immediately undo the move just taken. (These rules relate single moves, not macros.)
        if last_move in moves and move in moves:
          if move == inverse_move(last_move): continue
          # Opposite face moves commute; prune duplicate branches by enforcing canonical order.
          if last_move[0] > move[0] and (-last_move[0][0], -last_move[0][1], -last_move[0][2]) == move[0]: continue
        dst, cost = apply_move(move, src), cost_so_far[src] + move_cost(move)
        moves_simulated += 1
        budget_exceeded = budget is not None and moves_simulated >= budget
        if dst in cost_so_far and cost_so_far[dst] <= cost:
          if budget_exceeded: break
          continue
        cost_so_far[dst], came_from[dst] = cost, (src, move)
        if is_goal(dst): return reconstruct_solution(dst)
        if budget_exceeded: break
        h_weight = random.gauss(1, random_weight) if RANDOMIZE_SEARCH else 1
        priority = cost + h_weight * heuristic(dst)
        heapq.heappush(frontier, PrioritizedItem(dst, priority))
      if budget_exceeded:
        break
    if not budget_exceeded or budget is None or random_weight == 0 or not RANDOMIZE_SEARCH:
      return None
    print("search budget of %d moves exceeded; restarting" % budget)
    budget = max(int(1.5 * budget), budget + 1)

@functools.cache
def is_cubelet_solved(cubelet, rotation):
  colors = np.diag(cubelet)
  color_positions = rotation @ colors
  return np.array_equal(colors, color_positions)

def is_cube_solved(cube): return all(is_cubelet_solved(c, r) for c, r in cube)

def num_solved_with_criterion(cube, criterion):
  return sum(is_cubelet_solved(c, r) for c,r in cube if criterion(c))

@functools.cache
def min_moves_to_solved(cubelet, rotation):
  def is_dst(r): return is_cubelet_solved(cubelet, r)
  def apply_move(m, r): return tupled(rotation_matrix(m) @ r)
  _, path = astar(rotation, is_dst, apply_move)
  return len(path)

@functools.cache
def min_moves_to_position(cubelet, rotation):
  def is_dst(r): return position(cubelet, r) == cubelet
  def apply_move(m, r): return tupled(rotation_matrix(m) @ r)
  _, path = astar(rotation, is_dst, apply_move)
  return len(path)

def top_layer_heuristic(cube):
  p, n = 0.5, 8
  d = sum(min_moves_to_solved(c, r)**p for c, r in cube if c[2] == 1) ** (1/p)
  return d/n

def middle_layer_heuristic(cube):
  p, n = 0.5, 4
  d = sum(min_moves_to_solved(c, r)**p for c, r in cube if c[2] >= 0) ** (1/p)
  return d/n

def bottom_layer_edge_heuristic(cube):
  p, n = 0.5, 3
  d = sum(min_moves_to_solved(c, r)**p for c, r in cube
          if not (c[2] == -1 and norm1(c) == 3)) ** (1/p)
  return d/n

def bottom_layer_corner_heuristic(cube):
  p, n1, n2, n3 = 0.5, 5, 3, 8
  d1 = sum(min_moves_to_solved(c, r)**p for c, r in cube if c[2] == 1) ** (1/p)
  d2 = sum(min_moves_to_solved(c, r)**p for c, r in cube if c[2] == 0) ** (1/p)
  d3 = sum((min_moves_to_position(c, r) if norm1(c) == 3 else min_moves_to_solved(c, r))**p
           for c, r in cube if c[2] == -1) ** (1/p)
  return d1/n1 + d2/n2 + d3/n3

def whole_cube_heuristic(cube):
  p, n = 0.5, 4
  return sum(min_moves_to_solved(c, r)**p for c, r in cube) ** (1/p) / n

def is_top_edge(cubelet): return cubelet[2] == 1 and norm1(cubelet) == 2
def is_top_cubelet(cubelet): return cubelet[2] == 1
def is_top_or_middle_cubelet(cubelet): return cubelet[2] >= 0

# Macros are move sequences that disturb only a few cubelets (commutators, as it
# turns out). Rather than hard-coding them, we learn them from the solver's own
# searches and offer them to later searches as single steps.
MAX_MACRO_DISTURBANCE = 12
macros = {}  # the macro's effect on the solved cube -> shortest such macro

# The 48 symmetries of the cube: signed permutation matrices.
symmetries = [np.diag(signs) @ np.identity(3)[list(permutation)]
              for permutation in itertools.permutations(range(3))
              for signs in itertools.product((-1, 1), repeat=3)]

def apply_moves_to_cube(sequence, cube):
  for move in sequence: cube = apply_move_to_cube(move, cube)
  return cube

def num_disturbed(cube): return sum(not is_cubelet_solved(c, r) for c, r in cube)

def symmetric_move(symmetry, move):
  # Conjugating a slice rotation by a symmetry yields the rotation of the image slice.
  rotation = symmetry @ rotation_matrix(move) @ symmetry.T
  return next(m for m in moves if np.array_equal(rotation_matrix(m), rotation) and
              m[0] == tuple(int(x) for x in symmetry @ move[0]))

def learn_macro(sequence):
  if not 0 < num_disturbed(apply_moves_to_cube(sequence, solved_cube)) <= MAX_MACRO_DISTURBANCE: return
  inverse = tuple(inverse_move(m) for m in reversed(sequence))
  for symmetry, variant in itertools.product(symmetries, (sequence, inverse)):
    macro = tuple(symmetric_move(symmetry, m) for m in variant)
    effect = apply_moves_to_cube(macro, solved_cube)
    if effect not in macros or len(macro) < len(macros[effect]): macros[effect] = macro

@functools.cache
def apply_macro_to_cubelet_rotation(macro, cubelet, rotation):
  for move in macro: rotation = apply_move_to_cubelet_rotation(move, cubelet, rotation)
  return rotation

# A search step is either a single move or a macro.
def moves_of_step(step): return (step,) if step in moves else step

def apply_step_to_cube(step, cube):
  if step in moves: return apply_move_to_cube(step, cube)
  return tuple((c, apply_macro_to_cubelet_rotation(step, c, r)) for c, r in cube)

def is_bottom_edge(cubelet): return cubelet[2] == -1 and norm1(cubelet) == 2
def is_bottom_corner(cubelet): return cubelet[2] == -1 and norm1(cubelet) == 3
def has_yellow_bottom(cubelet, rotation):
  return cubelet[2] == -1 and position((0, 0, -1), rotation) == (0, 0, -1)
def is_in_right_place(c, r): return position(c, r) == c
def num_bottom_edges_positioned(cube):
  return sum(is_bottom_edge(c) and has_yellow_bottom(c, r) for c, r in cube)
def num_bottom_corners_positioned(cube):
  return sum(is_bottom_corner(c) and is_in_right_place(c, r) for c, r in cube)

# Each phase solves a few more cubelets per step, keeping the ones already solved.
def top_layer_goal(cube, i): return (
  num_solved_with_criterion(cube, is_top_edge) >= min(4, i + 1) and
  num_solved_with_criterion(cube, is_top_cubelet) >= i + 1)
def middle_layer_goal(cube, i): return (
  num_solved_with_criterion(cube, is_top_cubelet) == 9 and
  num_solved_with_criterion(cube, is_top_or_middle_cubelet) >= 10 + i)
def bottom_cross_goal(cube, i): return (
  num_solved_with_criterion(cube, is_top_or_middle_cubelet) == 17 and
  num_bottom_edges_positioned(cube) >= min(4, i + 1) and
  num_solved_with_criterion(cube, is_bottom_edge) >= min(4, i - 3))
def bottom_corner_positions_goal(cube, i): return (
  num_solved_with_criterion(cube, is_top_or_middle_cubelet) == 17 and
  num_solved_with_criterion(cube, is_bottom_edge) == 4 and
  num_bottom_corners_positioned(cube) >= i + 1)
def bottom_corner_twists_goal(cube, i): return NUM_CUBELETS - num_disturbed(cube) >= 23 + i

phases = (  # (description, number of steps, goal, heuristic, random weight)
  ("solving top layer", 9, top_layer_goal, top_layer_heuristic, 0.25),
  ("solving middle layer", 8, middle_layer_goal, middle_layer_heuristic, 0.25),
  ("solving bottom cross", 8, bottom_cross_goal, bottom_layer_edge_heuristic, 0.25),
  ("positioning bottom corners", 4, bottom_corner_positions_goal, bottom_layer_corner_heuristic, 0.3),
  ("twisting bottom corners", 4, bottom_corner_twists_goal, whole_cube_heuristic, 0.3),
)

def solve(cube, report_progress_callback=lambda cube: None):
  solution = ()
  for description, num_steps, goal, heuristic, random_weight in phases:
    for i in range(num_steps):
      report_progress_callback(cube)
      print("%s #%d" % (description, i + 1))
      steps = moves + list(macros.values())
      cube, path = astar(cube, lambda cube: goal(cube, i), apply_step_to_cube, heuristic,
                         lambda _: steps, random_weight,
                         move_cost=lambda step: len(moves_of_step(step)))
      next_moves = sum((moves_of_step(step) for step in path), ())
      print("-> found solution with %d moves (%d macros known)" % (len(next_moves), len(macros)))
      learn_macro(next_moves)
      solution += next_moves
    print(50 * "-")
  print("Solved cube in %d moves." % len(solution))
  print("is_cube_solved: ", is_cube_solved(cube))
  return solution

def print_stats():
  secs_elapsed = (datetime.now() - STARTUP_TIME).total_seconds()
  cache_info = apply_move_to_cubelet_rotation.cache_info()
  moves = (cache_info.hits + cache_info.misses) / len(solved_cube)
  print("- time elapsed: %.1f sec" % secs_elapsed)
  print("- moves simulated: %d (%.0f moves/sec) " % (
    moves,
    moves / secs_elapsed
  ))
  cache_info = min_moves_to_solved.cache_info()
  print("- min moves to solved calculations: ", cache_info.hits + cache_info.misses)

if __name__ == "__main__":
  import sys
  run_tests()
  arg = sys.argv[1] if len(sys.argv) > 1 else "42"
  if arg in ("-h", "--help"):
    print("Usage: python eigencube.py [seed | --benchmark]")
  elif arg == "--benchmark":
    for seed in range(100):
      print("== SEED:", seed, "==========================================")
      solve(shuffle(solved_cube, iterations=100_000, seed=seed))
      print_stats()
  else:
    seed = int(arg)
    print("Solving scrambled cube (seed=%d)..." % seed)
    solve(shuffle(solved_cube, iterations=100_000, seed=seed))
    print_stats()
