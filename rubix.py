# We model a rubix cube as a discrete object in 3-dimensional Euclidean space,
# centered at the origin (0, 0, 0).
# Each *cubelet* has coordinates (x, y, z) in {-1, 0, 1}^3.
# Each *move* is a 90 degree hyperplane rotation, with the hyperplane given
# by a standard unit vector or its oposite.

import numpy as np
import random
import functools
from collections import deque
from truth.truth import AssertThat
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
import heapq
import signal
import math

RANDOMIZE_SEARCH = True

_print = print
def print(*args, **kw): 
  _print("[%s]" % (datetime.now().strftime('%H:%M:%S')), *args, **kw)
def norm1(v): return sum(abs(x) for x in v)
def tupled(np_mat): return tuple(tuple(int(x) for x in row) for row in np_mat)

startup_time = datetime.now()
crange = [-1, 0, 1]
vectors = tuple((x,y,z) for x in crange for y in crange for z in crange)
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
  (0, 0, -1): "YELLOW",
}
assert all(v in color_names for v in unit_vectors)

def describe_cubelet_type(cubelet):
  if norm1(cubelet) == 0: return "hidden"
  if norm1(cubelet) == 1: return "center"
  if norm1(cubelet) == 2: return "edge"
  if norm1(cubelet) == 3: return "corner"
  assert False

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

# for move iall_n moves[::-1]:
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


@dataclass(order=True, frozen=True)
class PrioritizedItem:
  item: Any = field(compare=False)
  priority: int

def astar(start, is_goal, get_moves, apply_move, heuristic = lambda _: 0, 
          random_weight=0):
  if is_goal(start): return (start, ())
  frontier = [PrioritizedItem(start, 0)]
  came_from = {}
  cost_so_far = { start : 0 }

  def reconstruct_solution(dst):
    path = []
    current = dst
    while current in came_from:
      src = came_from[current]
      move = next(m for m in get_moves(src) if apply_move(m, src) == current)
      path.append(move)
      current = src
    return (dst, tuple(reversed(path)))

  while frontier:
    src = heapq.heappop(frontier).item
    for move in get_moves(src):
      dst, cost = apply_move(move, src), cost_so_far[src] + 1
      if dst in cost_so_far and cost_so_far[dst] <= cost: continue
      cost_so_far[dst], came_from[dst] = cost, src
      if is_goal(dst): return reconstruct_solution(dst)
      h_weight = random.gauss(1, random_weight) if RANDOMIZE_SEARCH else 1
      priority = cost + h_weight * heuristic(dst)
      heapq.heappush(frontier, PrioritizedItem(dst, priority))
  return None

def idastar(start, is_goal, get_moves, apply_move, heuristic = lambda _: 0,
            random_weight=0):
  class FoundSolution(Exception):
    def __init__(self, solution): self.solution = solution

  def reconstruct_solution(path):
    moves = []
    for i in range(len(path) - 1):
      src, dst = path[i:i+2]
      move = next(m for m in get_moves(src) if apply_move(m, src) == dst)
      moves.append(move)
    return (path[-1], tuple(moves))
  
  def search(path, cost, bound):
    node = path[-1]
    h_weight = random.gauss(1, random_weight) if RANDOMIZE_SEARCH else 1
    estimate = cost + h_weight * heuristic(node)
    if is_goal(node): raise FoundSolution(reconstruct_solution(path))
    if estimate > bound: return estimate
    min_estimate = math.inf
    for move in get_moves(node):
      succ = apply_move(move, node)
      if succ in path: continue
      estimate = search(path + (succ,), cost+1, bound)
      min_estimate = min(min_estimate, estimate)
    return min_estimate

  bound = heuristic(start)
  while bound < math.inf:
    try: bound = search((start,), 0, bound)
    except FoundSolution as e:  return e.solution
  
  return None

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
  def is_dst(rotation): return is_cubelet_solved(cubelet, rotation)
  def get_moves(_): return moves
  def apply_move(m, x): return tupled(rotation_matrix(m) @ x)
  _, path = astar(rotation, is_dst, get_moves, apply_move)
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
  d3 = sum(min_moves_to_solved(c, r)**p for c, r in cube if c[2] == -1) ** (1/p)
  return d1/n1 + d2/n2 + d3/n3

def is_top_edge(cubelet): return cubelet[2] == 1 and norm1(cubelet) == 2
def is_top_cubelet(cubelet): return cubelet[2] == 1
def is_top_or_middle_cubelet(cubelet): return cubelet[2] >= 0

def with_restarts(timeout, f, *args, **kwargs):
  def raise_timeout(signum, frame): raise TimeoutError()
  signal.signal(signal.SIGALRM, raise_timeout)
  signal.alarm(timeout)
  while True:
    try:
      result = f(*args, **kwargs)
      signal.alarm(0)
      return result
    except TimeoutError:
      print("timed out after %d seconds; restarting" % timeout)
      timeout = min(2 * timeout, 300)
      signal.alarm(timeout)

def solve_top_and_middle_layer(cube):
  solution_moves = ()
  for num_solved in range(17):
    print("solving cubelet #%d" % (num_solved + 1))
    def is_goal(cube): return all([
      num_solved_with_criterion(cube, is_top_edge) >= min(4, num_solved + 1),
      num_solved_with_criterion(cube, is_top_cubelet) >= min(9, num_solved + 1),
      num_solved_with_criterion(cube, is_top_or_middle_cubelet) >= min(17, num_solved + 1),
    ])
    def get_moves(_): return moves
    heuristic = top_layer_heuristic if num_solved < 9 else middle_layer_heuristic
    cube, next_moves = astar(cube, is_goal, get_moves, apply_move_to_cube,
                              heuristic, random_weight=0.25)
    print("-> found solution with %d moves" % len(next_moves))
    solution_moves += next_moves
  return (cube, solution_moves)

def is_bottom_edge(cubelet): return cubelet[2] == -1 and norm1(cubelet) == 2
def is_bottom_corner(cubelet): return cubelet[2] == -1 and norm1(cubelet) == 3
def is_bottom_cubelet(cubelet): return cubelet[2] == -1
def has_orange_bottom(cubelet, rotation):
  return cubelet[2] == -1 and all((rotation @ np.array([0, 0, -1])) == [0, 0, -1])
def is_in_right_place(c, r):
  return position(c, r) == c
def num_bottom_edges_positioned(cube):
  return sum(is_bottom_edge(c) and has_orange_bottom(c, r) for c, r in cube)
def num_bottom_corners_positioned(cube):
  return sum(is_bottom_corner(c) and is_in_right_place(c, r) for c, r in cube)

def solve_bottom_layer_edges(cube):
  solution_moves = ()
  for i in range(8):
    print("solving bottom cross #%d" % (i + 1))
    def is_goal(cube): return all([
      num_solved_with_criterion(cube, is_top_or_middle_cubelet) == 17,
      num_bottom_edges_positioned(cube) >= min(4, i + 1),
      num_solved_with_criterion(cube, is_bottom_edge) >= min(4, i-3),
    ])
    def get_moves(_): return moves
    heuristic = bottom_layer_edge_heuristic
    try:
      cube, next_moves = astar(cube, is_goal, get_moves, apply_move_to_cube,
                                heuristic)
    except KeyboardInterrupt:
      return (cube, solution_moves)
    print("-> found solution with %d moves" % len(next_moves))
    solution_moves += next_moves
  return (cube, solution_moves)

def solve_bottom_layer_corners(cube):
  solution_moves = ()
  for i in range(4):
    print("positioning bottom corners #%d" % (i + 1))
    def is_goal(cube): return all([
      num_solved_with_criterion(cube, is_top_or_middle_cubelet) == 17,
      num_solved_with_criterion(cube, is_bottom_edge) == 4,
      num_bottom_corners_positioned(cube) >= min(4, i + 1),
    ])
    def get_moves(_): return moves
    heuristic = bottom_layer_corner_heuristic
    try:
      cube, next_moves = astar(cube, is_goal, get_moves, apply_move_to_cube,
                                heuristic, random_weight=0.3)
    except KeyboardInterrupt:
      return (cube, solution_moves)
    print("-> found solution with %d moves" % len(next_moves))
    solution_moves += next_moves
  return (cube, solution_moves)

def bottom_left_front_corner(cube):
  return next((c,r) for c,r in cube if position(c, r) == (1, -1, -1))  

def solve_endgame(cube):
  solution = []
  move_by_name = { describe_move(move) : move for move in moves }
  def apply_move(m, cunbe):
    move = move_by_name[m]
    solution.append(m)
    return apply_move_to_cube(move, cube)
  routine = 2 * [
    "counterclockwise rotation of left slice",
    "counterclockwise rotation of top slice",
    "clockwise rotation of left slice",
    "clockwise rotation of top slice",
  ]
  def is_bottom_left_front_corner_ok(cube):
    c, r = bottom_left_front_corner(cube)
    move = move_by_name["clockwise rotation of bottom slice"]
    for i in range(4):
      if is_cubelet_solved(c, r): return True
      r = apply_move_to_cubelet_rotation(move, c, r)
    return False
  
  for _ in range(4):
    while not is_bottom_left_front_corner_ok(cube):
      for move in routine: cube = apply_move(move, cube)
    cube = apply_move("clockwise rotation of bottom slice", cube)
  
  while not is_cube_solved(cube):
    cube = apply_move("clockwise rotation of bottom slice", cube)

  return (cube, tuple(solution))


def solve(cube):
  cube, solution1 = with_restarts(20, solve_top_and_middle_layer, cube)
  print(50 * "-")
  cube, solution2 = solve_bottom_layer_edges(cube)
  print(50 * "-")
  cube, solution3 = with_restarts(30, solve_bottom_layer_corners, cube)
  print(50 * "-")
  cube, solution4 = solve_endgame(cube)
  solution = solution1 + solution2 + solution3 + solution4
  print("Solved cube in %d moves. Final cube:" % len(solution))
  # print(describe_cube(cube))
  print("is_cube_solved: ", is_cube_solved(cube))
  print("cube == solved_cube: ", cube == solved_cube)
  return solution

def print_stats():
  secs_elapsed = (datetime.now() - startup_time).total_seconds()
  cache_info = apply_move_to_cubelet_rotation.cache_info()
  moves = (cache_info.hits + cache_info.misses) / len(solved_cube)
  print("- time elapsed: %.1f sec" % secs_elapsed)
  print("- moves simulated: %d (%.0f moves/sec) " % (
    moves,
    moves / secs_elapsed
  ))
  cache_info = min_moves_to_solved.cache_info()
  print("- min moves to solved calculations: ", cache_info.hits + cache_info.misses)

tough_seeds_for_top_layer = [17, 33]
tough_seeds_for_middle_layer = [0, 3, 4, 7, 8]
tough_seeds_for_bottom_layer = [6]

for seed in range(100):
  print("== SEED:", seed, "==========================================")
  random_cube = shuffle(solved_cube, iterations=100_000, seed=seed)
  solve(random_cube)
  print_stats()
