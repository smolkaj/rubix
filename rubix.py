""" A minimalstic Rubik's cube solver.

We model a rubix cube as a discrete object in 3-dimensional Euclidean space,
centered at the origin (0, 0, 0).
Each *cubelet* has coordinates (x, y, z) in {-1, 0, 1}^3.
Each *move* is a 90 degree hyperplane rotation, with the hyperplane given
by a standard unit vector or its opposite.
"""

import numpy as np
import heapq
import random
import functools
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

# Whether or not to use randomization during search.
RANDOMIZE_SEARCH = True

# Remember start up time and move count for stats.
STARTUP_TIME, TOTAL_MOVES = datetime.now(), 0

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
cubelets = tuple(c for c in vectors if any(c))
solved_cube = tuple((c, tupled(np.identity(3))) for c in cubelets)
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
    assert len(set(cubes)) == 4 and apply_move_to_cube(move, cubes[-1]) == cubes[0]

@dataclass(order=True, frozen=True)
class PrioritizedItem:
  item: Any = field(compare=False)
  priority: int

def astar(start, is_goal, apply_move, heuristic = lambda _: 0,
          get_moves = lambda _: moves, random_weight=0, max_moves=100_000):
  global TOTAL_MOVES
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
        # Never immediately undo the move just taken.
        if last_move:
          if move == inverse_move(last_move): continue
          # Opposite face moves commute; prune duplicate branches by enforcing canonical order.
          if last_move[0] > move[0] and (-last_move[0][0], -last_move[0][1], -last_move[0][2]) == move[0]: continue
        dst, cost = apply_move(move, src), cost_so_far[src] + 1
        moves_simulated += 1
        TOTAL_MOVES += 1
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

@functools.cache
def min_moves_to_solved(cubelet, rotation):
  _, path = astar(rotation, lambda r: is_cubelet_solved(cubelet, r),
                  lambda m, r: tupled(rotation_matrix(m) @ r))
  return len(path)

@functools.cache
def min_moves_to_position(cubelet, rotation):
  _, path = astar(rotation, lambda r: position(cubelet, r) == cubelet,
                  lambda m, r: tupled(rotation_matrix(m) @ r))
  return len(path)

# The 24 rotational orientations of a cubelet in SO(3).
identity_3 = tupled(np.identity(3))
rotations = [identity_3]
for r in rotations:
  for m in moves:
    mr = tupled(rotation_matrix(m) @ r)
    if mr not in rotations: rotations.append(mr)
rot_to_id = {r: i for i, r in enumerate(rotations)}

# Fast state transitions compiled directly from linear algebra hyperplane tests.
transitions = tuple(
  tuple(
    tuple(
      rot_to_id[tupled(rotation_matrix(m) @ r)] if np.dot(m[0], position(c, r)) > 0 else r_idx
      for r_idx, r in enumerate(rotations)
    ) for c in cubelets
  ) for m in moves
)
move_to_id = {m: i for i, m in enumerate(moves)}

def to_cube(state): return tuple((cubelets[i], rotations[state[i]]) for i in range(NUM_CUBELETS))
def from_cube(cube):
  d = dict(cube)
  return tuple(rot_to_id[d[c]] for c in cubelets)

def apply_move_fast(m, s):
  t = transitions[move_to_id[m]]
  return tuple(t[i][s[i]] for i in range(NUM_CUBELETS))

def has_yellow_bottom(c, r): return c[2] == -1 and position((0, 0, -1), r) == (0, 0, -1)
def is_in_right_place(c, r): return position(c, r) == c

is_solved_tab = tuple(tuple(is_cubelet_solved(c, r) for r in rotations) for c in cubelets)
dist_solved_tab = tuple(tuple(min_moves_to_solved(c, r) for r in rotations) for c in cubelets)
dist_pos_tab = tuple(tuple(min_moves_to_position(c, r) for r in rotations) for c in cubelets)
has_yellow_tab = tuple(tuple(has_yellow_bottom(c, r) for r in rotations) for c in cubelets)
is_in_place_tab = tuple(tuple(is_in_right_place(c, r) for r in rotations) for c in cubelets)

top_edges = tuple(i for i, c in enumerate(cubelets) if c[2] == 1 and norm1(c) == 2)
top_cubelets = tuple(i for i, c in enumerate(cubelets) if c[2] == 1)
top_or_mid = tuple(i for i, c in enumerate(cubelets) if c[2] >= 0)
bot_edges = tuple(i for i, c in enumerate(cubelets) if c[2] == -1 and norm1(c) == 2)
bot_corners = tuple(i for i, c in enumerate(cubelets) if c[2] == -1 and norm1(c) == 3)
bot_edge_heur_idx = tuple(i for i, c in enumerate(cubelets) if not (c[2] == -1 and norm1(c) == 3))
mid_cubelets = tuple(i for i, c in enumerate(cubelets) if c[2] == 0)
bot_cubelets = tuple(i for i, c in enumerate(cubelets) if c[2] == -1)
bot_corner_dist = tuple(dist_pos_tab[i] if norm1(c) == 3 else dist_solved_tab[i] for i, c in enumerate(cubelets))

p = 0.5
def top_layer_heuristic(s): return (sum(dist_solved_tab[i][s[i]]**p for i in top_cubelets)**(1/p)) / 8
def middle_layer_heuristic(s): return (sum(dist_solved_tab[i][s[i]]**p for i in top_or_mid)**(1/p)) / 4
def bottom_layer_edge_heuristic(s): return (sum(dist_solved_tab[i][s[i]]**p for i in bot_edge_heur_idx)**(1/p)) / 3
def bottom_layer_corner_heuristic(s):
  d1 = sum(dist_solved_tab[i][s[i]]**p for i in top_cubelets)**(1/p)
  d2 = sum(dist_solved_tab[i][s[i]]**p for i in mid_cubelets)**(1/p)
  d3 = sum(bot_corner_dist[i][s[i]]**p for i in bot_cubelets)**(1/p)
  return d1/5 + d2/3 + d3/8

TOTAL_MOVES = 0

def solve_top_and_middle_layer(st, report_progress_callback):
  solution_moves = ()
  for num_solved in range(17):
    report_progress_callback(to_cube(st))
    print("solving cubelet #%d" % (num_solved + 1))
    tgt_e, tgt_t, tgt_tm = min(4, num_solved + 1), min(9, num_solved + 1), min(17, num_solved + 1)
    def is_goal(s): return (
      sum(is_solved_tab[i][s[i]] for i in top_edges) >= tgt_e and
      sum(is_solved_tab[i][s[i]] for i in top_cubelets) >= tgt_t and
      sum(is_solved_tab[i][s[i]] for i in top_or_mid) >= tgt_tm
    )
    h = top_layer_heuristic if num_solved < 9 else middle_layer_heuristic
    st, next_moves = astar(st, is_goal, apply_move_fast, h, random_weight=0.25)
    print("-> found solution with %d moves" % len(next_moves))
    solution_moves += next_moves
  return (st, solution_moves)

def solve_bottom_layer_edges(st, report_progress_callback):
  solution_moves = ()
  for i in range(8):
    report_progress_callback(to_cube(st))
    print("solving bottom cross #%d" % (i + 1))
    tgt_p, tgt_s = min(4, i + 1), min(4, i - 3)
    def is_goal(s): return (
      sum(is_solved_tab[idx][s[idx]] for idx in top_or_mid) == 17 and
      sum(has_yellow_tab[idx][s[idx]] for idx in bot_edges) >= tgt_p and
      sum(is_solved_tab[idx][s[idx]] for idx in bot_edges) >= tgt_s
    )
    st, next_moves = astar(st, is_goal, apply_move_fast, bottom_layer_edge_heuristic, random_weight=0.25)
    print("-> found solution with %d moves" % len(next_moves))
    solution_moves += next_moves
  return (st, solution_moves)

def solve_bottom_layer_corners(st, report_progress_callback):
  solution_moves = ()
  for i in range(4):
    report_progress_callback(to_cube(st))
    print("positioning bottom corners #%d" % (i + 1))
    tgt_c = min(4, i + 1)
    def is_goal(s): return (
      sum(is_solved_tab[idx][s[idx]] for idx in top_or_mid) == 17 and
      sum(is_solved_tab[idx][s[idx]] for idx in bot_edges) == 4 and
      sum(is_in_place_tab[idx][s[idx]] for idx in bot_corners) >= tgt_c
    )
    st, next_moves = astar(st, is_goal, apply_move_fast, bottom_layer_corner_heuristic, random_weight=0.3)
    print("-> found solution with %d moves" % len(next_moves))
    solution_moves += next_moves
  return (st, solution_moves)

def solve_endgame(st, report_progress_callback):
  solution = []
  bottom_m = ((0, 0, -1), 1)
  left_m = ((0, -1, 0), 1)
  top_m = ((0, 0, 1), 1)
  routine = 2 * (inverse_move(left_m), inverse_move(top_m), left_m, top_m)

  def apply(move):
    nonlocal st
    solution.append(move)
    st = apply_move_fast(move, st)

  def is_corner_oriented():
    c_idx = next(i for i in range(NUM_CUBELETS) if position(cubelets[i], rotations[st[i]]) == (1, -1, -1))
    rot_idx = st[c_idx]
    for _ in range(4):
      if is_solved_tab[c_idx][rot_idx]: return True
      rot_idx = transitions[move_to_id[bottom_m]][c_idx][rot_idx]
    return False

  for _ in range(4):
    report_progress_callback(to_cube(st))
    while not is_corner_oriented():
      for move in routine: apply(move)
    apply(bottom_m)
  while not all(is_solved_tab[i][st[i]] for i in range(NUM_CUBELETS)): apply(bottom_m)
  return (st, tuple(solution))

def solve(cube, report_progress_callback=lambda cube: None):
  st = from_cube(cube)
  st, solution1 = solve_top_and_middle_layer(st, report_progress_callback)
  print(50 * "-")
  st, solution2 = solve_bottom_layer_edges(st, report_progress_callback)
  print(50 * "-")
  st, solution3 = solve_bottom_layer_corners(st, report_progress_callback)
  print(50 * "-")
  st, solution4 = solve_endgame(st, report_progress_callback)
  solution = solution1 + solution2 + solution3 + solution4
  final_cube = to_cube(st)
  print("Solved cube in %d moves. Final cube:" % len(solution))
  print("is_cube_solved: ", is_cube_solved(final_cube))
  print("cube == solved_cube: ", final_cube == solved_cube)
  return solution

def print_stats():
  secs_elapsed = (datetime.now() - STARTUP_TIME).total_seconds()
  print("- time elapsed: %.1f sec" % secs_elapsed)
  print("- moves simulated: %d (%.0f moves/sec)" % (TOTAL_MOVES, TOTAL_MOVES / secs_elapsed))
  cache_info = min_moves_to_solved.cache_info()
  print("- min moves to solved calculations: ", cache_info.hits + cache_info.misses)

if __name__ == "__main__":
  import sys
  run_tests()
  arg = sys.argv[1] if len(sys.argv) > 1 else "42"
  if arg in ("-h", "--help"):
    print("Usage: python rubix.py [seed | --benchmark]")
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
