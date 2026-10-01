"""Rubix in Mojo: A Functional Pearl in Linear Algebra.

A Rubik's cube modeled as a discrete object in 3D Euclidean space centered at (0,0,0).
Each cubelet has coordinates (x, y, z) in {-1, 0, 1}^3.
Each move is a 90-degree hyperplane rotation.
"""

from std.time import perf_counter_ns
from std.math import sqrt

# ---------------------------------------------------------------------------
# 1. Discrete 3D Vector & Matrix Algebra
# ---------------------------------------------------------------------------

@fieldwise_init
struct Vec3(ImplicitlyCopyable, Movable, Writable, KeyElement):
    var x: Int; var y: Int; var z: Int

    def __eq__(self, o: Vec3) -> Bool: return self.x == o.x and self.y == o.y and self.z == o.z
    def __ne__(self, o: Vec3) -> Bool: return not (self == o)
    def __hash__(self) -> UInt: return UInt(self.x * 73856093 ^ self.y * 19349663 ^ self.z * 83492791)
    def norm1(self) -> Int: return abs(self.x) + abs(self.y) + abs(self.z)
    def dot(self, o: Vec3) -> Int: return self.x * o.x + self.y * o.y + self.z * o.z
    def write_to[W: Writer](self, mut writer: W): writer.write("(", self.x, ", ", self.y, ", ", self.z, ")")

@fieldwise_init
struct Mat3(ImplicitlyCopyable, Movable, Writable, KeyElement):
    var m00: Int; var m01: Int; var m02: Int
    var m10: Int; var m11: Int; var m12: Int
    var m20: Int; var m21: Int; var m22: Int

    @staticmethod
    def identity() -> Mat3: return Mat3(1, 0, 0, 0, 1, 0, 0, 0, 1)

    def __matmul__(self, v: Vec3) -> Vec3:
        return Vec3(self.m00 * v.x + self.m01 * v.y + self.m02 * v.z,
                    self.m10 * v.x + self.m11 * v.y + self.m12 * v.z,
                    self.m20 * v.x + self.m21 * v.y + self.m22 * v.z)

    def __matmul__(self, o: Mat3) -> Mat3:
        return Mat3(
            self.m00*o.m00 + self.m01*o.m10 + self.m02*o.m20,
            self.m00*o.m01 + self.m01*o.m11 + self.m02*o.m21,
            self.m00*o.m02 + self.m01*o.m12 + self.m02*o.m22,
            self.m10*o.m00 + self.m11*o.m10 + self.m12*o.m20,
            self.m10*o.m01 + self.m11*o.m11 + self.m12*o.m21,
            self.m10*o.m02 + self.m11*o.m12 + self.m12*o.m22,
            self.m20*o.m00 + self.m21*o.m10 + self.m22*o.m20,
            self.m20*o.m01 + self.m21*o.m11 + self.m22*o.m21,
            self.m20*o.m02 + self.m21*o.m12 + self.m22*o.m22)

    def __eq__(self, o: Mat3) -> Bool:
        return (self.m00 == o.m00 and self.m01 == o.m01 and self.m02 == o.m02 and
                self.m10 == o.m10 and self.m11 == o.m11 and self.m12 == o.m12 and
                self.m20 == o.m20 and self.m21 == o.m21 and self.m22 == o.m22)

    def __ne__(self, o: Mat3) -> Bool: return not (self == o)

    def __hash__(self) -> UInt:
        return UInt(self.m00 * 31 + self.m01 * 37 + self.m02 * 41 +
                    self.m10 * 43 + self.m11 * 47 + self.m12 * 53 +
                    self.m20 * 59 + self.m21 * 61 + self.m22 * 67)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[[", self.m00, ",", self.m01, ",", self.m02, "],[",
                     self.m10, ",", self.m11, ",", self.m12, "],[",
                     self.m20, ",", self.m21, ",", self.m22, "]]")

# ---------------------------------------------------------------------------
# 2. Domain Model: Cubelets and Moves
# ---------------------------------------------------------------------------

@fieldwise_init
struct Move(ImplicitlyCopyable, Movable, Writable, KeyElement):
    var v: Vec3
    var direction: Int

    def inverse(self) -> Move: return Move(self.v, -self.direction)

    def matrix(self) -> Mat3:
        if self.v.x != 0:
            return Mat3(1, 0, 0,  0, 0, self.direction,  0, -self.direction, 0)
        elif self.v.y != 0:
            return Mat3(0, 0, self.direction,  0, 1, 0,  -self.direction, 0, 0)
        else:
            return Mat3(0, self.direction, 0,  -self.direction, 0, 0,  0, 0, 1)

    def __eq__(self, o: Move) -> Bool: return self.v == o.v and self.direction == o.direction
    def __ne__(self, o: Move) -> Bool: return not (self == o)
    def __hash__(self) -> UInt: return self.v.__hash__() ^ UInt(self.direction * 101)
    def write_to[W: Writer](self, mut writer: W):
        writer.write("Move(", self.v, ", ", self.direction, ")")

def init_cubelets() -> List[Vec3]:
    var res = List[Vec3]()
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                var v = Vec3(x, y, z)
                if v.norm1() > 0: res.append(v)
    return res^

def init_moves() -> List[Move]:
    var res = List[Move]()
    var axes = List[Vec3]()
    axes.append(Vec3(1, 0, 0)); axes.append(Vec3(-1, 0, 0))
    axes.append(Vec3(0, 1, 0)); axes.append(Vec3(0, -1, 0))
    axes.append(Vec3(0, 0, 1)); axes.append(Vec3(0, 0, -1))
    for i in range(len(axes)):
        res.append(Move(axes[i], 1))
        res.append(Move(axes[i], -1))
    return res^

@fieldwise_init
struct Cube(Copyable, Movable, KeyElement):
    var rotations: List[Mat3]

    def __init__(out self):
        self.rotations = List[Mat3]()
        for _ in range(26): self.rotations.append(Mat3.identity())

    def __eq__(self, o: Cube) -> Bool:
        for i in range(26):
            if self.rotations[i] != o.rotations[i]: return False
        return True

    def __ne__(self, o: Cube) -> Bool: return not (self == o)

    def __hash__(self) -> UInt:
        var h: UInt = 0
        for i in range(26): h = (h * 31) ^ self.rotations[i].__hash__()
        return h

    def apply_move(self, m: Move, cubelets: List[Vec3]) -> Cube:
        var res = Cube()
        var M = m.matrix()
        for i in range(26):
            var c = cubelets[i]
            var pos = self.rotations[i] @ c
            res.rotations[i] = (M @ self.rotations[i]) if (m.v.dot(pos) > 0) else self.rotations[i]
        return res^

def is_cubelet_solved_single(c: Vec3, R: Mat3) -> Bool:
    if not ((R @ c) == c): return False
    if c.x != 0 and not ((R @ Vec3(c.x, 0, 0)) == Vec3(c.x, 0, 0)): return False
    if c.y != 0 and not ((R @ Vec3(0, c.y, 0)) == Vec3(0, c.y, 0)): return False
    if c.z != 0 and not ((R @ Vec3(0, 0, c.z)) == Vec3(0, 0, c.z)): return False
    return True

def is_cubelet_solved(cube: Cube, i: Int, cubelets: List[Vec3]) -> Bool:
    return is_cubelet_solved_single(cubelets[i], cube.rotations[i])

def is_cube_solved(cube: Cube, cubelets: List[Vec3]) -> Bool:
    for i in range(26):
        if not is_cubelet_solved(cube, i, cubelets): return False
    return True

# ---------------------------------------------------------------------------
# 3. Cayley Geodesic Heuristics & Search Structures
# ---------------------------------------------------------------------------

@fieldwise_init
struct DistanceTables:
    var dist_solved: List[Dict[Mat3, Int]]
    var dist_pos: List[Dict[Mat3, Int]]

def build_distance_tables(cubelets: List[Vec3], moves: List[Move]) raises -> DistanceTables:
    var orientations = List[Mat3]()
    var seen = Dict[Mat3, Int]()
    orientations.append(Mat3.identity())
    seen[Mat3.identity()] = 0
    var head = 0
    while head < len(orientations):
        var r = orientations[head]; head += 1
        for i in range(len(moves)):
            var r_next = moves[i].matrix() @ r
            if r_next not in seen:
                seen[r_next] = len(orientations); orientations.append(r_next)

    var dist_solved = List[Dict[Mat3, Int]]()
    var dist_pos = List[Dict[Mat3, Int]]()
    for i in range(26):
        var c = cubelets[i]
        var d_s = Dict[Mat3, Int]()
        var d_p = Dict[Mat3, Int]()

        var q_s = List[Mat3]()
        for j in range(len(orientations)):
            var r = orientations[j]
            if is_cubelet_solved_single(c, r): d_s[r] = 0; q_s.append(r)
        var qs_head = 0
        while qs_head < len(q_s):
            var r = q_s[qs_head]; qs_head += 1
            var d = d_s[r]
            for m in range(len(moves)):
                var r_prev = moves[m].matrix() @ r
                if r_prev not in d_s: d_s[r_prev] = d + 1; q_s.append(r_prev)
        dist_solved.append(d_s^)

        var q_p = List[Mat3]()
        for j in range(len(orientations)):
            var r = orientations[j]
            if (r @ c) == c: d_p[r] = 0; q_p.append(r)
        var qp_head = 0
        while qp_head < len(q_p):
            var r = q_p[qp_head]; qp_head += 1
            var d = d_p[r]
            for m in range(len(moves)):
                var r_prev = moves[m].matrix() @ r
                if r_prev not in d_p: d_p[r_prev] = d + 1; q_p.append(r_prev)
        dist_pos.append(d_p^)

    return DistanceTables(dist_solved^, dist_pos^)

@fieldwise_init
struct PQItem(Copyable, Movable):
    var priority: Float64
    var state_idx: Int

struct PriorityQueue:
    var items: List[PQItem]
    def __init__(out self): self.items = List[PQItem]()

    def push(mut self, item: PQItem):
        self.items.append(item.copy())
        var i = len(self.items) - 1
        while i > 0:
            var p = (i - 1) // 2
            if self.items[i].priority < self.items[p].priority:
                var tmp = self.items[i].copy()
                self.items[i] = self.items[p].copy()
                self.items[p] = tmp^
                i = p
            else: break

    def pop(mut self) -> PQItem:
        var top = self.items[0].copy()
        var last = self.items.pop()
        if len(self.items) > 0:
            self.items[0] = last^
            var i = 0
            var n = len(self.items)
            while True:
                var sm = i
                var l = 2 * i + 1; var r = 2 * i + 2
                if l < n and self.items[l].priority < self.items[sm].priority: sm = l
                if r < n and self.items[r].priority < self.items[sm].priority: sm = r
                if sm != i:
                    var tmp = self.items[i].copy()
                    self.items[i] = self.items[sm].copy()
                    self.items[sm] = tmp^
                    i = sm
                else: break
        return top^

struct LCG:
    var state: UInt64
    def __init__(out self, seed: UInt64): self.state = seed
    def next(mut self) -> UInt64:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return self.state
    def next_float(mut self) -> Float64:
        return Float64(self.next() & 0xFFFFFFFFFFFFF) / Float64(0x10000000000000)

def shuffle(cube: Cube, moves: List[Move], cubelets: List[Vec3], iterations: Int, seed: Int) -> Cube:
    var rng = LCG(UInt64(seed))
    var curr = cube.copy()
    for _ in range(iterations):
        curr = curr.apply_move(moves[Int(rng.next() % UInt64(len(moves)))], cubelets)
    return curr^

# ---------------------------------------------------------------------------
# 4. Multi-Stage Solver (A* with Reduction Stages)
# ---------------------------------------------------------------------------

def is_inverse(m1: Move, m2: Move) -> Bool: return m1.v == m2.v and m1.direction == -m2.direction
def is_opposite(v1: Vec3, v2: Vec3) -> Bool: return v1.x == -v2.x and v1.y == -v2.y and v1.z == -v2.z
def vec_greater(v1: Vec3, v2: Vec3) -> Bool:
    if v1.x != v2.x: return v1.x > v2.x
    if v1.y != v2.y: return v1.y > v2.y
    return v1.z > v2.z

def check_goal(cube: Cube, cubelets: List[Vec3], stage: Int, sub: Int) -> Bool:
    var top_edges = 0; var top_all = 0; var top_mid = 0
    var bot_edges_pos = 0; var bot_edges_solved = 0; var bot_corners_pos = 0

    for i in range(26):
        var c = cubelets[i]
        var solved = is_cubelet_solved(cube, i, cubelets)
        if c.z == 1 and c.norm1() == 2 and solved: top_edges += 1
        if c.z == 1 and solved: top_all += 1
        if c.z >= 0 and solved: top_mid += 1
        if c.z == -1 and c.norm1() == 2:
            if (cube.rotations[i] @ Vec3(0, 0, -1)) == Vec3(0, 0, -1): bot_edges_pos += 1
            if solved: bot_edges_solved += 1
        if c.z == -1 and c.norm1() == 3 and (cube.rotations[i] @ c) == c: bot_corners_pos += 1

    if stage == 1:
        return (top_edges >= min(4, sub + 1) and top_all >= min(9, sub + 1) and top_mid >= min(17, sub + 1))
    elif stage == 2:
        return (top_mid == 17 and bot_edges_pos >= min(4, sub + 1) and bot_edges_solved >= min(4, sub - 3))
    else: # stage 3
        return (top_mid == 17 and bot_edges_solved == 4 and bot_corners_pos >= min(4, sub + 1))

def compute_heuristic(cube: Cube, cubelets: List[Vec3], tables: DistanceTables, stage: Int, sub: Int) raises -> Float64:
    if stage == 1:
        var sum_sqrt: Float64 = 0.0
        for i in range(26):
            if (cubelets[i].z == 1 if sub < 9 else cubelets[i].z >= 0):
                sum_sqrt += sqrt(Float64(tables.dist_solved[i][cube.rotations[i]]))
        return (sum_sqrt * sum_sqrt) / (8.0 if sub < 9 else 4.0)
    elif stage == 2:
        var sum_sqrt: Float64 = 0.0
        for i in range(26):
            if not (cubelets[i].z == -1 and cubelets[i].norm1() == 3):
                sum_sqrt += sqrt(Float64(tables.dist_solved[i][cube.rotations[i]]))
        return (sum_sqrt * sum_sqrt) / 3.0
    else: # stage 3
        var s1: Float64 = 0.0; var s2: Float64 = 0.0; var s3: Float64 = 0.0
        for i in range(26):
            var c = cubelets[i]
            if c.z == 1: s1 += sqrt(Float64(tables.dist_solved[i][cube.rotations[i]]))
            elif c.z == 0: s2 += sqrt(Float64(tables.dist_solved[i][cube.rotations[i]]))
            else:
                var d = tables.dist_pos[i][cube.rotations[i]] if c.norm1() == 3 else tables.dist_solved[i][cube.rotations[i]]
                s3 += sqrt(Float64(d))
        return (s1 * s1) / 5.0 + (s2 * s2) / 3.0 + (s3 * s3) / 8.0

def astar_step(start: Cube, cubelets: List[Vec3], moves: List[Move],
               tables: DistanceTables, stage: Int, sub: Int,
               mut rng: LCG, random_weight: Float64, max_budget: Int) raises -> Tuple[Cube, List[Move]]:
    if check_goal(start, cubelets, stage, sub): return (start.copy(), List[Move]())
    var budget = max_budget
    while True:
        var pq = PriorityQueue()
        var states = List[Cube](); var state_indices = Dict[Cube, Int]()
        var parent_idx = List[Int](); var move_taken = List[Move](); var cost_so_far = List[Int]()

        states.append(start.copy()); state_indices[start.copy()] = 0
        parent_idx.append(-1); move_taken.append(moves[0]); cost_so_far.append(0)
        pq.push(PQItem(compute_heuristic(start, cubelets, tables, stage, sub), 0))

        var moves_simulated = 0; var budget_exceeded = False; var goal_found = -1

        while len(pq.items) > 0:
            var it = pq.pop()
            var src_idx = it.state_idx
            var src_cube = states[src_idx].copy()
            var has_last = parent_idx[src_idx] != -1
            var last_m = move_taken[src_idx] if has_last else moves[0]

            for m_idx in range(len(moves)):
                var m = moves[m_idx]
                if has_last:
                    if is_inverse(m, last_m): continue
                    if is_opposite(last_m.v, m.v) and vec_greater(last_m.v, m.v): continue

                var dst = src_cube.apply_move(m, cubelets)
                var cost = cost_so_far[src_idx] + 1
                moves_simulated += 1
                if moves_simulated >= budget: budget_exceeded = True; break

                var dst_idx: Int
                if dst in state_indices:
                    dst_idx = state_indices[dst]
                    if cost_so_far[dst_idx] <= cost: continue
                    cost_so_far[dst_idx] = cost; parent_idx[dst_idx] = src_idx; move_taken[dst_idx] = m
                else:
                    dst_idx = len(states)
                    states.append(dst.copy()); state_indices[dst.copy()] = dst_idx
                    parent_idx.append(src_idx); move_taken.append(m); cost_so_far.append(cost)

                if check_goal(dst, cubelets, stage, sub): goal_found = dst_idx; break

                var h = compute_heuristic(dst, cubelets, tables, stage, sub)
                var jitter = 1.0 + (rng.next_float() * 2.0 - 1.0) * random_weight
                pq.push(PQItem(Float64(cost) + jitter * h, dst_idx))

            if goal_found != -1 or budget_exceeded: break

        if goal_found != -1:
            var path = List[Move](); var curr = goal_found
            var final_cube = states[goal_found].copy()
            while parent_idx[curr] != -1:
                path.append(move_taken[curr]); curr = parent_idx[curr]
            var rev = List[Move](); var n = len(path)
            for k in range(n): rev.append(path[n - 1 - k])
            return (final_cube^, rev^)

        budget = max(budget * 3 // 2, budget + 1)

def is_corner_oriented(c: Cube, cubelets: List[Vec3], bottom: Move) -> Bool:
    var idx = -1
    for i in range(26):
        if (c.rotations[i] @ cubelets[i]) == Vec3(1, -1, -1): idx = i; break
    var r = c.rotations[idx]
    var cub = cubelets[idx]
    for _ in range(4):
        if is_cubelet_solved_single(cub, r): return True
        r = bottom.matrix() @ r
    return False

def solve_endgame(cube: Cube, cubelets: List[Vec3]) -> Tuple[Cube, List[Move]]:
    var res_cube = cube.copy(); var solution = List[Move]()
    var left = Move(Vec3(0, -1, 0), 1); var top = Move(Vec3(0, 0, 1), 1); var bottom = Move(Vec3(0, 0, -1), 1)

    var routine = List[Move]()
    for _ in range(2):
        routine.append(left.inverse()); routine.append(top.inverse())
        routine.append(left); routine.append(top)

    for _ in range(4):
        while not is_corner_oriented(res_cube, cubelets, bottom):
            for k in range(len(routine)):
                res_cube = res_cube.apply_move(routine[k], cubelets)
                solution.append(routine[k])
        res_cube = res_cube.apply_move(bottom, cubelets)
        solution.append(bottom)

    while not is_cube_solved(res_cube, cubelets):
        res_cube = res_cube.apply_move(bottom, cubelets)
        solution.append(bottom)

    return (res_cube^, solution^)

def solve(cube: Cube, cubelets: List[Vec3], moves: List[Move], tables: DistanceTables) raises -> Tuple[Cube, List[Move]]:
    var total_solution = List[Move]()
    var curr = cube.copy(); var rng = LCG(12345)

    for sub in range(17):
        var step = astar_step(curr, cubelets, moves, tables, 1, sub, rng, 0.25, 100_000)
        curr = step[0].copy()
        for m in step[1]: total_solution.append(m)

    for sub in range(8):
        var step = astar_step(curr, cubelets, moves, tables, 2, sub, rng, 0.25, 100_000)
        curr = step[0].copy()
        for m in step[1]: total_solution.append(m)

    for sub in range(4):
        var step = astar_step(curr, cubelets, moves, tables, 3, sub, rng, 0.30, 100_000)
        curr = step[0].copy()
        for m in step[1]: total_solution.append(m)

    var eg = solve_endgame(curr, cubelets)
    curr = eg[0].copy()
    for m in eg[1]: total_solution.append(m)

    return (curr^, total_solution^)

# ---------------------------------------------------------------------------
# 5. CLI & Verification
# ---------------------------------------------------------------------------

def main() raises:
    print("Rubix (Mojo): A Functional Pearl in Linear Algebra")
    var cubelets = init_cubelets()
    var moves = init_moves()
    var tables = build_distance_tables(cubelets, moves)
    var solved = Cube()

    print("Canonical cubelets:", len(cubelets))
    print("Elementary slice moves:", len(moves))
    print("Is initial cube solved?", is_cube_solved(solved, cubelets))

    # Move order-4 cycle test
    for i in range(len(moves)):
        var c = solved.copy()
        for _ in range(4): c = c.apply_move(moves[i], cubelets)
        if not (c == solved):
            print("Error: Move", i, "does not have cycle length 4!")
            return
    print("All 12 elementary slice moves verified with cycle length 4.")

    # Benchmark raw linear algebra move throughput
    var t0 = perf_counter_ns()
    var c = solved.copy()
    var N = 1_000_000
    for i in range(N):
        c = c.apply_move(moves[i % len(moves)], cubelets)
    var t1 = perf_counter_ns()
    var elapsed_sim = Float64(t1 - t0) / 1_000_000_000.0
    print("Simulated", N, "moves in", elapsed_sim, "sec (", Int(Float64(N) / elapsed_sim), "moves/sec )")

    # Scramble & solve verification
    print("-" * 50)
    print("Scrambling cube (100 moves)...")
    var scrambled = shuffle(solved, moves, cubelets, 100, 42)
    print("Is scrambled cube solved?", is_cube_solved(scrambled, cubelets))

    var t_solve0 = perf_counter_ns()
    var sol = solve(scrambled, cubelets, moves, tables)
    var t_solve1 = perf_counter_ns()

    var final_cube = sol[0].copy()
    var total_moves = sol[1].copy()
    var elapsed_solve = Float64(t_solve1 - t_solve0) / 1_000_000_000.0

    print("Solved:", is_cube_solved(final_cube, cubelets))
    print("Solution length:", len(total_moves), "moves")
    print("Solve time:", elapsed_solve, "sec")
    print("-" * 50)
