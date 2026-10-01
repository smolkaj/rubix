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
struct Vec3(ImplicitlyCopyable, KeyElement):
    var x: Int; var y: Int; var z: Int
    def dot(self, o: Vec3) -> Int: return self.x * o.x + self.y * o.y + self.z * o.z
    def norm1(self) -> Int: return abs(self.x) + abs(self.y) + abs(self.z)

@fieldwise_init
struct Mat3(ImplicitlyCopyable, KeyElement):
    var r0: Vec3; var r1: Vec3; var r2: Vec3

    @staticmethod
    def identity() -> Mat3: return Mat3(Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    def __matmul__(self, v: Vec3) -> Vec3:
        return Vec3(self.r0.dot(v), self.r1.dot(v), self.r2.dot(v))

    def col(self, j: Int) -> Vec3:
        return Vec3(self.r0.x if j == 0 else (self.r0.y if j == 1 else self.r0.z),
                    self.r1.x if j == 0 else (self.r1.y if j == 1 else self.r1.z),
                    self.r2.x if j == 0 else (self.r2.y if j == 1 else self.r2.z))

    def __matmul__(self, o: Mat3) -> Mat3:
        var c0 = self @ o.col(0); var c1 = self @ o.col(1); var c2 = self @ o.col(2)
        return Mat3(Vec3(c0.x, c1.x, c2.x), Vec3(c0.y, c1.y, c2.y), Vec3(c0.z, c1.z, c2.z))

# ---------------------------------------------------------------------------
# 2. Domain Model: Cubelets, Moves, and State
# ---------------------------------------------------------------------------

@fieldwise_init
struct Move(ImplicitlyCopyable, KeyElement):
    var v: Vec3
    var direction: Int

    def inverse(self) -> Move: return Move(self.v, -self.direction)

    def matrix(self) -> Mat3:
        if self.v.x != 0:
            return Mat3(Vec3(1, 0, 0), Vec3(0, 0, self.direction), Vec3(0, -self.direction, 0))
        elif self.v.y != 0:
            return Mat3(Vec3(0, 0, self.direction), Vec3(0, 1, 0), Vec3(-self.direction, 0, 0))
        else:
            return Mat3(Vec3(0, self.direction, 0), Vec3(-self.direction, 0, 0), Vec3(0, 0, 1))

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
    for v in [Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1)]:
        res.append(Move(v, 1)); res.append(Move(v, -1))
    return res^

@fieldwise_init
struct Cube(Copyable, KeyElement):
    var rotations: List[Mat3]

    def __init__(out self):
        self.rotations = List[Mat3]()
        for _ in range(26): self.rotations.append(Mat3.identity())

    def apply_move(self, m: Move, cubelets: List[Vec3]) -> Cube:
        var res = Cube()
        var M = m.matrix()
        for i in range(26):
            res.rotations[i] = (M @ self.rotations[i]) if (m.v.dot(self.rotations[i] @ cubelets[i]) > 0) else self.rotations[i]
        return res^

def is_cubelet_solved(c: Vec3, r: Mat3) -> Bool:
    if (r @ c) != c: return False
    if c.x != 0 and (r @ Vec3(c.x, 0, 0)) != Vec3(c.x, 0, 0): return False
    if c.y != 0 and (r @ Vec3(0, c.y, 0)) != Vec3(0, c.y, 0): return False
    if c.z != 0 and (r @ Vec3(0, 0, c.z)) != Vec3(0, 0, c.z): return False
    return True

def is_cube_solved(cube: Cube, cubelets: List[Vec3]) -> Bool:
    for i in range(26):
        if not is_cubelet_solved(cubelets[i], cube.rotations[i]): return False
    return True

# ---------------------------------------------------------------------------
# 3. Cayley Geodesic Heuristics
# ---------------------------------------------------------------------------

def bfs_distances(c: Vec3, pos_only: Bool, orientations: List[Mat3], moves: List[Move]) raises -> Dict[Mat3, Int]:
    var dist = Dict[Mat3, Int](); var q = List[Mat3]()
    for j in range(len(orientations)):
        var r = orientations[j]
        if (r @ c == c) if pos_only else is_cubelet_solved(c, r):
            dist[r] = 0; q.append(r)
    var head = 0
    while head < len(q):
        var r = q[head]; head += 1
        var d = dist[r]
        for m in moves:
            var prev = m.matrix() @ r
            if prev not in dist: dist[prev] = d + 1; q.append(prev)
    return dist^

@fieldwise_init
struct DistanceTables:
    var dist_solved: List[Dict[Mat3, Int]]
    var dist_pos: List[Dict[Mat3, Int]]

def build_distance_tables(cubelets: List[Vec3], moves: List[Move]) raises -> DistanceTables:
    var orientations = List[Mat3]()
    orientations.append(Mat3.identity())
    var head = 0
    while head < len(orientations):
        var r = orientations[head]; head += 1
        for m in moves:
            var nxt = m.matrix() @ r
            var exists = False
            for k in range(len(orientations)):
                if orientations[k] == nxt: exists = True; break
            if not exists: orientations.append(nxt)

    var ds = List[Dict[Mat3, Int]](); var dp = List[Dict[Mat3, Int]]()
    for i in range(26):
        ds.append(bfs_distances(cubelets[i], False, orientations, moves))
        dp.append(bfs_distances(cubelets[i], True, orientations, moves))
    return DistanceTables(ds^, dp^)

# ---------------------------------------------------------------------------
# 4. Search Structures: Priority Queue & Random Jitter
# ---------------------------------------------------------------------------

@fieldwise_init
struct PQItem(Copyable):
    var priority: Float64
    var state_idx: Int

struct PriorityQueue:
    var items: List[PQItem]
    def __init__(out self): self.items = List[PQItem]()

    def push(mut self, item: PQItem):
        self.items.append(item.copy())
        var i = len(self.items) - 1
        while i > 0 and self.items[i].priority < self.items[(i - 1) // 2].priority:
            var p = (i - 1) // 2
            var tmp = self.items[i].copy(); self.items[i] = self.items[p].copy(); self.items[p] = tmp^
            i = p

    def pop(mut self) -> PQItem:
        var top = self.items[0].copy(); var last = self.items.pop()
        if len(self.items) > 0:
            self.items[0] = last^
            var i = 0
            while 2 * i + 1 < len(self.items):
                var sm = 2 * i + 1
                if sm + 1 < len(self.items) and self.items[sm + 1].priority < self.items[sm].priority: sm += 1
                if self.items[sm].priority >= self.items[i].priority: break
                var tmp = self.items[i].copy(); self.items[i] = self.items[sm].copy(); self.items[sm] = tmp^
                i = sm
        return top^

struct LCG:
    var state: UInt64
    def __init__(out self, seed: UInt64): self.state = seed
    def next(mut self) -> UInt64:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return self.state
    def jitter(mut self, weight: Float64) -> Float64:
        var f = Float64(self.next() & 0xFFFFFFFFFFFFF) / Float64(0x10000000000000)
        return 1.0 + (f * 2.0 - 1.0) * weight

def shuffle(cube: Cube, moves: List[Move], cubelets: List[Vec3], n: Int, seed: Int) -> Cube:
    var rng = LCG(UInt64(seed)); var curr = cube.copy()
    for _ in range(n): curr = curr.apply_move(moves[Int(rng.next() % UInt64(len(moves)))], cubelets)
    return curr^

# ---------------------------------------------------------------------------
# 5. Multi-Stage Solver (A* with Reduction Stages)
# ---------------------------------------------------------------------------

def check_goal(cube: Cube, cubelets: List[Vec3], stage: Int, sub: Int) -> Bool:
    var top_edges = 0; var top_all = 0; var top_mid = 0
    var bot_edges_pos = 0; var bot_edges_solved = 0; var bot_corners_pos = 0

    for i in range(26):
        var c = cubelets[i]; var r = cube.rotations[i]
        var solved = is_cubelet_solved(c, r)
        if c.z == 1 and c.norm1() == 2 and solved: top_edges += 1
        if c.z == 1 and solved: top_all += 1
        if c.z >= 0 and solved: top_mid += 1
        if c.z == -1 and c.norm1() == 2:
            if (r @ Vec3(0, 0, -1)) == Vec3(0, 0, -1): bot_edges_pos += 1
            if solved: bot_edges_solved += 1
        if c.z == -1 and c.norm1() == 3 and (r @ c) == c: bot_corners_pos += 1

    if stage == 1:
        return top_edges >= min(4, sub + 1) and top_all >= min(9, sub + 1) and top_mid >= min(17, sub + 1)
    elif stage == 2:
        return top_mid == 17 and bot_edges_pos >= min(4, sub + 1) and bot_edges_solved >= min(4, sub - 3)
    else:
        return top_mid == 17 and bot_edges_solved == 4 and bot_corners_pos >= min(4, sub + 1)

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
    else:
        var s1: Float64 = 0.0; var s2: Float64 = 0.0; var s3: Float64 = 0.0
        for i in range(26):
            var c = cubelets[i]; var r = cube.rotations[i]
            if c.z == 1: s1 += sqrt(Float64(tables.dist_solved[i][r]))
            elif c.z == 0: s2 += sqrt(Float64(tables.dist_solved[i][r]))
            else:
                var d = tables.dist_pos[i][r] if c.norm1() == 3 else tables.dist_solved[i][r]
                s3 += sqrt(Float64(d))
        return (s1 * s1) / 5.0 + (s2 * s2) / 3.0 + (s3 * s3) / 8.0

def astar_step(start: Cube, cubelets: List[Vec3], moves: List[Move],
               tables: DistanceTables, stage: Int, sub: Int,
               mut rng: LCG, weight: Float64) raises -> Tuple[Cube, List[Move]]:
    if check_goal(start, cubelets, stage, sub): return (start.copy(), List[Move]())
    var budget = 100_000
    while True:
        var pq = PriorityQueue()
        var states = List[Cube](); var state_idx = Dict[Cube, Int]()
        var parent = List[Int](); var move_hist = List[Move](); var cost = List[Int]()

        states.append(start.copy()); state_idx[start.copy()] = 0
        parent.append(-1); move_hist.append(moves[0]); cost.append(0)
        pq.push(PQItem(compute_heuristic(start, cubelets, tables, stage, sub), 0))

        var moves_sim = 0; var goal_id = -1
        while len(pq.items) > 0:
            var src = pq.pop().state_idx
            var src_cube = states[src].copy()
            var has_last = parent[src] != -1
            var last_m = move_hist[src] if has_last else moves[0]

            for m in moves:
                if has_last:
                    if m.v == last_m.v and m.direction == -last_m.direction: continue
                    if m.v.x == -last_m.v.x and m.v.y == -last_m.v.y and m.v.z == -last_m.v.z:
                        if (last_m.v.x > m.v.x) or (last_m.v.x == m.v.x and last_m.v.y > m.v.y) or (last_m.v.x == m.v.x and last_m.v.y == m.v.y and last_m.v.z > m.v.z): continue

                var dst = src_cube.apply_move(m, cubelets)
                var next_c = cost[src] + 1
                moves_sim += 1
                if moves_sim >= budget: break

                if dst in state_idx:
                    var id = state_idx[dst]
                    if cost[id] <= next_c: continue
                    cost[id] = next_c; parent[id] = src; move_hist[id] = m
                else:
                    var id = len(states)
                    states.append(dst.copy()); state_idx[dst.copy()] = id
                    parent.append(src); move_hist.append(m); cost.append(next_c)
                    if check_goal(dst, cubelets, stage, sub): goal_id = id; break
                    var h = compute_heuristic(dst, cubelets, tables, stage, sub)
                    pq.push(PQItem(Float64(next_c) + rng.jitter(weight) * h, id))

            if goal_id != -1 or moves_sim >= budget: break

        if goal_id != -1:
            var path = List[Move](); var curr = goal_id
            var final_cube = states[goal_id].copy()
            while parent[curr] != -1:
                path.append(move_hist[curr]); curr = parent[curr]
            var rev = List[Move]()
            for k in range(len(path)): rev.append(path[len(path) - 1 - k])
            return (final_cube^, rev^)
        budget = max(budget * 3 // 2, budget + 1)

def solve_endgame(cube: Cube, cubelets: List[Vec3]) -> Tuple[Cube, List[Move]]:
    var res = cube.copy(); var sol = List[Move]()
    var L = Move(Vec3(0, -1, 0), 1); var U = Move(Vec3(0, 0, 1), 1); var D = Move(Vec3(0, 0, -1), 1)
    var routine = [L.inverse(), U.inverse(), L, U, L.inverse(), U.inverse(), L, U]

    for _ in range(4):
        while True:
            var r = Mat3.identity(); var cub = Vec3(0, 0, 0)
            for i in range(26):
                if (res.rotations[i] @ cubelets[i]) == Vec3(1, -1, -1):
                    r = res.rotations[i]; cub = cubelets[i]; break
            var oriented = False
            for _ in range(4):
                if is_cubelet_solved(cub, r): oriented = True; break
                r = D.matrix() @ r
            if oriented: break
            for m in routine:
                res = res.apply_move(m, cubelets); sol.append(m)
        res = res.apply_move(D, cubelets); sol.append(D)

    while not is_cube_solved(res, cubelets):
        res = res.apply_move(D, cubelets); sol.append(D)
    return (res^, sol^)

def solve(cube: Cube, cubelets: List[Vec3], moves: List[Move], tables: DistanceTables) raises -> Tuple[Cube, List[Move]]:
    var sol = List[Move](); var curr = cube.copy(); var rng = LCG(12345)
    for sub in range(17):
        var s = astar_step(curr, cubelets, moves, tables, 1, sub, rng, 0.25)
        curr = s[0].copy(); sol.extend(s[1].copy())
    for sub in range(8):
        var s = astar_step(curr, cubelets, moves, tables, 2, sub, rng, 0.25)
        curr = s[0].copy(); sol.extend(s[1].copy())
    for sub in range(4):
        var s = astar_step(curr, cubelets, moves, tables, 3, sub, rng, 0.30)
        curr = s[0].copy(); sol.extend(s[1].copy())
    var eg = solve_endgame(curr, cubelets)
    sol.extend(eg[1].copy())
    return (eg[0].copy(), sol^)

# ---------------------------------------------------------------------------
# 6. CLI & Verification
# ---------------------------------------------------------------------------

def main() raises:
    print("Rubix (Mojo): A Functional Pearl in Linear Algebra")
    var cubelets = init_cubelets()
    var moves = init_moves()
    var tables = build_distance_tables(cubelets, moves)
    var solved = Cube()

    # Move order-4 cycle test
    for m in moves:
        var c = solved.copy()
        for _ in range(4): c = c.apply_move(m, cubelets)
        if not (c == solved): print("Cycle test failed!"); return
    print("All 12 elementary slice moves verified with cycle length 4.")

    # Benchmark raw linear algebra move throughput
    var t0 = perf_counter_ns()
    var c = solved.copy()
    var N = 1_000_000
    for i in range(N): c = c.apply_move(moves[i % len(moves)], cubelets)
    var t1 = perf_counter_ns()
    var elapsed_sim = Float64(t1 - t0) / 1_000_000_000.0
    print("Simulated", N, "moves in", elapsed_sim, "sec (", Int(Float64(N) / elapsed_sim), "moves/sec )")

    # Scramble & solve
    print("-" * 50)
    print("Scrambling cube (100 moves)...")
    var scrambled = shuffle(solved, moves, cubelets, 100, 42)
    var t_s0 = perf_counter_ns()
    var sol = solve(scrambled, cubelets, moves, tables)
    var t_s1 = perf_counter_ns()
    print("Solved:", is_cube_solved(sol[0], cubelets))
    print("Solution length:", len(sol[1]), "moves")
    print("Solve time:", Float64(t_s1 - t_s0) / 1_000_000_000.0, "sec")
    print("-" * 50)
