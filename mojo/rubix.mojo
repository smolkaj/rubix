"""Rubix in Mojo: A Functional Pearl in Linear Algebra.

A Rubik's cube modeled as a discrete object in 3D Euclidean space centered at (0,0,0).
Each cubelet has coordinates (x, y, z) in {-1, 0, 1}^3.
Each move is a 90-degree hyperplane rotation.
"""

from std.time import perf_counter_ns
from std.math import sqrt

# 1. Discrete 3D Vector & Matrix Algebra

@fieldwise_init
struct Vec3(ImplicitlyCopyable, KeyElement):
    var x: Int
    var y: Int
    var z: Int
    def dot(self, o: Vec3) -> Int: return self.x * o.x + self.y * o.y + self.z * o.z
    def norm1(self) -> Int: return abs(self.x) + abs(self.y) + abs(self.z)
    def __neg__(self) -> Vec3: return Vec3(-self.x, -self.y, -self.z)
    def __gt__(self, o: Vec3) -> Bool:
        if self.x != o.x: return self.x > o.x
        if self.y != o.y: return self.y > o.y
        return self.z > o.z

@fieldwise_init
struct Mat3(ImplicitlyCopyable, KeyElement):
    var r0: Vec3
    var r1: Vec3
    var r2: Vec3
    @staticmethod
    def identity() -> Mat3: return Mat3(Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
    def __matmul__(self, v: Vec3) -> Vec3: return Vec3(self.r0.dot(v), self.r1.dot(v), self.r2.dot(v))
    def col(self, j: Int) -> Vec3:
        if j == 0: return Vec3(self.r0.x, self.r1.x, self.r2.x)
        return Vec3(self.r0.y, self.r1.y, self.r2.y) if j == 1 else Vec3(self.r0.z, self.r1.z, self.r2.z)
    def __matmul__(self, o: Mat3) -> Mat3:
        var c0 = self @ o.col(0)
        var c1 = self @ o.col(1)
        var c2 = self @ o.col(2)
        return Mat3(Vec3(c0.x, c1.x, c2.x), Vec3(c0.y, c1.y, c2.y), Vec3(c0.z, c1.z, c2.z))

# 2. Domain Model: Cubelets, Moves, and State

@fieldwise_init
struct Move(ImplicitlyCopyable, KeyElement):
    var v: Vec3
    var direction: Int
    def inverse(self) -> Move: return Move(self.v, -self.direction)
    def matrix(self) -> Mat3:
        var d = self.direction
        if self.v.x != 0: return Mat3(Vec3(1, 0, 0), Vec3(0, 0, d), Vec3(0, -d, 0))
        if self.v.y != 0: return Mat3(Vec3(0, 0, d), Vec3(0, 1, 0), Vec3(-d, 0, 0))
        return Mat3(Vec3(0, d, 0), Vec3(-d, 0, 0), Vec3(0, 0, 1))

def init_cubelets() -> List[Vec3]:
    var res = List[Vec3]()
    for i in range(27):
        var x = (i % 3) - 1
        var y = ((i // 3) % 3) - 1
        var z = (i // 9) - 1
        if x != 0 or y != 0 or z != 0:
            res.append(Vec3(x, y, z))
    return res^

def init_moves() -> List[Move]:
    var res = List[Move]()
    for v in [Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1)]:
        res.append(Move(v, 1))
        res.append(Move(v, -1))
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
            var pos = self.rotations[i] @ cubelets[i]
            res.rotations[i] = (M @ self.rotations[i]) if (m.v.dot(pos) > 0) else self.rotations[i]
        return res^

def is_cubelet_solved(c: Vec3, r: Mat3) -> Bool:
    if (r @ c) != c: return False
    if c.x != 0 and (r @ Vec3(c.x, 0, 0)) != Vec3(c.x, 0, 0): return False
    if c.y != 0 and (r @ Vec3(0, c.y, 0)) != Vec3(0, c.y, 0): return False
    return c.z == 0 or (r @ Vec3(0, 0, c.z)) == Vec3(0, 0, c.z)

def is_cube_solved(cube: Cube, cubelets: List[Vec3]) -> Bool:
    for i in range(26):
        if not is_cubelet_solved(cubelets[i], cube.rotations[i]): return False
    return True

# 3. Cayley Geodesic Heuristics

def bfs_distances(c: Vec3, pos_only: Bool, orientations: List[Mat3], moves: List[Move]) raises -> Dict[Mat3, Int]:
    var dist = Dict[Mat3, Int]()
    var queue = List[Mat3]()
    for r in orientations:
        if (r @ c == c) if pos_only else is_cubelet_solved(c, r):
            dist[r] = 0
            queue.append(r)
    var head = 0
    while head < len(queue):
        var r = queue[head]
        head += 1
        for m in moves:
            var prev = m.matrix() @ r
            if prev not in dist:
                dist[prev] = dist[r] + 1
                queue.append(prev)
    return dist^

@fieldwise_init
struct DistanceTables:
    var dist_solved: List[Dict[Mat3, Int]]
    var dist_pos: List[Dict[Mat3, Int]]

def build_distance_tables(cubelets: List[Vec3], moves: List[Move]) raises -> DistanceTables:
    var orientations = List[Mat3]()
    var seen = Dict[Mat3, Bool]()
    orientations.append(Mat3.identity())
    seen[Mat3.identity()] = True
    var head = 0
    while head < len(orientations):
        var r = orientations[head]
        head += 1
        for m in moves:
            var nxt = m.matrix() @ r
            if nxt not in seen:
                seen[nxt] = True
                orientations.append(nxt)
    var ds = List[Dict[Mat3, Int]]()
    var dp = List[Dict[Mat3, Int]]()
    for i in range(26):
        ds.append(bfs_distances(cubelets[i], False, orientations, moves))
        dp.append(bfs_distances(cubelets[i], True, orientations, moves))
    return DistanceTables(ds^, dp^)

# 4. Search Structures: Priority Queue & Random Jitter

@fieldwise_init
struct PQItem(Copyable):
    var priority: Float64
    var state_idx: Int

struct PriorityQueue:
    var items: List[PQItem]
    def __init__(out self): self.items = List[PQItem]()

    def swap(mut self, i: Int, j: Int):
        var tmp = self.items[i].copy()
        self.items[i] = self.items[j].copy()
        self.items[j] = tmp^

    def push(mut self, item: PQItem):
        self.items.append(item.copy())
        var i = len(self.items) - 1
        while i > 0 and self.items[i].priority < self.items[(i - 1) // 2].priority:
            self.swap(i, (i - 1) // 2)
            i = (i - 1) // 2

    def pop(mut self) -> PQItem:
        var top = self.items[0].copy()
        var last = self.items.pop()
        if len(self.items) == 0: return top^
        self.items[0] = last^
        var i = 0
        while 2 * i + 1 < len(self.items):
            var sm = 2 * i + 1
            if sm + 1 < len(self.items) and self.items[sm + 1].priority < self.items[sm].priority: sm += 1
            if self.items[sm].priority >= self.items[i].priority: break
            self.swap(i, sm)
            i = sm
        return top^

struct LCG:
    var state: UInt64
    def __init__(out self, seed: UInt64): self.state = seed
    def next(mut self) -> UInt64:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return self.state
    def jitter(mut self, weight: Float64) -> Float64:
        return 1.0 + (Float64(self.next() & 0xFFFFFFFFFFFFF) / Float64(0x10000000000000) * 2.0 - 1.0) * weight

def shuffle(cube: Cube, moves: List[Move], cubelets: List[Vec3], n: Int, seed: Int) -> Cube:
    var rng = LCG(UInt64(seed))
    var curr = cube.copy()
    for _ in range(n):
        var idx = Int(rng.next() % UInt64(len(moves)))
        curr = curr.apply_move(moves[idx], cubelets)
    return curr^

# 5. Multi-Stage Solver (A* with Reduction Stages)

@fieldwise_init
struct SearchNode(Copyable):
    var cube: Cube
    var parent: Int
    var move: Move
    var cost: Int

def check_goal(cube: Cube, cubelets: List[Vec3], stage: Int, sub: Int) -> Bool:
    var top_edges = 0
    var top_all = 0
    var top_mid = 0
    var bot_edges_pos = 0
    var bot_edges_solved = 0
    var bot_corners_pos = 0
    for i in range(26):
        var c = cubelets[i]
        var r = cube.rotations[i]
        var solved = is_cubelet_solved(c, r)
        if solved and c.z >= 0:
            top_mid += 1
            if c.z == 1:
                top_all += 1
                if c.norm1() == 2: top_edges += 1
        if c.z == -1:
            if c.norm1() == 2:
                if (r @ Vec3(0, 0, -1)) == Vec3(0, 0, -1): bot_edges_pos += 1
                if solved: bot_edges_solved += 1
            elif (r @ c) == c:
                bot_corners_pos += 1

    if stage == 1: return top_edges >= min(4, sub + 1) and top_all >= min(9, sub + 1) and top_mid >= min(17, sub + 1)
    if stage == 2: return top_mid == 17 and bot_edges_pos >= min(4, sub + 1) and bot_edges_solved >= min(4, sub - 3)
    return top_mid == 17 and bot_edges_solved == 4 and bot_corners_pos >= min(4, sub + 1)

def compute_heuristic(cube: Cube, cubelets: List[Vec3], tables: DistanceTables, stage: Int, sub: Int) raises -> Float64:
    if stage <= 2:
        var sum_sqrt: Float64 = 0.0
        for i in range(26):
            var c = cubelets[i]
            var active = (c.z == 1 if sub < 9 else c.z >= 0) if stage == 1 else not (c.z == -1 and c.norm1() == 3)
            if active:
                sum_sqrt += sqrt(Float64(tables.dist_solved[i][cube.rotations[i]]))
        var norm = (8.0 if sub < 9 else 4.0) if stage == 1 else 3.0
        return (sum_sqrt * sum_sqrt) / norm

    var s1: Float64 = 0.0
    var s2: Float64 = 0.0
    var s3: Float64 = 0.0
    for i in range(26):
        var c = cubelets[i]
        var r = cube.rotations[i]
        if c.z == 1:
            s1 += sqrt(Float64(tables.dist_solved[i][r]))
        elif c.z == 0:
            s2 += sqrt(Float64(tables.dist_solved[i][r]))
        else:
            var d = tables.dist_pos[i][r] if c.norm1() == 3 else tables.dist_solved[i][r]
            s3 += sqrt(Float64(d))
    return (s1 * s1) / 5.0 + (s2 * s2) / 3.0 + (s3 * s3) / 8.0

def astar_step(start: Cube, cubelets: List[Vec3], moves: List[Move], tables: DistanceTables,
               stage: Int, sub: Int, mut rng: LCG, weight: Float64) raises -> Tuple[Cube, List[Move]]:
    if check_goal(start, cubelets, stage, sub):
        return (start.copy(), List[Move]())

    var budget = 100_000
    while True:
        var pq = PriorityQueue()
        var nodes = List[SearchNode]()
        var state_idx = Dict[Cube, Int]()
        nodes.append(SearchNode(start.copy(), -1, moves[0], 0))
        state_idx[start.copy()] = 0
        pq.push(PQItem(compute_heuristic(start, cubelets, tables, stage, sub), 0))
        var moves_sim = 0
        var goal_id = -1
        while len(pq.items) > 0:
            var src = pq.pop().state_idx
            var src_cube = nodes[src].cube.copy()
            var has_last = nodes[src].parent != -1
            var last_m = nodes[src].move
            for m in moves:
                if has_last and ((m.v == last_m.v and m.direction == -last_m.direction) or
                                 (m.v == -last_m.v and last_m.v > m.v)):
                    continue

                var dst = src_cube.apply_move(m, cubelets)
                var next_c = nodes[src].cost + 1
                moves_sim += 1
                if moves_sim >= budget: break

                var id: Int
                if dst in state_idx:
                    id = state_idx[dst]
                    if nodes[id].cost <= next_c: continue
                    nodes[id].cost = next_c
                    nodes[id].parent = src
                    nodes[id].move = m
                else:
                    id = len(nodes)
                    nodes.append(SearchNode(dst.copy(), src, m, next_c))
                    state_idx[dst.copy()] = id
                    if check_goal(dst, cubelets, stage, sub):
                        goal_id = id
                        break

                var h = compute_heuristic(dst, cubelets, tables, stage, sub)
                pq.push(PQItem(Float64(next_c) + rng.jitter(weight) * h, id))

            if goal_id != -1 or moves_sim >= budget: break

        if goal_id != -1:
            var path = List[Move]()
            var curr = goal_id
            while nodes[curr].parent != -1:
                path.append(nodes[curr].move)
                curr = nodes[curr].parent
            for k in range(len(path) // 2):
                var tmp = path[k]
                path[k] = path[len(path) - 1 - k]
                path[len(path) - 1 - k] = tmp
            return (nodes[goal_id].cube.copy(), path^)

        budget = max(budget * 3 // 2, budget + 1)

def solve_endgame(cube: Cube, cubelets: List[Vec3]) -> Tuple[Cube, List[Move]]:
    var res = cube.copy()
    var solution = List[Move]()
    var left = Move(Vec3(0, -1, 0), 1)
    var top = Move(Vec3(0, 0, 1), 1)
    var bottom = Move(Vec3(0, 0, -1), 1)
    var routine = [left.inverse(), top.inverse(), left, top, left.inverse(), top.inverse(), left, top]

    for _ in range(4):
        while True:
            var idx = 0
            for i in range(26):
                if (res.rotations[i] @ cubelets[i]) == Vec3(1, -1, -1):
                    idx = i
                    break
            var ok = False
            var r = res.rotations[idx]
            for _ in range(4):
                ok = is_cubelet_solved(cubelets[idx], r)
                if ok: break
                r = bottom.matrix() @ r
            if ok: break
            for m in routine:
                res = res.apply_move(m, cubelets)
                solution.append(m)
        res = res.apply_move(bottom, cubelets)
        solution.append(bottom)

    while not is_cube_solved(res, cubelets):
        res = res.apply_move(bottom, cubelets)
        solution.append(bottom)
    return (res^, solution^)

def solve(cube: Cube, cubelets: List[Vec3], moves: List[Move], tables: DistanceTables) raises -> Tuple[Cube, List[Move]]:
    var solution = List[Move]()
    var curr = cube.copy()
    var rng = LCG(12345)
    var counts = [17, 8, 4]
    var weights = [0.25, 0.25, 0.30]
    for st in range(3):
        for sub in range(counts[st]):
            var step = astar_step(curr, cubelets, moves, tables, st + 1, sub, rng, weights[st])
            curr = step[0].copy()
            solution.extend(step[1].copy())
    var endgame = solve_endgame(curr, cubelets)
    solution.extend(endgame[1].copy())
    return (endgame[0].copy(), solution^)

# 6. CLI & Verification

def main() raises:
    print("Rubix (Mojo): A Functional Pearl in Linear Algebra")
    var cubelets = init_cubelets()
    var moves = init_moves()
    var tables = build_distance_tables(cubelets, moves)
    var solved = Cube()
    for m in moves:
        var c = solved.copy()
        for _ in range(4): c = c.apply_move(m, cubelets)
        if c != solved: raise Error("Cycle length != 4")
    print("All 12 elementary slice moves verified with cycle length 4.")
    var t0 = perf_counter_ns()
    var c = solved.copy()
    for i in range(1_000_000): c = c.apply_move(moves[i % 12], cubelets)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("Simulated 1000000 moves in", dt, "sec (", Int(1e6 / dt), "moves/sec )")
    print("Scrambling cube (100 moves)...")
    t0 = perf_counter_ns()
    var sol = solve(shuffle(solved, moves, cubelets, 100, 42), cubelets, moves, tables)
    dt = Float64(perf_counter_ns() - t0) / 1e9
    print("Solved:", is_cube_solved(sol[0], cubelets), "in", len(sol[1]), "moves (", dt, "sec )")
