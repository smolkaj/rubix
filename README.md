<p align="center">
  <img src="img/logo.svg" alt="Eigencube logo: a cube with its rotation axis rising from the top center" width="160">
</p>

<h1 align="center">Eigencube</h1>

A Rubik's Cube model, solver, and visualizer built from vectors, rotation matrices, and dot products.

Each cubelet has a home coordinate and a rotation matrix. The same representation describes its position, its sticker directions, which face turns affect it, and whether it is solved. [`eigencube.py`](eigencube.py) implements the model and a multi-phase solver without external puzzle libraries or precomputed pattern databases.

Eigencube is an educational project: the goal is code you can read as a mathematical specification. The solver finds solutions; it does not promise shortest solutions or a fixed solve time. Webcam scanning is experimental and is not connected to the visualizer.

## Try it

Requires Python 3.10+.

```bash
git clone https://github.com/smolkaj/eigencube.git
cd eigencube
python3 -m venv eigencube_env
source eigencube_env/bin/activate
pip install -r requirements.txt
python eigencube_gui.py
```

Click **Shuffle** to scramble the cube and **Solve** to compute a solution in the background. Use the arrow keys to step forward or backward through the moves; holding a key speeds up playback.

![Eigencube visualizer](img/gui-preview.png)

For the command-line solver:

```bash
python eigencube.py       # seed 42
python eigencube.py 123   # another reproducible scramble
```

## The model

https://github.com/user-attachments/assets/5630e7e8-6458-42cd-a1d8-5ffd148936ac

The [16-minute animated explainer](explainer/README.md) walks through the encoding. Its cube geometry and code listings come directly from the implementation. The geometric approach was inspired by Grant Sanderson's [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).

| Concept | In linear algebra |
|:--|:--|
| **Cubelets** | vectors $c \in \lbrace -1, 0, 1 \rbrace^3$: each cubelet's home address, from the fixed core |
| **Cubelet type** | $\lVert c \rVert_1 = \lvert x \rvert + \lvert y \rvert + \lvert z \rvert$ = number of stickers (core 0, center 1, edge 2, corner 3) |
| **Colors** | $\pm\mathbf{e}_x, \pm\mathbf{e}_y, \pm\mathbf{e}_z$: the never-moving centers, each an eigenvector of the turns around it |
| **Sticker colors** | the non-zero columns of $\mathrm{diag}(c)$ |
| **State** | one configuration $(c, R)$ per cubelet, with $R$ a rotation matrix |
| **Position** | $p = R\,c$: where the cubelet is now |
| **Move** | the cubelets with $\mathbf{v} \cdot (R\,c) > 0$ turn ($\mathbf{v}$: the face's axis): $R \leftarrow M R$ ($M$: the quarter-turn matrix) |
| **Solved** | $R \cdot \mathrm{diag}(c) = \mathrm{diag}(c)$ for every cubelet |


## How the solver works

The solver uses heuristic search to reach progressively stricter goals:

1. Solve the top edges, then the rest of the top and middle layers.
2. Orient and place the bottom edges.
3. Place the bottom corners.
4. Twist the corners and align the bottom face using a fixed endgame routine.

Heuristics combine distances for individual cubelets, calculated on demand and cached. Search priorities use randomized weights; unsuccessful searches restart with increasing move budgets. The phase goals retain completed layers at phase boundaries, while intermediate search moves may disturb them.

These are practical search choices, rather than an optimality guarantee. Runtime and solution length depend on the scramble and machine; use the benchmark to measure them.

## Benchmark

```bash
# Ten fixed seeds (0–9), 100 scramble moves each, 60-second timeout per trial
python eigencube.py --benchmark --output benchmark.json

# Choose seeds, scramble length, and timeout explicitly
python eigencube.py --benchmark --seeds 0 42 123 --scramble-moves 100 --timeout 120
```

Each trial runs in a fresh process with cold caches and a seeded random generator. The benchmark replays the returned moves and verifies that the resulting cube is solved. It prints each trial's status, solve time, and solution length, then summarizes median, 95th-percentile, and maximum solve times and minimum, median, and maximum move counts.

Solve timing excludes imports, scrambling, and verification; the timeout covers the entire worker process. Timeouts, errors, and incorrect solutions are reported separately and cause a nonzero exit status. Summary statistics cover successful trials only; p95 uses the nearest-rank method. The optional JSON report includes all trials, Python and NumPy versions, platform, and benchmark settings. Keep those settings with any published measurements. Fixed seeds reproduce the scramble and search randomness within the same implementation and environment; elapsed times still vary.

## Python API

```python
from eigencube import (
    solved_cube, shuffle, solve, apply_move_to_cube, is_cube_solved,
)

cube = shuffle(solved_cube, iterations=100, seed=42)
solution = solve(cube)
for move in solution:
    cube = apply_move_to_cube(move, cube)
assert is_cube_solved(cube)
```

Cube states are immutable tuples. `solve` returns moves without modifying its input, and accepts an optional progress callback receiving intermediate states. The API assumes a valid cube state; the example creates one by applying legal moves.

## Development

```bash
python3 -m unittest discover tests
```

Tests cover cube and rotation invariants, move identities, search budgets, solver execution, the benchmark, and headless GUI rendering. Explainer geometry tests require its optional dependencies. See [HACKING.md](HACKING.md) for environment and headless execution instructions.

The main files are:

- [`eigencube.py`](eigencube.py): cube model, moves, and solver.
- [`eigencube_gui.py`](eigencube_gui.py): Pygame cube net and solution playback.
- [`eigencube_benchmark.py`](eigencube_benchmark.py): isolated solver trials and reports.
- [`eigencube_scanner.py`](eigencube_scanner.py): standalone webcam color-classification prototype.
- [`explainer/`](explainer/README.md): animated explanation built from the model.
- [`scripts/generate_logo.py`](scripts/generate_logo.py): logo, icons, and social preview.

## Invariants & Design Principles

- **A Functional Pearl (Maximally elegant, simple, and educational):** Eigencube is designed in the tradition of a *functional pearl*—an elegant, instructive gem where the code is an executable mathematical specification. Code clarity, linear algebra transparency, and pedagogical beauty always trump micro-optimizations or clever programming tricks.
- **Self-documenting, transparent code:** Code should read as self-explanatory prose. Reject cryptic abbreviations, single-letter domain shorthand, or dense tuple indexing. Prefer clear, descriptive names (`left`, `top`, `bottom` rather than `L`, `U`, `D`) so anyone can understand the logic without an external decoder ring.
- **Zero ambient magic:** No obscure puzzle encodings or heavyweight dependencies. Pure NumPy vector and matrix arithmetic.
- **Clarity over line count:** Keep the model and solver small and understandable. There is no hard line limit; prefer readable code over compression.
- **Headless-friendly:** GUI components decouple display initializers so importing `eigencube_gui` works seamlessly in headless CI/CD environments.
