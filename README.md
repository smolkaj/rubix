# Rubix

> A **Functional Pearl**: A minimalistic Rubik's Cube solver in **under 400 lines of Python**, powered by **linear algebra**, accompanied by an interactive visualizer.

<p align="center">
  <img src="img/gui-preview.png" alt="Rubix GUI Preview" width="500">
</p>

Most Rubik's cube software relies on complex combinatorial representations: 54 color stickers mapped across 6 face arrays, lookup tables for permutations, or massive pattern databases.

**Rubix takes a different path.** By framing the puzzle in discrete 3-dimensional Euclidean space, the entire physics and state of the Rubik's Cube reduces to **vectors, rotation matrices, and dot products**. The core solver ([`rubix.py`](rubix.py)) reliably solves full 100,000-move scrambles in **seconds** with zero external puzzle libraries, pattern databases, or precomputed tables (typically ~10–25s).

> **Shoutout:** The core insight for Rubix was directly sparked by Grant Sanderson’s ([3Blue1Brown](https://www.3blue1brown.com)) masterclass YouTube series, [**Essence of Linear Algebra**](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab). The series’ emphasis on geometric intuition—treating matrices as transformations of coordinate space and tracking where standard basis vectors land—inspired ditching messy combinatorial sticker permutations in favor of discrete 3D rotation matrices, vectors, and inner products. Massive praise and props to Grant for making linear algebra so intuitive, visual, and delightful!

---

## The Math Behind the Magic

### Key Terminology

<p align="center">
  <img src="img/cube-anatomy.png" alt="Rubix Anatomy: Face, Slice, Cubelet, and Facelet" width="700">
</p>

- **Face:** One of the 6 macroscopic $3 \times 3$ outer sides of the cube (Front, Back, Right, Left, Top, Bottom). Each face is a 2D exterior surface made of 9 coplanar facelets pointing along the same axis.
- **Slice:** A planar group of 9 cubelets sharing a coordinate plane (e.g. all cubelets with $x = 1$) that rotate together as a rigid 3D body. An **outer slice** is the physical 3D layer behind a **face**—turning a face physically rotates its corresponding slice.
- **Cubelet:** One of the 27 constituent $1 \times 1 \times 1$ cubes that make up the puzzle.
- **Facelet:** One of the 54 individual colored square stickers ($1 \times 1$) on the exterior of a cubelet.
- **Move:** A 90° clockwise or counterclockwise rotation of an outer slice around a coordinate axis.

---

### Step 1: Ditching the Sticker Permutation Nightmare

At first glance, representing a Rubik's cube might seem straightforward: just track 54 colored stickers in a flat list or 2D arrays. But this immediately runs into messy combinatorial bookkeeping:
- A single 90° face turn scrambles 12 edge and corner stickers across 4 adjacent faces.
- You must maintain lookup tables for how stickers permute, flip, and twist.
- The physics of 3D rigid bodies is lost in a tangle of 1D array index math.

**What if we treat the Rubik's Cube as what it physically is: a rigid 3D object in space?**

Instead of 54 independent stickers, there are really only **27 cubelets** arranged in a $3 \times 3 \times 3$ grid. Where should we anchor our coordinate frame? Right at the geometric center of the cube: $(0, 0, 0)$.

Each cubelet now has an integer coordinate $(x, y, z)$ where each coordinate is simply in $\lbrace -1, 0, 1 \rbrace$:

```
               +Z: Top (White)
                ^
                |   +Y: Right (Red)
                |  /
    -X: Back    | /
     (Blue) <----+----> +X: Front (Green)
               /|
              / |
             v  v
      -Y: Left  -Z: Bottom (Yellow)
      (Orange)
```

Each coordinate axis corresponds to an opposing pair of faces:
- **X-axis:** $+X$ points **Front** (Green) and $-X$ points **Back** (Blue).
- **Y-axis:** $+Y$ points **Right** (Red) and $-Y$ points **Left** (Orange).
- **Z-axis:** $+Z$ points **Top** (White) and $-Z$ points **Bottom** (Yellow).

---

### Step 2: Counting Facelets with the Manhattan Distance

Once we anchor the cubelets at $(x, y, z) \in \lbrace -1, 0, 1 \rbrace^3$, an immediate question arises:

> *How do we determine what kind of cubelet a coordinate represents—a corner, an edge, a center, or the hidden core?*

Do we need a hardcoded lookup dictionary? **No.** Look closely at what the coordinates mean:
- Along any axis, coordinate $0$ means the cubelet is flush inside the middle slice.
- Coordinate $+1$ or $-1$ means the cubelet extends outward to the boundary of that axis!

In other words, the absolute value $|x| \in \lbrace 0, 1 \rbrace$ acts as a binary indicator: $1$ if the cubelet touches an outer boundary, and $0$ if it is internal along that axis.

If we sum the absolute values—the **Manhattan distance** or **$L_1$ norm**—it **literally counts how many facelets the cubelet exposes to the outside world**:

$$\|c\|_1 = |x| + |y| + |z|$$

| Visible Facelets ($\|c\|_1$) | Cubelet Type | Number of Cubelets | Description |
|:---:|:---|:---:|:---|
| **0** | Interior | 1 | The hidden core mechanism; 0 facelets; never moves. |
| **1** | Center | 6 | Fixed centers; 1 facelet; rotate in place, define face colors. |
| **2** | Edge | 12 | 2 visible colored facelets. |
| **3** | Corner | 8 | 3 visible colored facelets. |

Total: $1 + 6 + 12 + 8 = 27$ cubelets.

<p align="center">
  <img src="img/cubelet-types.png" alt="Types of Cubelets" width="600">
</p>

No lookup tables, no hardcoded classifications. The physical nature of the cubelet falls directly out of its coordinate norm.

---

### Step 3: Colors *Are* the Center Cubelets

Now comes the next puzzle: *how do we represent colors?*

In traditional programs, colors are arbitrary strings (`'WHITE'`, `'GREEN'`) or integer tags (`0`, `1`). But look at the physical mechanism of a Rubik's Cube:

Slice moves twist the perimeter cubelets, but the **6 center cubelets are physically anchored to the central core spindle**. They never change their position in 3D space:
- The Green center is permanently at $(+1, 0, 0)$.
- The Red center is permanently at $(0, +1, 0)$.
- The White center is permanently at $(0, 0, +1)$.

Notice what just happened: **the 6 fixed center cubelets define the standard Cartesian basis vectors $\mathbf{e}_x, \mathbf{e}_y, \mathbf{e}_z$!**

$$\mathbf{e}_x = \text{Green Center}, \quad \mathbf{e}_y = \text{Red Center}, \quad \mathbf{e}_z = \text{White Center}$$

Why invent separate color constants when the center cubelets are already 3D vectors? We don't! **A color's identity is simply the constant position vector of its center cubelet**:

| Direction Vector | Color | Face | Defining Center Cubelet |
|:---|:---|:---|:---|
| $(+1, 0, 0)$ | Green | Front | Center at $(+1, 0, 0)$ |
| $(-1, 0, 0)$ | Blue | Back | Center at $(-1, 0, 0)$ |
| $(0, +1, 0)$ | Red | Right | Center at $(0, +1, 0)$ |
| $(0, -1, 0)$ | Orange | Left | Center at $(0, -1, 0)$ |
| $(0, 0, +1)$ | White | Top | Center at $(0, 0, +1)$ |
| $(0, 0, -1)$ | Yellow | Bottom | Center at $(0, 0, -1)$ |

With this identification, asking *"which color is on this facelet?"* becomes purely geometric: *"which center cubelet does this facelet point toward in 3D space?"*.

---

### Step 4: The $\mathrm{diag}(c)$ Magic Trick (Packing Facelets into a Matrix)

Now consider any cubelet at its canonical home position $c = (x, y, z)^T$. We know what facelets it has, but how do we represent the orientation of all its facelets at the same time?

Let's decompose $c$ along the coordinate axes:

$$c = \begin{pmatrix} x \\ y \\ z \end{pmatrix} = x \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} + y \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} + z \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} x \\ 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ y \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \\ z \end{pmatrix}$$

Look at the three component vectors:
- For the Front-Right-Top corner $c = (1, 1, 1)^T$, they are:
  - $(1, 0, 0)^T$ $\rightarrow$ Green (Front facelet normal)
  - $(0, 1, 0)^T$ $\rightarrow$ Red (Right facelet normal)
  - $(0, 0, 1)^T$ $\rightarrow$ White (Top facelet normal)
- Each non-zero component vector is an **outward unit normal** pointing directly toward one of the center cubelets!

What happens if we stack these three vectors side-by-side as the columns of a $3 \times 3$ matrix?

$$\mathrm{diag}(c) = \begin{pmatrix} x & 0 & 0 \\ 0 & y & 0 \\ 0 & 0 & z \end{pmatrix} = \begin{pmatrix} \mathbf{n}_x & \mathbf{n}_y & \mathbf{n}_z \end{pmatrix}$$

This diagonal matrix gives us an extraordinary unification:
- **Corners** (e.g. $c = (1, 1, 1)^T$): All 3 columns are non-zero unit vectors (Green, Red, White).
- **Edges** (e.g. $c = (1, 0, 1)^T$): Column 2 is $(0, 0, 0)^T$—the uncolored internal side ($y=0$) automatically drops out as a zero column!
- **Centers** (e.g. $c = (0, 0, 1)^T$): 2 columns are zero; only column 3 (White) is non-zero.
- **Interior** ($c = (0, 0, 0)^T$): All zero columns.

And notice the payoff connecting back to Step 2: **the matrix rank of $\mathrm{diag}(c)$ is exactly the Manhattan distance**:

$$\mathrm{rank}(\mathrm{diag}(c)) = \|c\|_1 = \text{number of visible facelets}$$

---

### Step 5: One-Shot Rotation & The Solved Invariant

Now, how do we track the state of the puzzle as moves are applied?

Every cubelet starts at its canonical home position $c \in \lbrace -1, 0, 1 \rbrace^3$ with an initial $3 \times 3$ identity rotation matrix $I_3$. When a sequence of 90° slice moves turns a cubelet, we accumulate those turns into a single $3 \times 3$ rotation matrix $R$.

1. **Where is the cubelet in 3D space right now?**
   $$p = R \cdot c$$
   In the solved cube, $p = I_3 \cdot c = c$.
2. **Where do all of its colored facelets point right now?**
   Because matrix multiplication distributes column-by-column:
   $$\text{Current facelet directions} = R \cdot \mathrm{diag}(c) = \begin{pmatrix} R \cdot \mathbf{n}_x & R \cdot \mathbf{n}_y & R \cdot \mathbf{n}_z \end{pmatrix}$$
   A single matrix multiply transforms **all facelets of the cubelet simultaneously**!
3. **When is a cubelet solved?**
   A cubelet is in its solved position and orientation if and only if all of its facelets point back toward their home center cubelets:
   $$R \cdot \mathrm{diag}(c) = \mathrm{diag}(c)$$

<p align="center">
  <img src="img/basis-colors-diag.png" alt="The Geometric Trick: Facelet Normals as Matrix Columns" width="700">
</p>

In [`rubix.py`](rubix.py), checking whether a cubelet is solved ([`is_cubelet_solved`](rubix.py#L204-L207)) or reading current facelet orientations ([`describe_config`](rubix.py#L75-L82)) takes just two lines:

```python
colors = np.diag(cubelet)
color_positions = rotation @ colors
return np.array_equal(colors, color_positions)
```

No sticker permutation tables, no orientation state machines—just discrete 3D linear transformations.

---

### Step 6: Slice Moves as Hyperplane Dot Products

When a move rotates an outer slice (e.g., turning the Top face clockwise), which cubelets move?

In combinatorial code, you maintain a list of cubelet indices belonging to each face. In linear algebra, a slice is just a half-space!

A move is specified by an axis unit vector $v \in \lbrace \pm e_x, \pm e_y, \pm e_z \rbrace$ and a direction $d \in \lbrace -1, +1 \rbrace$. Which cubelets lie in that slice? A single inner product:

$$v \cdot p > 0 \iff v \cdot (R \cdot c) > 0$$

If this dot product is positive, the cubelet lies in the slice. To turn it, multiply its rotation matrix by the elementary 90° rotation matrix $M$:

$$R_{\text{new}} = M \cdot R$$

Because every turn is a 90° rotation along a coordinate axis, all matrix entries in $R$ remain simple integers in $\lbrace -1, 0, 1 \rbrace$.

---

## Solver Architecture

Finding the optimal solution to an arbitrary Rubik's cube is NP-hard, and God's Number (20 moves) requires massive precomputed pattern databases.

[`rubix.py`](rubix.py) implements a **hierarchical multi-phase A\* search** that mimics human layer-by-layer reduction, staying strictly under 400 lines without precomputed databases:

```
[Scrambled Cube]
       |
       v
Phase 1: Top Layer & Centers (17 cubelets)
       |
       v
Phase 2: Middle Layer Edges (4 cubelets)
       |
       v
Phase 3: Bottom Cross (4 edge orientations & positions)
       |
       v
Phase 4: Bottom Corners (4 corner placements & twists)
       |
       v
Phase 5: Endgame Permutation Alignment
       |
       v
 [Solved Cube]
```

### Guiding the Search: Distance Estimation
A\* needs a sense of direction so it doesn't search aimlessly. Rather than storing gigabytes of precomputed lookup tables, Rubix calculates a quick distance estimate on the fly:
- **Individual cubelet distance (`min_moves_to_solved`):** Calculates how many 90° turns an isolated cubelet needs to reach its solved coordinate and orientation if no other cubelets were in the way.
- **Layer distance:** Combines the individual estimates of the active layer's target cubelets into a single distance score, pulling the search toward states where more cubelets are closer to home.
- **State caching:** Evaluates successor states with memoized transposition caching (`@functools.cache`).

### Search Performance & Randomized Restarts
- **Throughput:** Simulates ~50,000 moves/sec via vector dot products and memoized transposition caching.
- **Move-Budgeted Restarts:** To escape deep local plateaus in complex scrambles without storing massive precomputed pattern tables, A\* incorporates subtle priority randomization (`RANDOMIZE_SEARCH = True`) bounded by deterministic move budgets (`max_moves=100_000` with gentle $1.5\times$ expansion). Instead of wall-clock timers and OS signals, searches cut losses after exploring ~100k moves (~2s) and retry with fresh randomized weights without discarding prior layer progress. Typical scrambles solve in 5–25 seconds.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment (`venv`)

### Installation

```bash
# Clone the repository
git clone https://github.com/smolkaj/rubix.git
cd rubix

# Create and activate a virtual environment
python3 -m venv rubix_env
source rubix_env/bin/activate

# Install dependencies (numpy, pygame, opencv-python, Pillow)
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive Pygame Visualizer

Launch the 2D cube net visualizer:

```bash
python rubix_gui.py
```

- **Shuffle:** Click the **Shuffle** button to apply a 999-move scramble.
- **Solve:** Click the **Solve** button to run the A\* solver with a real-time progress bar.
- **Playback:**
  - `Right Arrow`: Step forward through the solution moves (hold to fast-forward at 15×).
  - `Left Arrow`: Step backward through the solution moves (rewind).

### 2. Command-Line Solver

Solve a scrambled cube directly from the terminal:

```bash
# Solve a 100,000-move scramble with default seed (42)
python rubix.py

# Solve with a specific random seed
python rubix.py 123

# Run benchmark across 100 random scrambles
python rubix.py --benchmark
```

### 3. Programmatic Python API

Use `rubix` as a lightweight puzzle simulation and solving library:

```python
from rubix import solved_cube, shuffle, solve, describe_move, is_cube_solved, apply_move_to_cube

# Create a scrambled cube
scrambled = shuffle(solved_cube, iterations=1000, seed=42)
print("Is solved?", is_cube_solved(scrambled))  # False

# Compute solution moves
solution = solve(scrambled)
print(f"Solved in {len(solution)} moves:")
for move in solution:
    print(" -", describe_move(move))

# Apply solution and verify final state
final_cube = scrambled
for move in solution:
    final_cube = apply_move_to_cube(move, final_cube)
print("Is solved?", is_cube_solved(final_cube))  # True
```

---

## Testing

Run the test suite:

```bash
python3 -m unittest discover tests
```

The test suite covers:
- Representation invariants (cubelet counts, $L_1$ norms, canonical positions).
- Rotation matrix algebra (orthogonality $R^T R = I$, $\det(R) = 1$, axis preservation).
- Scramble reproducibility and seed determinism.
- End-to-end multi-phase solver execution on scrambled states.
- Headless GUI snapshot rendering verification.

---

## Codebase Structure

```
rubix/
├── rubix.py            # Core solver & linear algebra model (< 400 lines)
├── rubix_gui.py        # Pygame GUI with animated moves & step playback
├── rubix_scanner.py    # Computer vision scanner for physical cubes (OpenCV)
├── scripts/
│   ├── generate_anatomy.py # Cube anatomy diagram generator
│   └── generate_diagram.py # Geometric intuition diagram generator
├── tests/
│   └── test_rubix.py   # Unit test suite
├── img/                # Architectural diagrams & preview snapshots
│   ├── basis-colors-diag.png
│   ├── cube.png
│   ├── cubelet-types.png
│   ├── cube-anatomy.png
│   ├── cube-in-plane.jpg
│   └── gui-preview.png
├── requirements.txt    # numpy, pygame, opencv-python, Pillow
└── README.md
```

---

## Invariants & Design Principles

- **A Functional Pearl (Maximally elegant, simple, and educational):** Rubix is designed in the tradition of a *functional pearl*—an elegant, instructive gem where the code is an executable mathematical specification. Code clarity, linear algebra transparency, and pedagogical beauty always trump micro-optimizations or clever programming tricks.
- **Self-documenting, transparent code:** Code should read as self-explanatory prose. Reject cryptic abbreviations, single-letter domain shorthand, or dense tuple indexing. Prefer clear, descriptive names (`left`, `top`, `bottom` rather than `L`, `U`, `D`) so anyone can understand the logic without an external decoder ring.
- **Zero ambient magic:** No obscure puzzle encodings or heavyweight dependencies. Pure NumPy vector and matrix arithmetic.
- **Strict code compactness:** The complete solver and domain model in `rubix.py` strictly stays below 400 lines of clean, readable Python.
- **Headless-friendly:** GUI components decouple display initializers so importing `rubix_gui` works seamlessly in headless CI/CD environments.
