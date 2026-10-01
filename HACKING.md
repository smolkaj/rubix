# Development & Hacking

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv rubix_env
source rubix_env/bin/activate
pip install -r requirements.txt
```

## Running the Solver & GUI

Launch the interactive Pygame GUI:
```bash
python rubix_gui.py
```

Run the command-line solver:
```bash
# Single scramble solve
python rubix.py [seed]

# 100-seed benchmark
python rubix.py --benchmark
```

## Running Tests

Run the full unit test suite:
```bash
python3 -m unittest discover tests
```

To run tests in a headless environment without an X11/Wayland display server:
```bash
SDL_VIDEODRIVER=dummy python3 -m unittest discover tests
```

## Architectural Invariants

All development must strictly adhere to the repository's canonical [Invariants & Design Principles](README.md#invariants--design-principles).

Key automated checks:
- **Strict line count invariant:** `rubix.py` must strictly remain under 400 lines of code (`wc -l rubix.py < 400`). Verify with:
  ```bash
  [ $(wc -l < rubix.py) -lt 400 ] && echo "OK" || echo "FAIL: Exceeds 400 lines"
  ```
- **Headless testability:** All GUI modules must support headless imports and execution (`init_display()` must remain lazy and respect `SDL_VIDEODRIVER=dummy`).
