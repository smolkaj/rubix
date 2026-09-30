# Working concurrently

The repository root is a clean checkout. Keep it eagerly fast-forwarded with mainline (`git fetch origin main && git merge --ff-only origin/main`) so repository inspections never reflect stale code. Work in isolated Git worktrees:

```sh
# Create a clean worktree branched from latest origin/main:
git worktree add ../rubix-<task> -b <agent>/<task>

# Inspect active worktrees:
git worktree list

# Safely clean up worktree after PR merges:
git worktree remove ../rubix-<task>
```

- `<agent>` is your short ID (e.g. `agy`, `codex`, `claude`); `<task>` is short yet descriptive.
- Never touch another agent's worktree or branch.
- One branch and PR per task.
- Never push directly to `main`; always open an upstream PR (`gh pr create`).
- Open PRs proactively and early; share the GitHub PR link with the user for review.
- Never merge PRs without explicit user approval.
- After merging a PR, consider whether your work uncovered a natural follow-up. Propose at most 1–2 concrete items, or state that the task is complete.
- For every proposal, verify the friction in the code and explicitly justify: is the value worth the added complexity? Never pad lists with speculative ideas or low-value filler.

# Visual verification & remote inspection

- The user connects remotely over `ghostty` + `mosh` + `zellij`.
- `mosh` synchronizes character cells and drops terminal graphics protocols (Kitty / Sixel). Do not expect native interactive windows to display over remote terminals, and do not expect the user to view local file artifacts or `file://` links directly.
- Rubix is a native desktop application: `rubix_gui.py` uses Pygame and `rubix_scanner.py` uses OpenCV.
- For UI inspections and visual verification:
  - Run Pygame in headless mode (`SDL_VIDEODRIVER=dummy`) or run off-screen surface rendering.
  - Export rendered frames or scanner detections to image files (e.g., `pygame.image.save()` or `cv2.imwrite()`).
  - Proactively attach visual diffs/images to PRs for browser inspection.

# Fast local iteration & verification

- Development environment:
  ```sh
  # Activate or create virtual environment
  python3 -m venv rubix_env && source rubix_env/bin/activate
  pip install -r requirements.txt
  ```
- Fast syntax and gate checks prior to review:
  ```sh
  # Compile check
  python3 -m py_compile rubix.py rubix_gui.py rubix_scanner.py
  # Run test suite
  python3 -m unittest discover tests
  ```

# Hindsight reflection

Before submitting a PR for review, pause, run the [hindsight reflection](.agents/skills/hindsight-reflection), and post the evaluation in dialogue with the user.

# Independent review loop

Every PR must pass the [independent PR review loop](.agents/skills/independent-pr-review) before merge.

# Escaped defect analysis & post-mortem

Whenever investigating or fixing a bug observed by a user or in production:

1. **Mandatory escape analysis:** Never treat a bug fix as just an isolated patch. Before declaring work complete, explicitly answer and document in the PR:
   - **Root cause:** What was the underlying conceptual, state-machine, or geometric flaw?
   - **Escape vector:** How did this reach main? Which commit/PR introduced the regression?
   - **Testing blind spot:** Why did existing checks pass when the bug was introduced?
2. **Generalize tests to catch the entire class:** Tests must aim to generalize beyond the specific bug and catch a whole class of similar bugs. Never write a test that only guards the one line or exact parameter that failed:
   - For cube rotations, faces, and solver: write round-trip transition tests (e.g., identity cycles like 4x turn, 6x sexy-move, inverse sequence cancellation).
   - For GUI and state machines: test edge-case inputs, rapid key sequences, and state consistency.
   - For CV / scanner: test color ambiguity thresholds and facelet ordering.
3. **Close the systemic gap:** The PR must introduce the preventative test or architectural invariant that would have blocked the original regression from merging. Do not declare a bug task complete until the testing blind spot itself is permanently closed.

# Philosophy & invariants

- **Simplicity above all:** Rubix is a minimalistic Rubik's cube solver (~300 lines of Python code) and visualizer. Reject accidental complexity, heavy external frameworks, or speculative abstractions.
- **Single source of truth:** The cube state representation, face notation, and move conventions must have one canonical representation across `rubix.py`, `rubix_gui.py`, and `rubix_scanner.py`.
- **Churn is free:** When a coherent simplification calls for a refactor, follow it through every affected file and call site. Diff size is not a reason to leave debt behind. The standard is a materially simpler, more understandable system.
