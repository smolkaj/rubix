"""Fixed-seed solver trials, isolated so caches and timeouts cannot leak across runs."""

import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def run_trial(seed, scramble_moves):
    import eigencube as cube

    # Keep solver progress separate from the worker's JSON result.
    with redirect_stdout(sys.stderr):
        scrambled = cube.shuffle(cube.solved_cube, iterations=scramble_moves, seed=seed)
        started = time.perf_counter()
        solution = cube.solve(scrambled)
        seconds = time.perf_counter() - started
        final = scrambled
        for move in solution:
            final = cube.apply_move_to_cube(move, final)
    return {"seconds": seconds, "solution_moves": len(solution),
            "status": "ok" if cube.is_cube_solved(final) else "incorrect"}


def collect_trial(seed, scramble_moves, timeout):
    command = [sys.executable, str(Path(__file__).resolve()), "--trial", str(seed),
               "--scramble-moves", str(scramble_moves)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=True)
        result = json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        result = {"status": "timeout"}
    except subprocess.CalledProcessError as error:
        result = {"status": "error", "detail": error.stderr[-2000:]}
    except (ValueError, TypeError) as error:
        result = {"status": "error", "detail": str(error)}
    return {"seed": seed, **result}


def summarize(results):
    successful = [row for row in results if row["status"] == "ok"]
    if not successful:
        return None
    seconds = sorted(row["seconds"] for row in successful)
    lengths = [row["solution_moves"] for row in successful]
    return {"seconds": {"median": statistics.median(seconds),
                        "p95": seconds[math.ceil(0.95 * len(seconds)) - 1],
                        "max": max(seconds)},
            "solution_moves": {"min": min(lengths), "median": statistics.median(lengths),
                               "max": max(lengths)}}


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("scramble moves must be nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--scramble-moves", type=nonnegative_int, default=100)
    parser.add_argument("--timeout", type=positive_seconds, default=60,
                        help="maximum seconds per worker process (default: 60)")
    parser.add_argument("--output", type=Path, help="save metadata, trials, and summary as JSON")
    parser.add_argument("--trial", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.trial is not None:
        print(json.dumps(run_trial(args.trial, args.scramble_moves)))
        return 0

    import numpy

    report = {"python": platform.python_version(), "numpy": numpy.__version__,
              "platform": platform.platform(), "scramble_moves": args.scramble_moves,
              "timeout_seconds": args.timeout, "trials": []}
    print(" seed  status       solve seconds  solution moves", flush=True)
    for seed in args.seeds:
        row = collect_trial(seed, args.scramble_moves, args.timeout)
        report["trials"].append(row)
        seconds = f'{row["seconds"]:.3f}' if "seconds" in row else "-"
        print(f'{seed:5}  {row["status"]:10}  {seconds:>13}  {row.get("solution_moves", "-"):>14}',
              flush=True)
        if "detail" in row:
            print(row["detail"], file=sys.stderr)
    report["summary"] = summarize(report["trials"])
    successes = sum(row["status"] == "ok" for row in report["trials"])
    print(f'Verified: {successes}/{len(report["trials"])}. Summary covers successful trials only.')
    if report["summary"]:
        print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if successes == len(report["trials"]) else 1


if __name__ == "__main__":
    sys.exit(main())
