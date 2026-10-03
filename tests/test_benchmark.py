import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import eigencube_benchmark as benchmark


class BenchmarkTest(unittest.TestCase):
    def test_fresh_processes_reproduce_verified_solutions(self):
        for seed in (0, 1, 42):
            first = benchmark.collect_trial(seed, 2, 10)
            second = benchmark.collect_trial(seed, 2, 10)
            self.assertEqual(first["status"], "ok")
            self.assertEqual(second["status"], "ok")
            self.assertEqual(first["solution_moves"], second["solution_moves"])
            self.assertGreater(first["seconds"], 0)

    def test_verification_rejects_incomplete_solutions(self):
        with patch("eigencube.solve", return_value=()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(benchmark.run_trial(42, 2)["status"], "incorrect")

    def test_failures_are_reported_and_do_not_stop_later_trials(self):
        outcomes = [subprocess.TimeoutExpired("worker", 1),
                    subprocess.CalledProcessError(1, "worker", stderr="failed"),
                    subprocess.CompletedProcess("worker", 0, stdout="invalid JSON"),
                    subprocess.CompletedProcess("worker", 0, stdout=json.dumps(
                        {"status": "ok", "seconds": 2.0, "solution_moves": 6}))]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch("subprocess.run", side_effect=outcomes), \
                 patch("platform.platform", return_value="test-platform"), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = benchmark.main(["--seeds", "0", "1", "2", "3", "--output", str(output)])
            report = json.loads(output.read_text())
        self.assertEqual(status, 1)
        self.assertEqual([row["status"] for row in report["trials"]],
                         ["timeout", "error", "error", "ok"])
        self.assertEqual(report["summary"]["seconds"]["median"], 2.0)

    def test_summary_excludes_failures_and_preserves_the_slow_tail(self):
        results = [{"status": "ok", "seconds": seconds, "solution_moves": 2 * seconds}
                   for seconds in range(1, 21)] + [{"status": "timeout"}]
        self.assertEqual(benchmark.summarize(results), {
            "seconds": {"median": 10.5, "p95": 19, "max": 20},
            "solution_moves": {"min": 2, "median": 21.0, "max": 40}})
        self.assertIsNone(benchmark.summarize([{"status": "timeout"}]))

    def test_invalid_limits_fail_before_starting_workers(self):
        for option, value in (("--timeout", "0"), ("--timeout", "-1"),
                              ("--timeout", "nan"), ("--timeout", "inf"),
                              ("--scramble-moves", "-1")):
            with patch("subprocess.run") as worker, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    benchmark.main([option, value])
                worker.assert_not_called()
