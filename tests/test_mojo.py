import os
import shutil
import subprocess
import unittest

class TestMojoPearl(unittest.TestCase):
    def setUp(self):
        self.mojo_bin = shutil.which("mojo")
        if not self.mojo_bin:
            pixi_mojo = os.path.expanduser("~/.pixi/bin/mojo")
            if os.path.isfile(pixi_mojo) and os.access(pixi_mojo, os.X_OK):
                self.mojo_bin = pixi_mojo

    def test_mojo_pearl_execution(self):
        """Verify that the Mojo Functional Pearl compiles, passes all cycle tests, and solves a scramble."""
        if not self.mojo_bin:
            self.skipTest("Mojo binary not found in PATH or ~/.pixi/bin")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mojo_file = os.path.join(repo_root, "mojo", "rubix.mojo")
        self.assertTrue(os.path.isfile(mojo_file), f"Missing {mojo_file}")

        result = subprocess.run(
            [self.mojo_bin, "run", mojo_file],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, f"Mojo run failed:\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}")
        self.assertIn("All 12 elementary slice moves verified with cycle length 4.", result.stdout)
        self.assertIn("Solved: True", result.stdout)
        self.assertIn("Simulated 1000000 moves", result.stdout)
