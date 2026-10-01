"""Check the animation's geometry and the completed film against Rubix."""

import json
import subprocess
import unittest

import numpy as np

import render


class GeometryChecks(unittest.TestCase):
    def test_every_move_interpolates_a_proper_rotation_to_the_exact_matrix(self):
        for move in render.rubix.moves:
            np.testing.assert_allclose(render.partial_rotation(move, 0), np.eye(3), atol=1e-12)
            np.testing.assert_allclose(render.partial_rotation(move, 1),
                                       render.rubix.rotation_matrix(move), atol=1e-12)
            for fraction in np.linspace(0, 1, 17):
                rotation = render.partial_rotation(move, fraction)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_animated_slice_endpoints_match_the_model_after_scrambles(self):
        for seed in [7, 19, 43]:
            state = render.rubix.shuffle(render.rubix.solved_cube, iterations=100, seed=seed)
            for move in render.rubix.moves:
                expected = render.rubix.apply_move_to_cube(move, state)
                for (home, stored), (same_home, final) in zip(state, expected):
                    self.assertEqual(home, same_home)
                    np.testing.assert_allclose(render.animated_rotation(home, stored, move, 0),
                                               stored, atol=1e-12)
                    animated = render.animated_rotation(home, stored, move, 1)
                    np.testing.assert_allclose(animated, final, atol=1e-12)
                    normals = animated @ np.diag(home)
                    np.testing.assert_allclose(normals.sum(axis=1),
                                               render.rubix.position(home, final), atol=1e-12)

    def test_demonstrated_edge_and_center_claims(self):
        np.testing.assert_array_equal(render.TOP_MATRIX @ render.EDGE, [0, -1, 1])
        center = (0, 0, 1)
        self.assertFalse(np.array_equal(render.TOP_MATRIX, np.eye(3)))
        self.assertTrue(render.rubix.is_cubelet_solved(center, render.rubix.tupled(render.TOP_MATRIX)))
        self.assertFalse(render.rubix.is_cubelet_solved(render.EDGE, render.rubix.tupled(render.TOP_MATRIX)))
        # Both demonstrated sticker normals face the camera throughout the turn.
        for fraction in np.linspace(0, 1, 17):
            rotation = render.partial_rotation(render.TOP, fraction)
            for normal in [[1, 0, 0], [0, 0, 1]]:
                self.assertGreater(np.dot(rotation @ normal, render.CAMERA), 0)


class CaptionChecks(unittest.TestCase):
    def test_canonical_punctuation_keeps_speech_timestamps(self):
        tokens = ["Rubix", "keeps", "a", "piece's", "identity", "Does", "it", "move"]
        boundaries = [{"text": word, "start": i, "end": i + .5}
                      for i, word in enumerate(tokens)]
        words = render.punctuated_words("Rubix keeps a piece's identity. Does it move?", boundaries)
        self.assertEqual(" ".join(word["text"] for word in words),
                         "Rubix keeps a piece's identity. Does it move?")
        self.assertEqual([(word["start"], word["end"]) for word in words],
                         [(word["start"], word["end"]) for word in boundaries])

    def test_misaligned_speech_cannot_silently_shift_captions(self):
        for boundaries in [[{"text": "different"}], [{"text": "hello"}, {"text": "again"}]]:
            with self.assertRaises(ValueError):
                render.punctuated_words("Hello!", boundaries)


def verify_film():
    path = render.HERE / "rubix-encoding.mp4"
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    assert (video["width"], video["height"], video["codec_name"]) == (1280, 720, "h264")
    assert audio["codec_name"] == "aac"
    assert abs(float(video["duration"]) - float(audio["duration"])) < .2
    timeline = json.loads((render.BUILD / "timeline.json").read_text())
    captions = json.loads((render.BUILD / "captions.json").read_text())
    duration = sum(beat["duration"] for beat in timeline)
    assert abs(float(data["format"]["duration"]) - duration) < .2
    assert 0 <= captions[0]["start"] < captions[-1]["end"] < duration
    for previous, following in zip(captions, captions[1:]):
        assert previous["start"] < previous["end"] <= following["start"]
    # Decode the whole deliverable, so a truncated or corrupt MP4 fails this gate.
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                    "-f", "null", "-"], check=True)
    print(f"Verified film: {duration:.2f}s, {len(captions)} timed captions, complete A/V decode.")


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(checks)
                               for checks in [GeometryChecks, CaptionChecks])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    verify_film()
