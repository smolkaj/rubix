"""Check the animation's geometry and the completed film against Rubix."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import wave

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


class SpeechTimingChecks(unittest.TestCase):
    def test_pauses_share_the_audio_and_caption_clock_at_different_frame_rates(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(render, "BUILD", Path(directory)):
            for pause in [0., .65, 1.8]:
                part = {"text": "Hello.", "rate": "-8%", "pitch": "+3Hz", "pause_after": pause}
                path = render.speech_path(part, "test")
                with wave.open(str(path.with_suffix(".wav")), "wb") as pcm:
                    pcm.setnchannels(1)
                    pcm.setsampwidth(2)
                    pcm.setframerate(render.SAMPLE_RATE)
                    pcm.writeframes(b"\0\0" * render.SAMPLE_RATE)
                path.with_suffix(".json").write_text(json.dumps([{"text": "Hello", "start": .1, "end": .6}]))
                for fps in [12, 24, 25, 29, 60]:
                    frames, words = render.write_beat_audio({"narration": [part, part]}, 0, "test", fps)
                    self.assertAlmostEqual(words[0]["start"], .55)
                    self.assertAlmostEqual(words[1]["start"], 1.55 + pause)
                    with wave.open(str(render.BUILD / "voice-00.wav"), "rb") as pcm:
                        duration = pcm.getnframes() / pcm.getframerate()
                    self.assertAlmostEqual(duration, frames / fps, delta=1 / render.SAMPLE_RATE)
                    self.assertGreaterEqual(duration, 2.45 + 2*pause - 1 / render.SAMPLE_RATE)

    def test_delivery_changes_invalidate_speech_but_pause_edits_reuse_it(self):
        part = {"text": "A discovery!", "rate": "+2%", "pitch": "+5Hz", "pause_after": 1.}
        baseline = render.speech_path(part, "voice-a")
        self.assertEqual(baseline, render.speech_path({**part, "pause_after": 2.}, "voice-a"))
        for name, value in [("text", "Another discovery!"), ("rate", "-10%"), ("pitch", "+0Hz")]:
            self.assertNotEqual(baseline, render.speech_path({**part, name: value}, "voice-a"))
        self.assertNotEqual(baseline, render.speech_path(part, "voice-b"))


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
                               for checks in [GeometryChecks, CaptionChecks, SpeechTimingChecks])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    verify_film()
