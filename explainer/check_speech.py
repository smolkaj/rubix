"""Checks that the synthetic voice says what the captions show, when they show it.

Transcribes every narration line of the latest build with a speech recognizer and lists the
lines where it heard something else, e.g. "diagram" for "diag". Fix those through
kit.PRONUNCIATION or by rephrasing, and confirm doubtful ones by ear.

With --sync, it instead transcribes the finished film and lists captions that appear more than
0.75 seconds before or after their first words are heard, or whose words are not heard at all.

    pip install faster-whisper
    python check_speech.py [--draft] [--sync]
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys

import numpy as np

from build import BUILD, CHAPTERS, cues
from kit import SPEECH_CACHE, speech

MODEL = "small.en"

# Spellings a recognizer may legitimately choose for what the narration says.
HOMOPHONES = {"are": "r", "our": "r", "see": "c", "sea": "c", "kubelet": "cubelet",
              "kubelets": "cubelets", "hole": "whole", "axis": "axes", "encode": "in code",
              "easy": "e z"}
# Differences the recognizer keeps making in lines that sound right by ear (expected, heard).
ACCEPTED = {("it", "is"), ("its", "it"), ("theirs", "their"), ("solved", "solve")}
NUMBERS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * n for n, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty "
                                        "ninety".split()) if n >= 2}


def spelled_numbers_as_digits(tokens):
    """["twenty", "six"] -> ["26"], ["four", "hundred"] -> ["400"]; other tokens unchanged."""
    out = []
    for token in tokens:
        previous = out[-1] if out else ""
        if token in NUMBERS or token in TENS:
            value = NUMBERS[token] if token in NUMBERS else TENS[token]
            # "twenty" followed by "six" is one number; "one" followed by "zero" is two.
            if previous.isdigit() and int(previous) % 10 == 0 and int(previous) >= 20 and value < 10:
                out[-1] = str(int(previous) + value)
            else:
                out.append(str(value))
        elif token == "hundred" and previous.isdigit():
            out[-1] = str(int(previous) * 100)
        else:
            out.append(token)
    return out


def words(text):
    text = re.sub(r"(?<=\w)\.(?=\w)", " dot ", text.lower())  # A recognizer writes "v.p".
    text = re.sub(r"'s\b|'", "", re.sub(r"[-–]", " ", text))  # Possessives sound the same.
    text = re.sub(r"\b0s\b", "zeros", text)
    tokens = " ".join(HOMOPHONES.get(w, w) for w in re.findall(r"[a-z0-9]+", text)).split()
    return spelled_numbers_as_digits(tokens)


def audio(path):
    # A plain decode, which ignores gaps in an audio track's timestamps the way some players do.
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", "16000", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768


def mismatches(expected, heard):
    """Word spans that differ, ignoring differences in spacing ("cube let" vs. "cubelet")."""
    a, b = words(expected), words(heard)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        expected_span, heard_span = " ".join(a[i1:i2]) or "∅", " ".join(b[j1:j2]) or "∅"
        if op != "equal" and "".join(a[i1:i2]) != "".join(b[j1:j2]) and \
                (expected_span, heard_span) not in ACCEPTED:
            yield expected_span, heard_span


def check_sync(model, film, tolerance=0.75, window=30):  # Word times jitter up to ~0.6s.
    """Captions whose start is off from when their first words are heard, and captions whose
    first words were not heard at all within `window` seconds (both count as failures)."""
    segments, _ = model.transcribe(audio(film), beam_size=5, word_timestamps=True)
    heard = [(w.start, token) for s in segments for w in s.words for token in words(w.word)]
    captions = re.findall(r"(\d+:\d+:\d+,\d+) --> .*\n(.*)\n", film.with_suffix(".srt").read_text())
    off, unheard, cursor = 0, 0, 0
    for stamp, text in captions:
        h, m, sec = stamp.replace(",", ".").split(":")
        start = int(h) * 3600 + int(m) * 60 + float(sec)
        # Compare the opening letters loosely: "e x" may be heard as "ex", "it's" as "is".
        target = "".join(words(text))[:12]
        # Only past the previous caption's match: a short caption could otherwise match an
        # earlier occurrence of its words, and hide drift.
        matches = [(i, t) for i, (t, _) in enumerate(heard) if i >= cursor and
                   abs(t - start) < window and difflib.SequenceMatcher(None, target, "".join(
                       tok for _, tok in heard[i:i + 8])[:len(target)]).ratio() >= 0.75]
        if not matches:
            unheard += 1
            print(f"{start:7.2f}s  not heard nearby: {text!r}")
            continue
        cursor, heard_at = min(matches, key=lambda match: abs(match[1] - start))
        cursor += 1
        if abs(heard_at - start) > tolerance:
            off += 1
            print(f"{start:7.2f}s  heard {heard_at - start:+.2f}s later: {text!r}")
    print(f"{off} of {len(captions)} caption(s) off by more than {tolerance}s; "
          f"{unheard} not heard within {window}s.")
    return off + unheard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
    media = BUILD / ("draft" if args.draft else "final")
    if args.sync:
        film = BUILD / ("draft.mp4" if args.draft else "eigencube-explainer.mp4")
        sys.exit(1 if check_sync(model, film) else 0)
    # What was heard in each line's audio, so that unchanged lines aren't transcribed again. The
    # audio files are named by their content, and the recognizer is deterministic.
    memo_path = SPEECH_CACHE / f"heard-{MODEL}.json"
    memo = json.loads(memo_path.read_text()) if memo_path.exists() else {}
    flagged = 0
    for scene in CHAPTERS:
        for cue in cues(media, scene):
            path, _, _ = speech(cue["text"])
            if path.name not in memo:
                segments, _ = model.transcribe(audio(path), beam_size=5)
                memo[path.name] = " ".join(s.text.strip() for s in segments)
            heard = memo[path.name]
            diffs = list(mismatches(cue["text"], heard))
            if diffs:
                flagged += 1
                print(f"{scene}: {cue['text']!r}\n  heard: {heard!r}\n  " +
                      "; ".join(f"{a!r} -> {b!r}" for a, b in diffs))
    memo_path.write_text(json.dumps(memo, indent=0))
    print(f"{flagged} line(s) heard differently.")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
