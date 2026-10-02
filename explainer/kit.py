"""Shared building blocks for the explainer: narration and a cube rendered from eigencube.py."""

import asyncio
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from manim import *
from manim import rotation_matrix as manim_rotation_matrix  # Not to be confused with eigencube's.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import eigencube  # noqa: E402

VOICE = "en-US-BrianMultilingualNeural"
RATE = "+4%"
SPEECH_CACHE = HERE / "build" / "speech"
# Spellings that make the synthetic voice say what the captions show; check_speech.py verifies them.
PRONUNCIATION = {r"\bdiag\b": "dyag"}

HEX = {"GREEN": "#2ECC71", "BLUE": "#3B82F6", "RED": "#E74C3C",
       "ORANGE": "#FF8C1A", "WHITE": "#F5F7FA", "YELLOW": "#F7D417"}
# Sticker colors keyed by the color's basis vector, i.e. by eigencube's own encoding of colors.
STICKER = {vector: HEX[name] for vector, name in eigencube.color_names.items()}
X_COLOR, Y_COLOR, Z_COLOR = STICKER[(1, 0, 0)], STICKER[(0, 1, 0)], STICKER[(0, 0, 1)]
BASIS_COLORS = [X_COLOR, Y_COLOR, Z_COLOR]
BODY = "#15171C"
ACCENT = "#58C4DD"  # the classic 3b1b blue
POSITION = "#E040FB"  # Position vectors like c and R c: a color no sticker has.
CAPTION_TOP = -2.9  # Captions overlay the frame below this height; keep text above it.


def above_captions(mobject):
    """Lifts a screen-pinned mobject clear of the caption band."""
    return mobject.shift(max(0, CAPTION_TOP - mobject.get_bottom()[1]) * UP)


def speakable(text):
    for pattern, spoken in PRONUNCIATION.items():
        text = re.sub(pattern, spoken, text)
    return text


def speech(text):
    """Synthesizes `text` once (cached by content).

    Returns the mp3 path, its length in seconds, and the timing of each spoken word as
    (start, end, word) in seconds, which lets captions follow the voice exactly.
    """
    SPEECH_CACHE.mkdir(parents=True, exist_ok=True)
    spoken = speakable(text)
    key = hashlib.sha1(f"{VOICE}|{RATE}|{spoken}".encode()).hexdigest()[:16]
    path, words_path = SPEECH_CACHE / f"{key}.mp3", SPEECH_CACHE / f"{key}.words.json"
    if not words_path.exists():
        asyncio.run(synthesize(spoken, path, words_path))
    return path, duration(path), json.loads(words_path.read_text())


async def synthesize(spoken, path, words_path):
    import edge_tts
    audio, words = bytearray(), []
    async for chunk in edge_tts.Communicate(spoken, VOICE, rate=RATE,
                                            boundary="WordBoundary").stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":  # Offsets are in units of 100 ns.
            start = chunk["offset"] / 1e7
            words.append((start, start + chunk["duration"] / 1e7, chunk["text"]))
    write_atomically(path, bytes(audio))
    # Written last: its presence marks a complete entry.
    write_atomically(words_path, json.dumps(words).encode())


def write_atomically(path, data):
    """Writes a cache file in one step, since chapters render in parallel and may share it."""
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_bytes(data)
    partial.replace(path)


def write_wav(path, signal, rate):
    """Writes a mono signal in [-1, 1] as 16-bit audio."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((signal * 32767).astype(np.int16).tobytes())
    SPEECH_CACHE.mkdir(parents=True, exist_ok=True)
    write_atomically(path, buffer.getvalue())
    return path


def music(seconds, rate=44100):
    """A gentle piano piece to close the film, synthesized here so that nothing needs licensing.

    Broken chords over C - G/B - Am - F, two and a half seconds each, ending on a held C major
    chord; it fades in, and slowly out, over exactly `seconds`."""
    # Keyed by this function's own source as well, so that editing the music invalidates the cache.
    version = hashlib.sha1(inspect.getsource(music).encode()).hexdigest()[:8]
    path = SPEECH_CACHE / f"music-{version}-{seconds:.2f}.wav"
    if path.exists():
        return path
    signal = np.zeros(int(seconds * rate))
    hz = lambda semitones: 261.63 * 2 ** (semitones / 12)  # Semitones above middle C.

    def strike(semitones, start, length, loudness):
        """A piano-like note: harmonics that decay, the higher ones faster."""
        begin, end = int(start * rate), min(len(signal), int((start + length + 1.0) * rate))
        if begin >= end:
            return
        t = np.arange(end - begin) / rate
        tone = sum(np.sin(2 * np.pi * hz(semitones) * k * t) * np.exp(-t * (0.9 + 0.6 * k)) / k ** 1.4
                   for k in range(1, 7))
        release = np.clip((start + length + 1.0 - (start + t)) / 1.0, 0, 1)  # Damper.
        signal[begin:end] += loudness * tone * np.minimum(t / 0.004, 1) * release

    bar = 2.5
    chords = [(-24, [0, 4, 7, 12, 16]), (-25, [-1, 2, 7, 11, 14]), (-27, [-3, 0, 4, 9, 12]),
              (-31, [-7, -3, 0, 5, 9])]  # (bass, broken chord), in semitones from middle C.
    bars = max(0, int((seconds - 4.0) // bar))
    for b in range(bars):
        bass, notes = chords[b % len(chords)]
        start = b * bar
        strike(bass, start, bar, 0.5)
        strike(bass + 7, start, bar, 0.3)
        for k, note in enumerate(notes + notes[-2:0:-1][:3]):  # Up and back down, in eighths.
            strike(note, start + k * bar / 8, bar / 4, 0.32 - 0.02 * k)
    for note in (-24, -12, 4, 7, 12, 16):  # The home chord, left to ring.
        strike(note, bars * bar, seconds, 0.3)
    t = np.arange(len(signal)) / rate
    signal *= np.clip(t / 1.5, 0, 1) * np.clip((seconds - t) / 6.0, 0, 1)
    signal *= 0.5 / max(np.abs(signal).max(), 1e-9)
    return write_wav(path, signal, rate)


def click(rate=44100):
    """The soft clack of one quarter turn, for the solver's montage: a short burst of noise
    through a resonance, the way plastic sounds."""
    version = hashlib.sha1(inspect.getsource(click).encode()).hexdigest()[:8]
    path = SPEECH_CACHE / f"click-{version}.wav"
    if path.exists():
        return path
    t = np.arange(int(0.04 * rate)) / rate
    noise = np.random.default_rng(0).uniform(-1, 1, len(t))
    signal = (0.6 * noise + np.sin(2 * np.pi * 1900 * t)) * np.exp(-t / 0.006)
    return write_wav(path, 0.5 * signal / np.abs(signal).max(), rate)


def duration(path):
    """Length of an audio or video file in seconds."""
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                          "format=duration", "-of", "csv=p=0", str(path)]))


class Narrated(ThreeDScene):
    """A scene whose timeline is paced by its narration.

    `with self.voice("..."):` starts the spoken line, runs the animations in the
    block alongside it, and afterwards waits until the line has finished.
    Each line is also logged as a caption cue for the final subtitles.
    """

    def setup(self):
        # Flat colors: Cairo's lighting would make one sticker color look different on each face.
        self.renderer.camera.should_apply_shading = False
        self.cues = []
        self.collisions = []
        self.labels = []
        self.speaking_until = 0
        self.spoken_words = []
        # Chapters render in parallel; separate LaTeX work directories keep them from reading each
        # other's half-written output for the same formula.
        config.tex_dir = str(Path(config.media_dir) / "Tex" / type(self).__name__)

    @contextmanager
    def voice(self, text, pause=0.35):
        path, seconds, words = speech(text)
        start = self.renderer.time
        self.add_sound(str(path))
        self.cues.append({"start": start, "end": start + seconds, "text": text,
                          "spoken": speakable(text),
                          "words": [(start + a, start + b, w) for a, b, w in words]})
        self.speaking_until = start + seconds
        self.spoken_words = self.cues[-1]["words"]
        yield
        self.spoken_words = []  # when_said outside a line fails instead of timing against this one.
        remaining = start + seconds + pause - self.renderer.time
        if remaining > 1 / config.frame_rate:
            self.wait(remaining)

    def when_said(self, phrase, lead=0.15):
        """Waits until the current line reaches `phrase`, so the next animation lands on it."""
        target = [w.lower() for w in re.findall(r"[\w']+", speakable(phrase))]
        # The voice reports words with stray punctuation at times (": x"), which matching ignores.
        spoken = [(start, re.sub(r"[^\w']", "", w).lower()) for start, _, w in self.spoken_words]
        for i in range(len(spoken)):
            if [w for _, w in spoken[i:i + len(target)]] == target:
                remaining = spoken[i][0] - lead - self.renderer.time
                if remaining > 1 / config.frame_rate:
                    self.wait(remaining)
                return
        raise ValueError(f"{phrase!r} is not in the current line")

    def add_music_since(self, start, gain=-12):
        """Lays music under everything since time `start` (up to now), fading in and out."""
        seconds = self.renderer.time - start
        self.add_sound(str(music(seconds)), time_offset=-seconds, gain=gain)

    def turn_while_speaking(self, cube, moves, run_time=1.0):
        """Keeps turning `cube` through `moves` (cyclically) until the current line ends.

        Returns the moves made, so that they can be undone."""
        made = []
        while self.renderer.time + run_time <= self.speaking_until:
            move = moves[len(made) % len(moves)]
            self.play(cube.turn(move), run_time=run_time)
            made.append(move)
        return made

    def undo(self, cube, moves, run_time=0.35):
        for move in reversed(moves):
            self.play(cube.turn(eigencube.inverse_move(move)), run_time=run_time)

    def say(self, text):
        """Speaks a line over what is already on screen."""
        with self.voice(text):
            pass

    def play(self, *animations, **kwargs):
        # Cairo flattens mobjects that are not animated into a background and paints animated ones
        # on top of it. Every object in the 3D world therefore joins each animation (as a no-op),
        # so that all of them are depth-sorted together. This relies on cube animations targeting
        # the cube itself: an animation of a new group of cubelets would make Manim move those
        # cubelets out of the cube and into the group.
        # When only screen-pinned text changes, the static world stays one cached background.
        pinned = self.renderer.camera.fixed_in_frame_mobjects

        def in_world(animation):
            mobject = getattr(animation, "mobject", None)
            return (not isinstance(animation, Wait) and mobject is not None and mobject not in pinned
                    and not (mobject.submobjects and all(m in pinned for m in mobject.submobjects)))

        if any(in_world(a) for a in animations):
            animated = {id(getattr(a, "mobject", None)) for a in animations}
            world = [m for m in self.mobjects if id(m) not in animated and m not in pinned]
            animations = (*animations, *(Animation(m) for m in world))
        super().play(*animations, **kwargs)
        self.check_labels()

    def label_boxes(self):
        """Where each label on screen lands, as (description, lower left, upper right).

        A label is whatever was placed with hud() or facing_camera(), however many of its parts
        are on screen at the moment."""
        camera = self.renderer.camera
        on_screen = set(self.get_mobject_family_members())
        for label in self.labels:
            parts = [m for m in label.get_family() if m in on_screen and m.has_points() and
                     (m.get_fill_opacity() > 0 or m.get_stroke_opacity() > 0)]
            if not parts:
                continue
            points = np.vstack([p.points for p in parts])
            if label in camera.fixed_orientation_mobjects:
                # Such a label is drawn where its center projects to, without perspective.
                center = camera.fixed_orientation_mobjects[label]()
                points = points + (camera.project_point(center) - center)
            elif label not in camera.fixed_in_frame_mobjects:
                points = camera.project_points(points)  # A label placed in the world.
            yield (getattr(label, "tex_string", type(label).__name__),
                   points.min(axis=0), points.max(axis=0))

    def obstacles(self, samples=24):
        """What text must not cover, on screen: points along visible arrows, and the outlines of
        opaque stickers."""
        camera, points, boxes = self.renderer.camera, [], []
        for m in self.get_mobject_family_members():
            if isinstance(m, Line3D) and m.get_fill_opacity() > 0.3:
                start, end = m.get_start(), m.get_end()
                # Out to the farthest point along the arrow, so that an arrowhead counts too.
                direction = (end - start) / np.linalg.norm(end - start)
                reach = max((np.vstack([p.points for p in m.get_family() if p.has_points()])
                             - start) @ direction)
                line = start + np.linspace(0, reach, samples)[:, None] * direction
                pinned = m in camera.fixed_in_frame_mobjects
                points.append(line if pinned else camera.project_points(line))
            elif getattr(m, "is_sticker", False) and m.get_fill_opacity() > 0.9:
                corners = camera.project_points(m.points)
                boxes.append((corners.min(axis=0), corners.max(axis=0)))
        return (np.vstack(points) if points else np.zeros((0, 3))), boxes

    def check_labels(self, slack=0.03):
        """Records text that overlaps other text, the caption band, an arrow or an opaque
        sticker; the build fails on any."""
        boxes = list(self.label_boxes())
        arrow_points, stickers = self.obstacles()
        for name, low, high in boxes:
            inside = ((arrow_points[:, :2] > low[:2] + slack) &
                      (arrow_points[:, :2] < high[:2] - slack)).all(axis=1)
            if inside.any():
                self.collisions.append((self.renderer.time, name, "an arrow"))
            for low2, high2 in stickers:
                if (min(high[0], high2[0]) - max(low[0], low2[0]) > slack and
                        min(high[1], high2[1]) - max(low[1], low2[1]) > slack):
                    self.collisions.append((self.renderer.time, name, "a sticker"))
                    break
        # Text that was never registered as a label would escape this check; it may not exist.
        registered = {m for label in self.labels for m in label.get_family()}
        for m in self.get_mobject_family_members():
            if isinstance(m, (SingleStringMathTex, Text, MarkupText)) and m not in registered:
                owner = next(top for top in self.mobjects if m in top.get_family())
                self.collisions.append((self.renderer.time, getattr(owner, "tex_string",
                                        type(owner).__name__), "no label"))
        for i, (name, low, high) in enumerate(boxes):
            if low[1] < CAPTION_TOP - slack:
                self.collisions.append((self.renderer.time, name, "the captions"))
            for other, low2, high2 in boxes[i + 1:]:
                if (min(high[0], high2[0]) - max(low[0], low2[0]) > slack and
                        min(high[1], high2[1]) - max(low[1], low2[1]) > slack):
                    self.collisions.append((self.renderer.time, name, other))

    def hud(self, *mobjects, overlay=False):
        """Pins mobjects to the screen (not the 3D world) without showing them yet.

        Pinned mobjects must not overlap each other or the captions, unless they are an `overlay`
        drawn on top of something on purpose."""
        self.add_fixed_in_frame_mobjects(*mobjects)
        self.remove(*mobjects)
        if not overlay:
            self.labels += mobjects
        return mobjects[0] if len(mobjects) == 1 else mobjects

    def label(self, *mobjects):
        """Registers text placed in the world of a flat scene, so that its layout is checked."""
        self.labels += mobjects
        return mobjects[0] if len(mobjects) == 1 else mobjects

    def facing_camera(self, mobject):
        """Keeps a label placed in the 3D world turned toward the camera, without showing it yet."""
        self.add_fixed_orientation_mobjects(mobject)
        self.remove(mobject)
        self.labels.append(mobject)
        return mobject

    def axes(self):
        """The coordinate frame (x, y, z through the core), with labels turned to the camera."""
        arrows = coordinate_axes()
        labels = VGroup(*(MathTex(name, color=color, font_size=40)
                          .add_background_rectangle(color=BODY, opacity=0.8)
                          .move_to((AXIS_LENGTHS[i] + 0.3) * np.array(unit(i)))
                          for i, (name, color) in enumerate(zip("xyz", BASIS_COLORS))))
        for label in labels:
            self.facing_camera(label)
        return arrows, labels

    def show_axes(self):
        """Puts the coordinate frame on screen at once (after it has been introduced)."""
        arrows, names = self.axes()
        self.add(arrows, *names)
        return arrows, names

    def tear_down(self):
        # Chapters end on a short fade rather than a hard cut.
        if self.mobjects:
            self.play(*(FadeOut(m) for m in self.mobjects), run_time=0.6)
        cues_dir = Path(config.media_dir) / "cues"
        cues_dir.mkdir(parents=True, exist_ok=True)
        (cues_dir / f"{type(self).__name__}.json").write_text(json.dumps(self.cues, indent=1))
        (cues_dir / f"{type(self).__name__}.collisions.json").write_text(
            json.dumps(self.collisions, indent=1))


def ints(vector):
    """A numpy vector as a hashable tuple of ints, e.g. to look up its sticker color."""
    return tuple(int(v) for v in vector)


def eigencube_source(name):
    """The live source of a top-level eigencube.py definition (minus decorators), so code on screen
    cannot drift from the code it shows."""
    obj = getattr(eigencube, name)
    if callable(obj):
        return "\n".join(l for l in inspect.getsource(obj).splitlines() if not l.startswith("@"))
    (line,) = (l for l in inspect.getsource(eigencube).splitlines() if l.startswith(f"{name} ="))
    return line


def code_listing(name):
    return Code(code_string=eigencube_source(name), language="python", background="window",
                add_line_numbers=False, formatter_style="monokai")


# --- The 3D world ---------------------------------------------------------------

SPACING = 0.9  # Distance between neighboring cubelet centers.
# The x axis points almost at the camera; extra length carries its tip and label past the cube.
AXIS_LENGTHS = (4.2 * SPACING, 3.0 * SPACING, 3.0 * SPACING)


def unit(i):
    return tuple(int(i == j) for j in range(3))


def at(cubelet, scale=1.0):
    """The point in the scene at cubelet address (or direction) `cubelet`."""
    return SPACING * scale * np.array(cubelet, dtype=float)


def arrow(start, end, color, thickness=0.03, segments=12, tip=1.0):
    # Short segments along the shaft, so that each depth-sorts where it actually is.
    return Arrow3D(np.array(start, dtype=float), np.array(end, dtype=float), color=color,
                   thickness=thickness, height=0.22 * tip, base_radius=0.07 * tip,
                   resolution=(segments, 6))


class OutlinedArrow(VGroup):
    """A black outline arrow, and the arrow it outlines (drawn on top, with the same pieces)."""


def overlay_arrow(start, end, color, thickness=0.03):
    """An arrow drawn over everything in the 3D world, with a thin black outline so that it stays
    visible wherever it crosses the cube or other arrows."""
    outline = arrow(start, end, BLACK, thickness=2.4 * thickness, segments=6, tip=1.35)
    return OutlinedArrow(outline, arrow(start, end, color, thickness=thickness, segments=6)
                         ).set_shade_in_3d(False)


class Draw(Create):
    """Create, except that an outlined arrow's outline grows along with the arrow it outlines.

    Create draws pieces one after another, so it would draw the whole black outline first, and
    every arrow would enter as a black silhouette. Here each outline piece shares its time slot
    with the matching piece of the arrow."""

    def begin(self):
        members = self.mobject.family_members_with_points()
        twin = {id(piece): match
                for outlined in self.mobject.get_family() if isinstance(outlined, OutlinedArrow)
                for piece, match in zip(outlined[0].family_members_with_points(),
                                        outlined[1].family_members_with_points())}
        own_slots = [i for i, m in enumerate(members) if id(m) not in twin]
        slot_of = {id(members[i]): slot for slot, i in enumerate(own_slots)}
        self.slots = [slot_of[id(twin.get(id(m), m))] for m in members]
        self.slot_count = len(own_slots)
        super().begin()

    def interpolate_mobject(self, alpha):
        for slot, pieces in zip(self.slots, self.get_all_families_zipped()):
            self.interpolate_submobject(*pieces, self.get_sub_alpha(alpha, slot, self.slot_count))


def piercing_arrow(start, surface, end, color, thickness=0.03):
    """An arrow from inside the cube, out through its surface, to `end`.

    Cairo sorts each face of the cube by its center, so a big front face counts as nearer than the
    stretch of arrow sticking out in front of it, and would cover it. That stretch therefore draws
    over the cube, outlined (the camera always sees the arrow's way out); the stretch inside is
    depth-sorted as usual."""
    inside = Line3D(np.array(start, dtype=float), np.array(surface, dtype=float), color=color,
                    thickness=thickness, resolution=(20, 6))
    return VGroup(inside, overlay_arrow(surface, end, color, thickness))


def coordinate_axes():
    """The x, y and z axes through the core, in both directions, pointing to the positive side."""
    edge = 1.55 * SPACING  # Just outside the stickers.
    # Only the front half of the x axis is longer; the back halves stay short, behind the cube.
    return VGroup(*(piercing_arrow(-AXIS_LENGTHS[1] * np.array(unit(i)), edge * np.array(unit(i)),
                                   AXIS_LENGTHS[i] * np.array(unit(i)), color, thickness=0.016)
                    for i, color in enumerate(BASIS_COLORS)))


class Logo(Triangle):
    """A mark that stays depth-sorted above its sticker while the sticker turns.

    Cairo sorts each shape by a reference point that defaults to its bounding-box center. A
    triangle's bounding box shifts as it rotates, by more than the mark's lift off the sticker,
    so mid-turn the sticker would cover it. The mean of its points rotates with the mark instead.
    """

    def get_z_index_reference_point(self):
        return self.points.mean(axis=0)


def cubelet_mobject(cubelet):
    """A cubelet drawn at its home position `c`, stickers facing along diag(c)'s columns."""
    c = np.array(cubelet, dtype=float)
    size = SPACING * 0.96
    body = Cube(side_length=size, fill_color=BODY, fill_opacity=1,
                stroke_color=BLACK, stroke_width=1).move_to(SPACING * c)
    stickers = VGroup()
    for normal in np.diag(c).T:
        if not normal.any():
            continue
        sticker = Square(side_length=size * 0.84, fill_color=STICKER[ints(normal)],
                         fill_opacity=1, stroke_width=0,
                         shade_in_3d=True)
        # Square lies in the xy-plane facing +z; turn it to face `normal`.
        if normal[2] == 0:
            sticker.rotate(PI / 2, axis=RIGHT if normal[1] else UP)
        sticker.move_to(SPACING * c + normal * (size / 2 + 0.006))
        sticker.is_sticker = True
        stickers.add(sticker)
    if cubelet == (0, 0, 1):
        # Like the logo on a real cube's white center: it makes the center's spin visible.
        logo = Logo(fill_color=GREY_C, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        logo.scale(size * 0.3).rotate(-PI / 2)
        logo.shift(SPACING * c + OUT * (size / 2 + 0.012) - logo.get_z_index_reference_point())
        stickers.add(logo)
    group = VGroup(body, stickers)
    group.cubelet = tuple(cubelet)
    return group


class CubeMobject(VGroup):
    """The 26 visible cubelets, rendered from an eigencube cube state.

    Each cubelet is drawn at home and then transformed by its rotation matrix R
    about the origin -- exactly the encoding the video explains.
    """

    def __init__(self, state=eigencube.solved_cube, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.pieces = {}
        for cubelet, rotation in state:
            piece = cubelet_mobject(cubelet)
            piece.apply_matrix(np.array(rotation, dtype=float), about_point=ORIGIN)
            self.pieces[cubelet] = piece
            self.add(piece)

    def slice_cubelets(self, move):
        """The cubelets that eigencube's own move turns (those whose rotation it changes)."""
        turned = dict(eigencube.apply_move_to_cube(move, self.state))
        return [c for c, r in self.state if turned[c] != r]

    def turn(self, move, **kwargs):
        """Animates `move` and advances the underlying eigencube state."""
        turning = [self.pieces[c] for c in self.slice_cubelets(move)]
        self.state = eigencube.apply_move_to_cube(move, self.state)
        return Turn(self, turning, **move_angle_axis(move), **kwargs)

    def ghosted(self, keep=(), opacity=0.08):
        """Every cubelet except `keep` faded to a translucent outline (use via `cube.animate`)."""
        for cubelet, piece in self.pieces.items():
            piece.set_opacity(1 if cubelet in keep else opacity)
        return self


class Turn(Animation):
    """Rotates some of a cube's cubelets about the origin, animating the cube as a whole."""

    def __init__(self, cube, turning, angle, axis, **kwargs):
        self.turning, self.angle, self.axis = turning, angle, axis
        super().__init__(cube, **kwargs)

    def begin(self):
        self.start = [piece.copy() for piece in self.turning]
        super().begin()

    def interpolate_mobject(self, alpha):
        angle = self.rate_func(alpha) * self.angle
        for piece, start in zip(self.turning, self.start):
            piece.become(start).rotate(angle, axis=self.axis, about_point=ORIGIN)


def move_angle_axis(move):
    """The (angle, axis) whose rotation equals eigencube.rotation_matrix(move)."""
    axis, _ = move
    matrix = eigencube.rotation_matrix(move)
    unit_axis = np.abs(np.array(axis, dtype=float))
    for angle in (PI / 2, -PI / 2):
        if np.allclose(manim_rotation_matrix(angle, unit_axis), matrix):
            return {"angle": angle, "axis": unit_axis}
    raise ValueError(f"not a quarter turn: {move}")

