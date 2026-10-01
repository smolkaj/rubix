"""Render an original narrated explanation using Rubix as the geometry authority."""

import argparse
import asyncio
import hashlib
import json
import math
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rubix

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
STORY = json.loads((HERE / "storyboard.json").read_text())
WIDTH, HEIGHT = 1280, 720
BACKGROUND = "#101722"
INK = "#edf2f7"
MUTED = "#9caec5"
ACCENT = "#62cef0"
PALETTE = {
    "GREEN": "#42cf91", "RED": "#f16b79", "WHITE": "#f0f3fa",
    "BLUE": "#568eff", "ORANGE": "#ffad59", "YELLOW": "#f1d16a",
}
AXIS_COLORS = [PALETTE[rubix.color_names[tuple(v)]] for v in np.eye(3, dtype=int)]
CAMERA = np.array([7., -5., 5.])
CAMERA /= np.linalg.norm(CAMERA)
RIGHT = np.cross([0., 0., 1.], CAMERA)
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(CAMERA, RIGHT)
EDGE = (1, 0, 1)
TOP = ((0, 0, 1), 1)
LEFT = ((0, -1, 0), 1)
TOP_MATRIX = rubix.rotation_matrix(TOP)
IDENTITY = np.eye(3)
FONT_CACHE = {}
SAMPLE_RATE = 48000
ACTS = ["ADDRESS", "COLORS", "MOTION"]


def font(size, bold=False, math_text=False):
    key = (size, bold, math_text)
    if key not in FONT_CACHE:
        name = "Roboto-Bold.ttf" if bold else "Roboto-Regular.ttf"
        path = HERE / "fonts" / "DejaVuSans.ttf" if math_text else ROOT / "fonts" / name
        FONT_CACHE[key] = ImageFont.truetype(str(path), size)
    return FONT_CACHE[key]


def wrap(text, text_font, width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and text_font.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def mathematical_text(draw, point, text, size, fill):
    """Unicode has subscript x and y, but no z; typeset that subscript explicitly."""
    x, y = point
    pieces = text.split("e_z")
    regular = font(size, math_text=True)
    for index, piece in enumerate(pieces):
        draw.text((x, y), piece, font=regular, fill=fill)
        x += regular.getlength(piece)
        if index < len(pieces) - 1:
            draw.text((x, y), "e", font=regular, fill=fill)
            x += regular.getlength("e")
            small = font(round(size * .65), math_text=True)
            draw.text((x, y + size * .48), "z", font=small, fill=fill)
            x += small.getlength("z")


def punctuated_words(text, boundaries):
    """Use speech timestamps, but preserve the storyboard's spelling and punctuation."""
    tokens = text.split()
    normalize = lambda word: re.sub(r"\W", "", word).lower()
    if len(tokens) != len(boundaries) or any(
        normalize(token) != normalize(boundary["text"])
        for token, boundary in zip(tokens, boundaries)
    ):
        raise ValueError("Speech word boundaries do not align with the narration text")
    return [{**boundary, "text": token} for token, boundary in zip(tokens, boundaries)]


def narration_text(beat):
    return " ".join(part["text"] for part in beat["narration"])


def speech_request(part, voice):
    return {"text": part["text"], "voice": voice,
            "rate": part["rate"], "pitch": part["pitch"], "boundary": "WordBoundary"}


def speech_path(part, voice):
    key = hashlib.sha256(json.dumps(speech_request(part, voice), sort_keys=True).encode()).hexdigest()
    return BUILD / f"speech-{key}.mp3"


async def make_audio(voice):
    import edge_tts
    semaphore = asyncio.Semaphore(3)

    async def segment(part):
        audio = speech_path(part, voice)
        timing = audio.with_suffix(".json")
        if audio.exists() and timing.exists():
            return
        async with semaphore:
            boundaries = []
            temporary = audio.with_suffix(".part.mp3")
            with temporary.open("wb") as output:
                async for chunk in edge_tts.Communicate(**speech_request(part, voice)).stream():
                    if chunk["type"] == "audio":
                        output.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        boundaries.append({
                            "start": chunk["offset"] / 1e7,
                            "end": (chunk["offset"] + chunk["duration"]) / 1e7,
                            "text": chunk["text"],
                        })
            if not boundaries:
                raise RuntimeError("Narration arrived without caption timing")
            temporary.replace(audio)
            temporary_timing = timing.with_suffix(".part.json")
            temporary_timing.write_text(json.dumps(boundaries))
            temporary_timing.replace(timing)
            print(f"Narrated: {part['text'][:65]}", flush=True)

    # Identical phrases share one synthesis and one content-addressed cache entry.
    parts = {speech_path(part, voice): part for beat in STORY for part in beat["narration"]}
    await asyncio.gather(*(segment(part) for part in parts.values()))


def write_beat_audio(beat, index, voice, fps):
    """Assemble PCM and word timestamps on the same sample clock, including pauses."""
    words, sample_count = [], 0
    destination = BUILD / f"voice-{index:02}.wav"
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)

        def silence(seconds):
            nonlocal sample_count
            samples = round(seconds * SAMPLE_RATE)
            output.writeframesraw(b"\0\0" * samples)
            sample_count += samples

        silence(.45)
        for part in beat["narration"]:
            audio = speech_path(part, voice)
            pcm = audio.with_suffix(".wav")
            if not pcm.exists():
                temporary_pcm = pcm.with_suffix(".part.wav")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio),
                                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(temporary_pcm)], check=True)
                temporary_pcm.replace(pcm)
            boundaries = punctuated_words(part["text"], json.loads(audio.with_suffix(".json").read_text()))
            offset = sample_count / SAMPLE_RATE
            words.extend({**word, "start": offset + word["start"], "end": offset + word["end"]}
                         for word in boundaries)
            with wave.open(str(pcm), "rb") as source:
                assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, SAMPLE_RATE)
                samples = source.getnframes()
                output.writeframesraw(source.readframes(samples))
                sample_count += samples
            silence(part["pause_after"])
        frames = math.ceil(sample_count / SAMPLE_RATE * fps)
        silence(frames / fps - sample_count / SAMPLE_RATE)
    return frames, words


def prepare_timeline(fps, voice):
    timeline, captions = [], []
    cursor = 0
    for index, beat in enumerate(STORY):
        frames, words = write_beat_audio(beat, index, voice, fps)
        group = []
        for word in words:
            group.append(word)
            if len(" ".join(w["text"] for w in group)) >= 105 or word["text"].endswith((".", "?", "!")):
                captions.append({"start": cursor + group[0]["start"],
                                 "end": cursor + group[-1]["end"] + .12,
                                 "text": " ".join(w["text"] for w in group)})
                group = []
        if group:
            captions.append({"start": cursor + group[0]["start"],
                             "end": cursor + group[-1]["end"] + .12,
                             "text": " ".join(w["text"] for w in group)})
        cue_times = {}
        for name, keyword in beat.get("cues", {}).items():
            matches = [w["start"] for w in words if w["text"].strip(".,?!:;").lower() == keyword.lower()]
            if not matches:
                raise ValueError(f"Missing narration cue {keyword!r} in {beat['title']}")
            cue_times[name] = matches[0]
        timeline.append({**beat, "start": cursor, "duration": frames / fps,
                         "frames": frames, "cue_times": cue_times})
        cursor += frames / fps
    (BUILD / "timeline.json").write_text(json.dumps(timeline, indent=2))
    for previous, following in zip(captions, captions[1:]):
        previous["end"] = min(previous["end"], following["start"])
    (BUILD / "captions.json").write_text(json.dumps(captions))
    playlist = BUILD / "audio.txt"
    playlist.write_text("".join(f"file 'voice-{i:02}.wav'\n" for i in range(len(STORY))))
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(playlist), "-af", "loudnorm=I=-18:TP=-2:LRA=11",
        "-c:a", "aac", "-b:a", "128k", str(BUILD / "narration.m4a"),
    ], check=True)
    write_documents(timeline, captions, voice)
    return timeline, captions


def timestamp(seconds, separator=","):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"


def write_documents(timeline, captions, voice):
    (HERE / "rubix-encoding.srt").write_text("\n".join(
        f"{i + 1}\n{timestamp(c['start'])} --> {timestamp(c['end'])}\n{c['text']}\n"
        for i, c in enumerate(captions)
    ))
    (HERE / "transcript.md").write_text(
        "# A cube made of transformations\n\n" + "\n\n".join(
            f"## {timestamp(b['start'], '.')[:-4]} — {b['title']}\n\n{narration_text(b)}"
            for b in timeline
        ) + f"\n\nNarration is synthetic (Microsoft Edge, {voice}). "
        "Original script and animation; no 3Blue1Brown footage, music, or voice.\n\n"
        "Recommended: [Essence of Linear Algebra by Grant Sanderson / 3Blue1Brown]"
        "(https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).\n"
    )


def ease(t):
    t = min(1., max(0., t))
    return t * t * (3 - 2 * t)


def partial_rotation(move, fraction):
    """Interpolate a proper rotation; endpoints come from the actual move matrix."""
    matrix = rubix.rotation_matrix(move)
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
                     matrix[1, 0] - matrix[0, 1]]) / 2
    angle = ease(fraction) * math.pi / 2
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                      [-axis[1], axis[0], 0.]])
    return IDENTITY + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def animated_rotation(home, stored, move, fraction):
    rotation = np.array(stored)
    if move and np.dot(move[0], rotation @ home) > 0:
        rotation = partial_rotation(move, fraction) @ rotation
    return rotation


def project(vector, center=(330, 350), scale=110):
    return (center[0] + scale * np.dot(vector, RIGHT),
            center[1] - scale * np.dot(vector, UP))


def arrow(draw, start, end, color, width=4, label=None):
    start, end = np.array(start), np.array(end)
    draw.line([tuple(start), tuple(end)], fill=color, width=width)
    delta = end - start
    if np.linalg.norm(delta) < 1:
        return
    delta /= np.linalg.norm(delta)
    perpendicular = np.array([-delta[1], delta[0]])
    draw.polygon([tuple(end), tuple(end - 13 * delta + 5 * perpendicular),
                  tuple(end - 13 * delta - 5 * perpendicular)], fill=color)
    if label:
        draw.text(tuple(end + [8, -12]), label, font=font(24, True), fill=color)


def axes(draw, rotation=IDENTITY, scale=110, length=2.2, center=(330, 350), muted=False):
    for axis, color, label in zip(np.eye(3), AXIS_COLORS, ["+x", "+y", "+z"]):
        arrow(draw, project(np.zeros(3), center, scale),
              project(rotation @ axis * length, center, scale), "#516176" if muted else color, label=label)


def cube(draw, state=rubix.solved_cube, move=None, fraction=0., highlight=None,
         isolate=None, normals=False):
    polygons = []
    for home, stored in state:
        if isolate is not None and home != isolate:
            continue
        rotation = animated_rotation(home, stored, move, fraction)
        position = np.zeros(3) if isolate else rotation @ home
        scale = 175 if isolate else 110
        center = (320, 355)
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for sign in [-1, 1]:
                normal = np.eye(3)[axis] * sign
                current_normal = rotation @ normal
                if np.dot(current_normal, CAMERA) <= 0:
                    continue
                vertices = []
                for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    vertex = .46 * normal + .46 * a * np.eye(3)[others[0]] + .46 * b * np.eye(3)[others[1]]
                    vertices.append(position + rotation @ vertex)
                colored = home[axis] == sign
                color = PALETTE[rubix.color_names[tuple(normal.astype(int))]] if colored else "#273143"
                if highlight and not highlight(home, stored):
                    rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                    color = tuple(round(v * .24 + 18) for v in rgb)
                polygons.append((np.dot(np.mean(vertices, axis=0), CAMERA),
                                 [project(v, center, scale) for v in vertices], color))
    for _, vertices, color in sorted(polygons, key=lambda p: p[0]):
        draw.polygon(vertices, fill=color, outline=BACKGROUND, width=3)
    if normals and isolate:
        stored = next(r for c, r in state if c == isolate)
        rotation = animated_rotation(isolate, stored, move, fraction)
        for axis, value in enumerate(isolate):
            if value:
                normal = np.eye(3)[axis] * value
                color = PALETTE[rubix.color_names[tuple(normal.astype(int))]]
                current = rotation @ normal
                arrow(draw, project(.52 * current, (320, 355), 175),
                      project(1.45 * current, (320, 355), 175), color,
                      label=rubix.color_names[tuple(normal.astype(int))].title())


def matrix(draw, value, origin=(810, 275), caption="", colors=AXIS_COLORS):
    x, y = origin
    cell_w, cell_h = 75, 46
    draw.line([(x + 9, y - 5), (x, y - 5), (x, y + 133), (x + 9, y + 133)], fill=INK, width=3)
    draw.line([(x + 221, y - 5), (x + 230, y - 5), (x + 230, y + 133), (x + 221, y + 133)], fill=INK, width=3)
    for row in range(3):
        for col in range(3):
            draw.text((x + 24 + col * cell_w, y + row * cell_h),
                      str(int(round(value[row, col]))).replace("-", "−"),
                      font=font(32), fill=colors[col])
    if caption:
        draw.text((x, y + 150), caption, font=font(24), fill=MUTED)


def grid(draw):
    for a in range(-3, 4):
        for direction in [0, 1]:
            points = []
            for b in [-3, 3]:
                vector = [a, b, -1.55] if direction == 0 else [b, a, -1.55]
                points.append(project(vector))
            draw.line(points, fill="#1b2938", width=1)


def render_beat(beat, local, caption=""):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    act = beat["act"]
    for index, label in enumerate(ACTS, 1):
        color = ACCENT if act == index else INK if act > index else MUTED
        draw.text((42 + (index - 1) * 180, 20), f"{index}  {label}", font=font(17, True), fill=color)
        if act >= index:
            draw.line([(42 + (index - 1) * 180, 45), (180 + (index - 1) * 180, 45)], fill=color, width=2)
    draw.text((42, 61), beat["title"], font=font(37, True), fill=INK)
    draw.line([(680, 130), (680, 568)], fill="#293747", width=2)
    grid(draw)
    progress = local / beat["duration"]
    cues = beat.get("cue_times", {})
    turn_start = cues.get("turn", beat["duration"] * .22)
    turn = min(1., max(0., (local - turn_start) / 3.))
    if not beat.get("animate", True):
        turn = 0.
    visual = beat["visual"]
    edge_highlight = lambda c, r: c == EDGE
    turned = rubix.apply_move_to_cube(TOP, rubix.solved_cube)

    if visual == "roadmap":
        cube(draw, highlight=edge_highlight)
        for index, (label, detail) in enumerate([
            ("1  An address", "Which cubelet is it?"),
            ("2  Its colors", "What does the address contain?"),
            ("3  Its motion", "How do we carry it through a turn?"),
        ], 1):
            y = 200 + (index - 1) * 105
            color = ACCENT if index == act or act == 0 else INK if index < act else MUTED
            draw.text((725, y), label, font=font(29, True), fill=color)
            draw.text((725, y + 43), detail, font=font(22), fill=MUTED)
    elif visual == "scramble":
        sequence = [TOP, ((1, 0, 0), 1), LEFT, ((0, 0, -1), -1),
                    ((0, 1, 0), 1), TOP, ((-1, 0, 0), -1), LEFT]
        phase = min(3., local / max(.1, cues.get("freeze", beat["duration"] * .6)) * 3)
        index = min(2, int(phase))
        state = rubix.solved_cube
        for move in sequence[:index]:
            state = rubix.apply_move_to_cube(move, state)
        cube(draw, state, sequence[index], phase - index)
    elif visual in {"home", "turn", "position", "selection", "compose"}:
        state = turned if visual in {"selection", "compose"} else rubix.solved_cube
        move = LEFT if visual == "compose" else TOP if visual in {"turn", "position"} else None
        cube(draw, state, move, turn, highlight=edge_highlight)
        if visual in {"home", "position", "turn", "selection", "compose"}:
            rotation = np.array(next(r for c, r in state if c == EDGE))
            if move:
                rotation = partial_rotation(move, turn) @ rotation
            arrow(draw, project([0, 0, 0], (320, 355)),
                  project(rotation @ EDGE, (320, 355)), ACCENT, label="c" if visual == "home" else "p")
        draw.text((65, 536), "Home address: c = (1, 0, 1)ᵀ", font=font(24, math_text=True), fill=MUTED)
    elif visual == "axes":
        cube(draw, highlight=lambda c, r: rubix.norm1(c) == 1)
        axes(draw)
    elif visual in {"lattice", "types"}:
        kind = 3 if local >= cues.get("corner", beat["duration"] * .67) else 2 if local >= cues.get("edge", beat["duration"] * .33) else 1
        for c in sorted(rubix.vectors, key=lambda c: np.dot(c, CAMERA)):
            point = project(c)
            selected = visual == "lattice" or rubix.norm1(c) == kind
            color = ACCENT if selected and any(c) else "#3a4a60"
            radius = 7 if selected else 4
            draw.ellipse((point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius), fill=color)
        axes(draw)
        if visual == "types":
            name = ["", "Centers: 1 sticker", "Edges: 2 stickers", "Corners: 3 stickers"][kind]
            draw.text((85, 525), name, font=font(29, True), fill=INK)
    elif visual in {"basis", "exact"}:
        rotation = partial_rotation(TOP, turn)
        axes(draw, rotation, scale=145, length=1.5)
        for axis, color in zip(np.eye(3), AXIS_COLORS):
            points = [project(partial_rotation(TOP, f) @ axis * 1.5, scale=145)
                      for f in np.linspace(0, turn, 30)]
            draw.line(points, fill=color, width=2)
        matrix(draw, TOP_MATRIX, caption="M · the top quarter-turn matrix")
    elif visual in {"normals", "rotate_normals"}:
        cube(draw, move=TOP, fraction=turn, isolate=EDGE, normals=True)
        if turn in (0., 1.):
            matrix(draw, TOP_MATRIX @ np.diag(EDGE) if turn == 1. else np.diag(EDGE),
                   caption="R D · current sticker directions")
        else:
            draw.text((760, 315), "The arrows turn together", font=font(26), fill=MUTED)
    elif visual in {"decompose", "diag", "sum"}:
        rotation = partial_rotation(TOP, turn) if visual == "sum" else IDENTITY
        axes(draw, length=1.7, muted=True)
        start = np.zeros(3)
        for axis, value in enumerate(EDGE):
            if not value:
                continue
            vector = rotation @ (np.eye(3)[axis] * value)
            if visual == "sum":
                reveal = ease((local - cues.get("sum", 0)) / 1.3)
                origin = reveal * start
                arrow(draw, project(origin), project(origin + vector), AXIS_COLORS[axis], width=6)
                start += vector
            else:
                keyword = "green" if axis == 0 else "white"
                reveal = ease((local - cues.get(keyword, 0)) / .8)
                arrow(draw, project([0, 0, 0]), project(reveal * vector), AXIS_COLORS[axis], width=6)
        if visual != "sum" or local >= cues.get("sum", 0):
            arrow(draw, project([0, 0, 0]), project(rotation @ EDGE), ACCENT,
                  label="p" if visual == "sum" else "c")
        if visual != "decompose" and local >= cues.get("matrix", 0) and (visual != "sum" or turn in (0., 1.)):
            matrix(draw, rotation @ np.diag(EDGE), caption="N = R D" if visual == "sum" else "D = diag(1, 0, 1)")
    elif visual == "slice":
        cube(draw, move=TOP, fraction=turn, highlight=lambda c, r: np.dot(TOP[0], np.array(r) @ c) > 0)
        draw.text((70, 530), "v · p:   −1     0     +1", font=font(28), fill=INK)
    elif visual == "code":
        cube(draw, turned, LEFT, turn, highlight=edge_highlight)
    elif visual == "cycle":
        phase = 4 * min(1., max(0., (local - turn_start) / 8.))
        index = min(3, int(phase))
        state = rubix.solved_cube
        for _ in range(index):
            state = rubix.apply_move_to_cube(TOP, state)
        cube(draw, state, TOP, phase - index)
        draw.text((85, 530), f"Quarter turn {min(4, int(phase)+1)} of 4", font=font(28), fill=INK)
    elif visual == "solved":
        cube(draw)
        matrix(draw, np.diag(EDGE), caption="R D = D · the edge is solved")
    elif visual == "center":
        cube(draw, move=TOP, fraction=turn, isolate=(0, 0, 1), normals=True)
        axes(draw, partial_rotation(TOP, turn), scale=130, length=1.0)
        matrix(draw, TOP_MATRIX @ np.diag([0, 0, 1]), caption="Only the white column matters")
    elif visual == "summary":
        cube(draw, turned, LEFT, turn)
    elif visual == "credits":
        axes(draw, partial_rotation(TOP, turn), scale=150, length=1.5)
        draw.text((65, 525), "See transformations. Understand the cube.", font=font(25), fill=INK)

    y = 152
    equation_font = font(28, math_text=True)
    if local >= cues.get("equation", 0) and visual != "roadmap":
        for line in wrap(beat["equation"], equation_font, 520):
            mathematical_text(draw, (720, y), line, 28, ACCENT)
            y += 41
    if visual == "code":
        code = ["p = rotation @ cubelet", "selected = dot(v, p) > 0", "if selected:", "    rotation = M @ rotation"]
        for i, line in enumerate(code):
            cue = ["position", "select", "compose", "compose"][i]
            color = INK if local >= cues.get(cue, 0) else "#344458"
            draw.text((715, 275 + 40*i), line, font=font(24), fill=color)
    y = 467 if visual not in {"summary", "credits"} else 310
    for note in beat["notes"]:
        for line in wrap(note, font(21, math_text=True), 520):
            mathematical_text(draw, (720, y), line, 21, MUTED)
            y += 29
        y += 9
    draw.rectangle((0, 594, WIDTH, HEIGHT), fill="#0b111a")
    if caption:
        lines = wrap(caption, font(25), WIDTH - 84)
        for i, line in enumerate(lines):
            box = draw.textbbox((0, 0), line, font=font(25))
            draw.text(((WIDTH - box[2]) / 2, 614 + i*32), line, font=font(25), fill=INK)
    draw.text((42, 693), "RUBIX / A CUBE MADE OF TRANSFORMATIONS", font=font(12, True), fill=MUTED)
    fade = min(1., local / .45, (beat["duration"] - local) / .35)
    return Image.blend(Image.new("RGB", image.size, BACKGROUND), image, max(0., fade))


def render(timeline, captions, fps, output):
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
               "-i", str(BUILD / "narration.m4a"), "-c:v", "libx264", "-preset", "fast",
               "-crf", "25", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
               "-movflags", "+faststart", str(BUILD / "rendering.mp4")]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    caption_index = 0
    try:
        for index, beat in enumerate(timeline):
            print(f"Rendering {index+1}/{len(timeline)}: {beat['title']}", flush=True)
            for frame in range(beat["frames"]):
                local = frame / fps
                now = beat["start"] + local
                while caption_index < len(captions) and captions[caption_index]["end"] < now:
                    caption_index += 1
                caption = ""
                if caption_index < len(captions) and captions[caption_index]["start"] <= now:
                    caption = captions[caption_index]["text"]
                process.stdin.write(render_beat(beat, local, caption).tobytes())
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError("Video encoding failed")
    (BUILD / "rendering.mp4").replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="en-US-AndrewNeural")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--preview", action="store_true", help="Render a labeled contact sheet only")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    BUILD.mkdir(exist_ok=True)
    if args.preview:
        sheet = Image.new("RGB", (1280, math.ceil(len(STORY) / 5)*190), BACKGROUND)
        for index, beat in enumerate(STORY):
            tile = render_beat({**beat, "duration": 20}, 13).resize((256, 144))
            x, y = (index % 5)*256, (index // 5)*190
            sheet.paste(tile, (x, y))
            ImageDraw.Draw(sheet).text((x+5, y+150), f"{index+1}. {beat['visual']}", font=font(18), fill=INK)
        sheet.save(HERE / "contact-sheet.jpg", quality=90)
        return
    asyncio.run(make_audio(args.voice))
    timeline, captions = prepare_timeline(args.fps, args.voice)
    render(timeline, captions, args.fps, HERE / "rubix-encoding.mp4")


if __name__ == "__main__":
    main()
