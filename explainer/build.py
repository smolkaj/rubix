"""Renders every chapter in parallel, then stitches them into one captioned film.

    python build.py            # 1080p30 final cut -> build/eigencube-explainer.mp4
    python build.py --draft    # fast 480p15 preview -> build/draft.mp4
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
sys.path.insert(0, str(HERE))
from kit import duration, speakable  # noqa: E402
from scenes import SCENES  # noqa: E402

CHAPTERS = [scene.__name__ for scene in SCENES]
CAPTION_WIDTH = 50  # Characters, so that a caption fits on one line in common players.
# The moment the README shows as the film's thumbnail: a chapter and the start of a spoken line.
POSTER = ("DiagTrick", "Each column is one sticker")
FINAL_FPS = 30
MIN_CAPTION_SECONDS = 0.8
MIN_CAPTION = 24  # Characters; shorter sentences are joined to a neighbor, so they don't flash by.


def render(scene, flags, media):
    log = media / f"{scene}.log"
    # Caching stays off: after replaying a cached animation, Manim silently drops any sound added
    # before the next one, which cost three narration lines their audio.
    with log.open("w") as out:
        rendered = subprocess.run([sys.executable, "-m", "manim", "render", *flags,
                                   "--disable_caching", "--media_dir", str(media),
                                   str(HERE / "scenes.py"), scene], stdout=out, stderr=out)
    if rendered.returncode:
        sys.exit(f"{scene} failed to render; see {log}")
    (video,) = media.glob(f"videos/scenes/*/{scene}.mp4")
    check_narration_audible(scene, video, media)
    check_layout(scene, media)
    return video


def check_narration_audible(scene, video, media):
    """Fails the build if any spoken line came out silent; captions alone would hide that."""
    for cue in cues(media, scene):
        stats = subprocess.run(["ffmpeg", "-v", "info", "-ss", str(cue["start"]),
                                "-t", str(cue["end"] - cue["start"]), "-i", str(video),
                                "-af", "volumedetect", "-f", "null", "-"],
                               capture_output=True, text=True, check=True).stderr
        match = re.search(r"max_volume: (-?[\d.]+) dB", stats)
        peak = float(match.group(1)) if match else float("-inf")  # No audio track at all.
        if peak < -40:
            sys.exit(f"{scene}: narration is silent ({peak} dB) at {cue['start']:.1f}s: "
                     f"{cue['text']!r}")


def check_layout(scene, media):
    """Fails the build if text overlapped other text or the caption band at any point."""
    collisions = json.loads((media / "cues" / f"{scene}.collisions.json").read_text())
    for time, name, other in collisions:
        print(f"{scene} at {time:.1f}s: {name!r} overlaps {other!r}", file=sys.stderr)
    if collisions:
        sys.exit(f"{scene}: {len(collisions)} text collision(s)")


def cues(media, scene):
    return json.loads((media / "cues" / f"{scene}.json").read_text())


def srt_time(seconds):
    millis = round(seconds * 1000)
    return "%02d:%02d:%02d,%03d" % (millis // 3600000, millis // 60000 % 60, millis // 1000 % 60,
                                    millis % 1000)


def caption_chunks(text):
    """Sentences, with long ones split at clause boundaries and short ones joined to a neighbor,
    so that each caption is comfortable to read."""
    pieces = []
    for sentence in re.findall(r"[^.!?]+[.!?]*", text):
        for clause in re.split(r"(?<=[,:;])\s+", sentence.strip()):
            # Still too long: break between words, into lines of even length. Filling each line
            # greedily would strand the clause's last word or two on a line of their own.
            while len(clause) > CAPTION_WIDTH:
                target = len(clause) / -(-len(clause) // CAPTION_WIDTH)
                spaces = [i for i, ch in enumerate(clause[:CAPTION_WIDTH + 1]) if ch == " "]
                if not spaces:
                    sys.exit(f"no place to break this caption: {clause!r}")
                cut = min(spaces, key=lambda i: abs(i - target))
                pieces.append(clause[:cut])
                clause = clause[cut + 1:]
            pieces.append(clause)
    chunks = [""]
    for piece in pieces:
        joined = f"{chunks[-1]} {piece}".strip()
        if len(joined) <= CAPTION_WIDTH and (len(chunks[-1]) < MIN_CAPTION or not
                                             re.search(r"[.!?]$", chunks[-1])):
            chunks[-1] = joined
        else:
            chunks.append(piece)
    # A short tail joins its predecessor; if the two don't fit on one line, they share it evenly,
    # breaking after punctuation where there is any, so as not to split a phrase.
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CAPTION:
        joined = f"{chunks[-2]} {chunks[-1]}"
        if len(joined) <= CAPTION_WIDTH:
            chunks[-2:] = [joined]
        else:
            spaces = [i for i, ch in enumerate(joined) if ch == " "]
            breaks = [i for i in spaces if joined[i - 1] in ",;:.!?"] or spaces
            middle = min((i for i in breaks if max(i, len(joined) - i - 1) <= CAPTION_WIDTH),
                         key=lambda i: abs(i - len(joined) / 2), default=None)
            if middle is not None:
                chunks[-2:] = [joined[:middle], joined[middle + 1:]]
    return chunks


def timed_chunks(cue):
    """Each caption chunk of a spoken line, timed by the words the voice actually spoke there.

    The voice speaks `speakable(text)`. Chunks and spoken words are both located in that text
    with a single cursor moving forward, so every word lands in the chunk that contains it."""
    spoken, bounds, position = cue["spoken"], [], 0
    chunks = caption_chunks(cue["text"])
    for chunk in chunks:
        position = spoken.index(speakable(chunk), position)
        bounds.append(position)
        position += len(speakable(chunk))
    starts, position = [None] * len(chunks), 0
    for word_start, _, word in cue["words"]:
        position = spoken.find(word, position)
        if position < 0:
            sys.exit(f"the voice reported {word!r}, which is not in the line: {cue['text']!r}")
        chunk = max(i for i, bound in enumerate(bounds) if bound <= position)
        if starts[chunk] is None:
            starts[chunk] = word_start
        position += len(word)
    for i in range(len(starts)):  # A chunk without a recognized word shares its neighbor's start.
        starts[i] = starts[i] if starts[i] is not None else (starts[i - 1] if i else cue["start"])
    ends = starts[1:] + [cue["end"]]  # A caption stays up until the next one starts.
    return list(zip(starts, ends, chunks))


def check_captions(entries):
    """Fails the build on captions that are out of order, too brief to read, or too long for one
    line."""
    for (a, b, text), (next_a, _, _) in zip(entries, entries[1:] + [(float("inf"), 0, "")]):
        if b - a < MIN_CAPTION_SECONDS or next_a < a or len(text) > CAPTION_WIDTH:
            sys.exit(f"caption at {a:.2f}s lasts {b - a:.2f}s, is out of order or is longer than "
                     f"{CAPTION_WIDTH} characters: {text!r}")


def captions(videos, media):
    entries, offset = [], 0.0
    for scene, video in zip(CHAPTERS, videos):
        for cue in cues(media, scene):
            entries += [(offset + a, offset + b, chunk) for a, b, chunk in timed_chunks(cue)]
        offset += duration(video)
    check_captions(entries)
    return "\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n"
                     for i, (a, b, text) in enumerate(entries, 1))


def check_av_lengths(film):
    """Fails the build unless the audio runs exactly as long as the picture."""
    lengths = dict(line.split(",") for line in subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0",
         str(film)], text=True).split() if line.startswith(("video", "audio")))
    if abs(float(lengths["video"]) - float(lengths["audio"])) > 0.1:
        sys.exit(f"audio ({lengths['audio']}s) and video ({lengths['video']}s) differ in length")


def contact_sheet(film, videos, media, tiles=36):
    """A grid of frames across the film, each taken as a spoken line ends: by then, the
    animations it narrates have settled, rather than being caught half-drawn."""
    ends, offset = [], 0.0
    for scene, video in zip(CHAPTERS, videos):
        ends += [offset + cue["end"] - 0.1 for cue in cues(media, scene)]
        offset += duration(video)
    frames = {round(ends[round(i * (len(ends) - 1) / (tiles - 1))] * FINAL_FPS) for i in range(tiles)}
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(film), "-vf",
                    "select='%s',scale=384:216,tile=6x6" % "+".join(f"eq(n,{n})" for n in frames),
                    "-fps_mode", "passthrough", "-frames:v", "1", "-q:v", "4",
                    str(HERE / "contact-sheet.jpg")], check=True)


def poster(film, videos, media):
    """The README's thumbnail: one frame of the film with a play button on it."""
    from PIL import Image, ImageDraw
    chapter, line = POSTER
    cue = next(c for c in cues(media, chapter) if c["text"].startswith(line))
    at = sum(duration(v) for v in videos[:CHAPTERS.index(chapter)]) + cue["start"] + 1
    frame = BUILD / "poster.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", str(film),
                    "-frames:v", "1", "-vf", "scale=1280:720", str(frame)], check=True)
    image = Image.open(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    x, y, r = 150, 600, 56  # In the empty lower left, clear of the subject.
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, 170), outline=(88, 196, 221), width=5)
    draw.polygon([(x - 19, y - 30), (x - 19, y + 30), (x + 32, y)], fill=(255, 255, 255))
    image.save(HERE / "poster.jpg", quality=88)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("scenes", nargs="*", choices=CHAPTERS, metavar="CHAPTER",
                        help=f"render only these chapters (default: all): {', '.join(CHAPTERS)}")
    args = parser.parse_args()
    flags = ["-ql"] if args.draft else ["--resolution", "1920,1080", "--frame_rate", str(FINAL_FPS)]
    media = BUILD / ("draft" if args.draft else "final")
    media.mkdir(parents=True, exist_ok=True)
    source_mtime = max(path.stat().st_mtime for path in [
        *HERE.glob("*.py"), HERE.parent / "eigencube.py", HERE.parent / "img" / "logo.svg"])

    def video(scene):
        if not args.scenes or scene in args.scenes:
            return render(scene, flags, media)
        existing = list(media.glob(f"videos/scenes/*/{scene}.mp4"))
        if not existing:
            sys.exit(f"{scene} has not been rendered at this quality yet; render it too.")
        # A final cut is what gets published, so it must not stitch in chapters (or the gates
        # they passed) from older code. Drafts may, to keep iteration fast.
        if not args.draft and existing[0].stat().st_mtime < source_mtime:
            sys.exit(f"{scene} was rendered before the film's code last changed; render it too.")
        print(f"reusing the earlier render of {scene}", file=sys.stderr)
        return existing[0]

    with ThreadPoolExecutor(min(len(CHAPTERS), os.cpu_count() or 1)) as pool:
        videos = list(pool.map(video, CHAPTERS))

    concat = media / "concat.txt"
    concat.write_text("".join(f"file '{v}'\n" for v in videos))
    film = BUILD / ("draft.mp4" if args.draft else "eigencube-explainer.mp4")
    subtitles = film.with_suffix(".srt")
    subtitles.write_text(captions(videos, media))
    picture = sum(duration(v) for v in videos)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-i", str(subtitles), "-map", "0", "-map", "1",
                    "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                    "-tune", "animation", "-pix_fmt", "yuv420p",
                    # A chapter's audio ends at its last sound, which leaves gaps between chapters.
                    # Players that ignore such gaps would run ahead of the picture, so they are
                    # filled with silence.
                    # Then normalized to the loudness usual for video online (-16 LUFS).
                    "-af", f"aresample=async=1:first_pts=0,apad=whole_dur={picture},"
                           "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000", "-c:a", "aac",
                    "-b:a", "160k",
                    "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
                    "-movflags", "+faststart", str(film)], check=True)
    check_av_lengths(film)
    if not args.draft:  # The committed contact sheet and poster track the latest final cut.
        contact_sheet(film, videos, media)
        poster(film, videos, media)
    print(f"{film}  ({duration(film) / 60:.1f} min)")


if __name__ == "__main__":
    main()
