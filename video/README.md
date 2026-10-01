# A cube made of transformations

[Download or play the narrated video](rubix-encoding.mp4) · [Transcript](transcript.md) · [Captions](rubix-encoding.srt)

An original visual explanation of Rubix's cube encoding, in English, with synthetic narration and burned-in captions. The camera keeps the green–white edge's stickers visible through the demonstrated top turn. All stored piece states, sticker colors, layer membership and move endpoints come from `rubix.py`; only the smooth paths between quarter turns are interpolated for presentation.

The story starts with the geometry and derives the algebra: fixed centers define a frame; a home coordinate identifies a piece; a rotation carries its position and sticker normals; a dot product selects the moving slice; left multiplication composes the next turn; and transformed normals define the solved condition. The example follows Rubix's exact top-move convention, rather than assuming standard face-letter notation.

The film explicitly recommends and praises [Grant Sanderson's **Essence of Linear Algebra**, from 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), which inspired Rubix. His [lesson on linear transformations](https://www.3blue1brown.com/lessons/linear-transformations/) is an especially good next step. The narration, illustrations and animation are original; no footage, music, branding or voice from that series is used. PR #18 supplied background ideas; its prose and diagram were not reused.

## Reproduce or edit

FFmpeg with `libx264` must be available on your path. Use Python 3.10 or newer in a separate environment; these production dependencies do not belong in Rubix's application requirements.

```sh
python3 -m venv /tmp/rubix-video-env
/tmp/rubix-video-env/bin/pip install -r video/requirements.txt
/tmp/rubix-video-env/bin/python video/render.py --preview
/tmp/rubix-video-env/bin/python video/render.py
```

The renderer resolves paths relative to itself, so it also works from another directory. Edit `storyboard.json` to change the narrative, on-screen equations or notes; edit `render.py` to change the geometry and layout. Optional `cues` name spoken words that trigger a turn or a piece-type reveal. Keep those words in the narration when editing: their speech timestamps synchronize the animations, and a missing cue stops the render. `--preview` renders the contact sheet without requesting audio. The full render uses Microsoft Edge's online speech service, with `en-US-AndrewNeural` at a slightly slower reading rate. It requires network access and sends the narration text to that service. `--voice` selects another supported voice; `--fps` changes the frame rate.

Intermediate narration, timing data and audio are cached in the ignored `video/build/` directory. The final MP4 is 1280 × 720 with H.264 video, AAC audio and streaming-friendly metadata. The SRT and transcript are generated from the same storyboard and narration timeline. No GUI, camera or window system is required.

Run `python video/verify.py` in the production environment after rendering. It checks interpolation and slice endpoints against all twelve Rubix moves on scrambled cubes, verifies the demonstrated sticker visibility and center symmetry, and decodes the entire MP4 to catch corruption or truncation. Mathematical labels use the bundled DejaVu Sans font, whose license is included under `video/fonts/`; prose uses Rubix's bundled Roboto font.
