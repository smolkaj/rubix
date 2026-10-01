# A cube made of transformations

[Download or play the narrated video](rubix-encoding.mp4) · [Transcript](transcript.md) · [Captions](rubix-encoding.srt)

An original visual explanation of Rubix's cube encoding, in English, with synthetic narration and burned-in captions. The camera keeps the green–white edge's stickers visible through the demonstrated top turn. All stored piece states, sticker colors, layer membership and move endpoints come from `rubix.py`; only the smooth paths between quarter turns are interpolated for presentation.

The opening asks what a computer needs to remember about the cube. A persistent **address → colors → motion** roadmap follows the same green–white cubelet through three discoveries. The home address names it, its components supply the sticker normals, and one rotation carries both its position and facing. The central reveal shows that adding the transformed normals recovers the current position. Slice selection, move composition and the solved condition then follow from that same representation. The ending answers the opening question: one home address and one rotation per cubelet. The example follows Rubix's exact top-move convention.

The film explicitly recommends and praises [Grant Sanderson's **Essence of Linear Algebra**, from 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), which inspired Rubix. His [lesson on linear transformations](https://www.3blue1brown.com/lessons/linear-transformations/) is an especially good next step. The narration, illustrations and animation are original; no footage, music, branding or voice from that series is used. PR #18 supplied background ideas; its prose and diagram were not reused.

## Reproduce or edit

FFmpeg with `libx264` must be available on your path. Use Python 3.10 or newer in a separate environment; these production dependencies do not belong in Rubix's application requirements.

```sh
python3 -m venv /tmp/rubix-video-env
/tmp/rubix-video-env/bin/pip install -r video/requirements.txt
/tmp/rubix-video-env/bin/python video/render.py --preview
/tmp/rubix-video-env/bin/python video/render.py
```

The renderer resolves paths relative to itself, so it also works from another directory. Edit `storyboard.json` to change the narrative, equations or notes; edit `render.py` to change the geometry and layout. Each scene's `narration` contains phrases with explicit `rate`, `pitch`, and `pause_after` values. Questions and mathematical derivations slow down; discoveries lift in pitch and pace; pauses leave time to watch and think. The phrases are assembled as PCM audio, with speech timestamps and silence on the same sample clock. Optional `cues` name spoken words that trigger motion or reveal an equation. Keep those words in the narration when editing: a missing cue stops the render. `--preview` renders the contact sheet without requesting audio.

The full render uses Microsoft Edge's online speech service, with `en-US-AndrewNeural` as the default voice. It requires network access and sends the narration text to that service. `--voice` selects another supported voice; `--fps` changes the frame rate. Text, voice, rate and pitch determine each phrase's speech cache key. Editing a pause reuses the generated voice and rebuilds the timing; editing delivery resynthesizes the affected phrase. Speech tokens must align with the script before they can become captions.

Intermediate narration, timing data and audio are cached in the ignored `video/build/` directory. The final MP4 is 1280 × 720 with H.264 video, AAC audio and streaming-friendly metadata. The SRT and transcript are generated from the same storyboard and narration timeline. No GUI, camera or window system is required.

Run `python video/verify.py` in the production environment after rendering. It checks interpolation and slice endpoints against all twelve Rubix moves on scrambled cubes, verifies the demonstrated sticker visibility and center symmetry, tests pause/timestamp consistency at several frame rates and speech cache invalidation, and decodes the entire MP4. Mathematical labels use the bundled DejaVu Sans font, whose license is included under `video/fonts/`; prose uses Rubix's bundled Roboto font.
