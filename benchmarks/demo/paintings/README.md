# paintings: the "recognizing the masters" social cut

An alternative, socials-friendly cut of the face-detect.cpp recognition demo that
runs the pipeline on 10 famous **public-domain paintings** (the Mona Lisa,
Vermeer, Rembrandt, Van Gogh, the Laughing Cavalier, Napoleon and more) instead
of photos of people. Non-divisive, no privacy concerns, fully public domain.

It is NOT the repo hero (the main README uses the photo carousel). This is a
spare cut to grab when you want something shareable for socials.

## Assets (rendered, ready to post)
- `../../media/face_carousel_paintings.mp4` / `.gif` - the 16:9 carousel
- `../../media/face_carousel_paintings_square.mp4` / `.gif` - the 1:1 social cut
- `../../media/face_demo_paintings.png` - the single-photo still (Mona Lisa)

## Honest notes
- Every on-screen number is real `facedetect-cli` output. SCRFD fires on classic
  realist oil portraits with sane 5-point landmarks.
- The age and gender are the model's literal reading of a painting (labeled as
  such), which makes for some charming mis-reads (it reads Van Gogh as 76, the
  Laughing Cavalier as 74). No fabricated numbers, no speed claims.
- The paintings are PD-Art faithful reproductions from Wikimedia Commons;
  per-file provenance and the public-domain basis are in `art/sources.txt`.

## Regenerate
Run from this directory (needs a built `build-cpu` CLI + `models/buffalo_l-f16.gguf`,
a Pillow python, and ffmpeg). Outputs land in `../../media/` under the
`*_paintings*` names, so they never overwrite the photo hero:

```sh
python face_carousel.py          # 16:9 -> face_carousel_paintings.{mp4,gif}
python face_carousel_square.py   # 1:1  -> face_carousel_paintings_square.{mp4,gif}
python face_demo.py              # still -> face_demo_paintings.png
```
