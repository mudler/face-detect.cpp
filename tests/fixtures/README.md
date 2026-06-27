# Test fixtures

This directory holds the image fixtures and golden baselines used by the parity
tests. The files are **not committed** (they are face images / large binary
baselines); generate them locally.

Expected contents (produced by the reference pipeline):

- `face_a.jpg`, `face_b.jpg` — two photos of the same identity (for the verify
  gate) plus at least one photo of a different identity.
- `multi.jpg` — an image with several faces (for the detection golden set).
- `*.gguf` baselines — dumped by `scripts/gen_baseline.py`, holding the
  reference detector outputs, the aligned 112x112 crop, and the ArcFace
  embedding for the parity tests (see `docs/parity.md`).

Do NOT fabricate face images. Use a dataset you have the rights to (e.g. the
insightface sample images) and point the tests at the generated baselines via
the `FACEDETECT_TEST_BASELINE` / `FACEDETECT_TEST_GGUF` environment variables.
