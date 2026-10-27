# AGENTS.md — Freedive Color Studio

## Mission
Build a local, cross-platform (Windows/macOS) desktop app for conservative, professional color grading of underwater freediving photography and video. Read `FREEDIVE_COLOR_STUDIO_SPEC.md` first; it is the source of truth.

## Non-negotiable constraints
- Never modify source media in place. Never synthesize people, anatomy, bubbles, objects, sun rays or background.
- Prioritize preserving identity, geometry, original camera framing and natural texture over artistic impact.
- Keep all processing offline. No API keys, cloud uploads or hidden network requests.
- Each input media file creates its own output. No collages, text or watermarks.
- Make colors accurate, not uniformly neutral: seawater is naturally blue/green, and underwater red may be irrecoverable.
- For videos, avoid per-frame grading flicker and preserve audio/timestamps.
- UI and CLI must share processing code; packages must work without Python installed.
- Test each milestone, avoid fake completion claims. If builds were not executed on an OS, say so.

## Engineering workflow
1. Read the specification and inspect the repository.
2. Create/update `DECISIONS.md` for assumptions and tradeoffs.
3. Implement milestone A fully before starting B, then C, then D.
4. Work in small cohesive commits and run tests after each meaningful change.
5. Generate real before/after previews as evidence, plus a small synthetic fixture dataset.
6. Keep `README.md`, tests, license disclosures and platform build scripts updated.
7. On completion, report implemented functionality, tests, build artifacts, limitations and reproduction steps.

## Default stack
Python 3.12+, PySide6/Qt, NumPy/OpenCV, FFmpeg/ffprobe, pytest, Ruff. Optional ONNX Runtime segmentation with reliable CPU fallback. Respect model and media library licenses.

## Definition of done
User can open the app on Windows or macOS, import photos/videos, preview natural/cinematic/vivid looks with before/after, adjust local skin and water corrections, batch-export individual files without altering originals, and inspect honest results/reports. If only part of this scope is implemented, identify the current milestone explicitly instead of declaring completion.
