# Image-only OpenCV build

The application's video backend uses PyAV and the separately bundled FFmpeg tools.
OpenCV is used for images, segmentation and mask optical flow. The upstream macOS
OpenCV wheel also loads its own video libraries; loading that wheel with PyAV caused
duplicate Objective-C AVFoundation classes. This build removes those unused
backends at compilation, while retaining the normal OpenCV modules, including
`dnn`, `video` (optical flow) and `gapi`.

Run from the project with its Python environment selected:

```sh
.venv/bin/python tools/build_image_opencv.py
```

The first command only builds. It downloads the official
`opencv-python-headless==4.14.0.94` PyPI source archive, verifies its SHA-256 from
PyPI metadata, and writes an installable wheel plus native build manifest under
`build/opencv-image-only/`. Caches, temporary files, source archive and the full
compiler log stay in that directory. No dependency or lock file changes are made.
An internet connection is needed for this developer build, never by the app.

After the build succeeds and other processes using this Python environment have
stopped, install that exact verified wheel:

```sh
.venv/bin/python tools/build_image_opencv.py --install-existing
```

For a fresh CI environment, `--install` combines build and installation. On
Windows select the equivalent environment Python (`.venv\Scripts\python.exe`).
`--work-dir` must stay inside the repository, and `--jobs` controls parallelism
(default four). A C++ compiler and the platform SDK are required; uv installs
isolated Python build dependencies.

The helper records every CMake flag. FFmpeg, AVFoundation, GStreamer, V4L,
Media Foundation, DirectShow, Qt and GTK are disabled. OpenCV 4.14's macOS
HighGUI module unconditionally retains its built-in Cocoa backend (the
`WITH_COCOA=OFF` hint is ignored upstream); the application uses PySide6 for its
interface. This does not enable AVFoundation capture. On macOS the
deployment target is 13.0 and the architecture matches the runner. JPEG, PNG,
TIFF, WebP, JPEG 2000 and their bundled codec dependencies remain enabled.
OpenEXR, AVIF and JPEG XL are outside the supported import formats and disabled.

Installation runs a fresh-process probe: JPEG/PNG/TIFF encode/decode, lossless
16-bit PNG, Farneback flow, DNN and G-API presence, video backend build flags, and
an OpenCV/PyAV import together when PyAV is present. The helper preserves stderr
and rejects duplicate native class warnings. Restart any already running app
after replacement; a loaded native library cannot be replaced inside a process.

Retain the wheel's extracted notices in `build/opencv-image-only/notices/` when
packaging. The manifest identifies the source hash, wheel hash, flags, notices
and verification. Each platform must compile and verify its own wheel; a macOS
artifact is not a Windows or Intel macOS build.
