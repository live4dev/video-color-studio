# Windows portable build

The Windows x64 distribution contains real PE launchers and a private official
CPython runtime. It is prepared on macOS by cross-compilation. **It has not been
executed on Windows.** PE inspection and checksums do not establish native runtime
acceptance. The build manifest and the archive status file retain this distinction.

The application needs no installed Python, pip, administrator account or network.
Extract the complete folder and open `Freedive Color Studio.exe`. The adjacent
`Freedive Color Studio CLI.exe` uses the same processing engine. Both launch the
private interpreter by absolute Unicode path with isolated imports and UTF-8 I/O.
No shell, PATH search, registry installation or runtime download is used.

Developer inputs are SHA-256-locked in `uv.lock` and the builder: CPython 3.13.17
embeddable x64, Zig 0.14.1 and production Windows wheels. Windows uses official
PyAV 16.1.0; macOS retains 15.1.0 for macOS 13 compatibility. The Windows video
wheel contains no SRT/OpenSSL 1.1 dependency. Its FFmpeg native libraries are
included; a system `ffmpeg` executable is unnecessary.

Stock OpenCV Windows binaries contain Intel IPP. They are provisional audit inputs
only. Release finalization requires the separately compiled, source-based OpenCV
package with IPP and FFmpeg disabled. The whole original `cv2` package is removed
before installing the verified clean extension and its loader. Unused Qt modules,
QML/database/PDF plugins and OpenCV's separate VideoIO FFmpeg plugin are omitted.
The retained PE dependency closure must resolve solely to bundled binaries and
Windows system libraries. No import table is rewritten.

Prepare a new stage after building the clean extension:

```console
python tools/build_windows_portable.py --stage "build/windows-portable/Freedive Color Studio" --opencv-manifest build/windows-opencv/manifest.json
```

Use `--offline` after the developer inputs are cached, or select an existing
verified Zig 0.14.1 executable with `--zig`. Existing stages and release files are
not overwritten. `BUILD-MANIFEST.json` records input/native/payload hashes, target
architecture, omitted binaries and the explicit native-execution status.

Collect the corresponding Windows source set and its notices, then finalize:

```console
python tools/collect_media_sources.py --windows-bundle "build/windows-portable/Freedive Color Studio" --work-dir build/windows-media-sources --output dist/Freedive_Color_Studio_Windows_Corresponding_Source.zip
python tools/build_windows_portable.py --stage "build/windows-portable/Freedive Color Studio" --finalize-only --source-marker dist/Freedive_Color_Studio_Windows_Corresponding_Source.json --output dist/Freedive_Color_Studio_Windows_x64_UNTESTED.zip
```

The source marker must explicitly include `windows-x64`, match the exact payload
hash and identify a complete checksum-verified source archive. Staged application
code/assets must match the archive's application sources. Notices are included in
the portable folder; the corresponding source archive accompanies the binary ZIP.
The ZIP is CRC-verified before atomic publication. A missing source, nonmatching
stage or unverified OpenCV package prevents release publication.

Static checks can run on macOS:

```console
python tools/check_windows_portable.py --stage "build/windows-portable/Freedive Color Studio" --output artifacts/windows-static.json --static-only
```

Native CI must extract into a folder containing spaces and Cyrillic characters,
then run on Windows:

```console
python tools/check_windows_portable.py --stage "C:/Temp/Проверка приложения/Freedive Color Studio" --output artifacts/windows-native.json --offscreen
```

The native check verifies private isolated imports, the actual clean extension,
image codecs, bundled human segmentation, GUI photo grading/export, video seeking
and paired playback, and independently audited VFR/audio export. It removes
external Python/FFmpeg from the application's PATH. Evidence is written outside
the immutable payload. Only actual successful Windows execution produces a
`PASSED` native record. The current macOS-built release remains `NOT RUN`.

Official format/runtime documentation: [CPython embeddable distribution](https://docs.python.org/3.13/using/windows.html#the-embeddable-package),
[CPython 3.13.17 files](https://www.python.org/downloads/release/python-31317/),
[Zig pinned releases](https://ziglang.org/download/index.json),
[Microsoft PE/COFF specification](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format).
